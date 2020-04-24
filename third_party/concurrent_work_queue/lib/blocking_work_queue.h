// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//===- blocking_work_queue.h - Work Queue Abstraction -----------*- C++ -*-===//
//
// This file defines the BlockingWorkQueue class for handling blocking tasks.
//
// BlockingWorkQueue supports running blocking tasks using an internal thread
// pool. A task is blocking if it spends most of its time waiting for an
// external event, e.g. IO, and consumes little CPU resources.
//
// BlockingWorkQueue manages a dynamic set of threads (up to
// max_num_pending_tasks) to run blocking tasks. When a thread finds the task
// queue to be empty, it waits for idle_wait_time before terminating itself. The
// implementation does not check for thread creation failures: It crashes the
// system when it fails to create threads.
//
// Work queue implementation is parameterized by `ThreadingEnvironment` that
// allows to provide custom thread implementation:
//
//  struct ThreadingEnvironment {
//    // Type alias for the underlying thread implementation.
//    using Thread = ...
//
//    // Starts a new thread running function `f` with arguments `args`.
//    template <class Function, class... Args>
//    std::unique_ptr<Thread> StartThread(Function&& f, Args&&... args) { ... }
//
//    // Blocks the current thread until the `thread` finishes its execution.
//    static void Join(Thread* thread) { ... }
//
//    // Separates the thread of execution from the thread object.
//    static void Detach(Thread* thread) { ... }
//
//    // Returns current thread id hash code. Must have characteristics of a
//    // good hash function and generate uniformly distributed values. Values
//    // are used as an initial seed for per-thread random number generation.
//    static uint64_t ThisThreadIdHash() {... }
//  }
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_BLOCKING_WORK_QUEUE_H_
#define TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_BLOCKING_WORK_QUEUE_H_

#include <cassert>
#include <chrono>
#include <functional>
#include <queue>

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"

namespace tfrt {
namespace internal {

template <typename ThreadingEnvironment>
class BlockingWorkQueue {
  using Thread = typename ThreadingEnvironment::Thread;

 public:
  BlockingWorkQueue(int max_num_threads, int max_num_pending_tasks,
                    std::chrono::nanoseconds idle_wait_time);
  BlockingWorkQueue(int max_num_threads,
                    std::chrono::nanoseconds idle_wait_time)
      : BlockingWorkQueue(max_num_threads, max_num_threads, idle_wait_time) {}
  ~BlockingWorkQueue();

  // Adds a blocking task to the queue. Return false if the number of pending
  // tasks is already at max_num_pending_tasks.
  Optional<TaskFunction> AddBlockingTask(TaskFunction task)
      TFRT_EXCLUDES(mutex_);

  // Quiesce blocks until all threads are exited. All the work items are
  // completed before quiesce() returns.
  void Quiesce() TFRT_EXCLUDES(mutex_);

  // Empty returns true if the number of pending tasks (including currently
  // running tasks) is zero.
  bool Empty() TFRT_EXCLUDES(mutex_);

 private:
  void DoWork(TaskFunction task);
  // Get the number of pending tasks, including waiting tasks and currently
  // running tasks.
  int GetNumPendingTasks() const TFRT_REQUIRES(mutex_);

  // Try to get the next work item with the mutex held by the lock provided in
  // the argument.
  llvm::Optional<TaskFunction> GetNextTask(mutex_lock* lock)
      TFRT_REQUIRES(mutex_);

  mutex mutex_;
  condition_variable wake_do_work_cv_;
  condition_variable thread_exited_cv_;

  bool quiescing_ TFRT_GUARDED_BY(mutex_) = false;
  int num_threads_ TFRT_GUARDED_BY(mutex_) = 0;
  int num_idle_threads_ TFRT_GUARDED_BY(mutex_) = 0;
  std::queue<TaskFunction> task_queue_ TFRT_GUARDED_BY(mutex_);

  const int max_num_threads_;
  const int max_num_pending_tasks_;
  const std::chrono::nanoseconds idle_wait_time_;

  ThreadingEnvironment threading_environment_;
};

template <typename ThreadingEnvironment>
BlockingWorkQueue<ThreadingEnvironment>::BlockingWorkQueue(
    int max_num_threads, int max_num_pending_tasks,
    std::chrono::nanoseconds idle_wait_time)
    : max_num_threads_{max_num_threads},
      max_num_pending_tasks_{max_num_pending_tasks},
      idle_wait_time_{idle_wait_time} {
  assert(max_num_pending_tasks > 0 &&
         "BlockingWorkQueue ctor max_num_pending_tasks <= 0");
}

template <typename ThreadingEnvironment>
BlockingWorkQueue<ThreadingEnvironment>::~BlockingWorkQueue() {
  Quiesce();
}

template <typename ThreadingEnvironment>
void BlockingWorkQueue<ThreadingEnvironment>::Quiesce() {
  mutex_lock lock(mutex_);

  quiescing_ = true;
  wake_do_work_cv_.notify_all();
  thread_exited_cv_.wait(
      lock, [this]() TFRT_REQUIRES(mutex_) { return num_threads_ == 0; });

  assert(task_queue_.empty());
}

template <typename ThreadingEnvironment>
bool BlockingWorkQueue<ThreadingEnvironment>::Empty() {
  mutex_lock lock(mutex_);
  return GetNumPendingTasks() == 0;
}

template <typename ThreadingEnvironment>
void BlockingWorkQueue<ThreadingEnvironment>::DoWork(TaskFunction task) {
  task();
  // Reset executed task to call destructor without holding the lock, because
  // it might be expensive. Also we want to call it before notifying quiescing
  // thread, because destructor potentially could drop the last references on
  // captured async values.
  task.reset();

  mutex_lock lock(mutex_);

  // Try to get the next task. If one is found, run it. If there is no task
  // to execute, GetNextTask will return None that converts to false.
  while (llvm::Optional<TaskFunction> task = GetNextTask(&lock)) {
    mutex_.unlock();
    // Do not hold the lock while executing and destructing the task.
    (*task)();
    task.reset();
    mutex_.lock();
  }

  // No more work to do or shutdown occurred. Exit the thread.
  --num_threads_;
  if (quiescing_) thread_exited_cv_.notify_one();
}

// Try to get the next work item with the mutex held by the lock provided in
// the argument.
template <typename ThreadingEnvironment>
llvm::Optional<TaskFunction>
BlockingWorkQueue<ThreadingEnvironment>::GetNextTask(mutex_lock* lock) {
  ++num_idle_threads_;

  const auto timeout = std::chrono::system_clock::now() + idle_wait_time_;
  wake_do_work_cv_.wait_until(*lock, timeout, [this]() TFRT_REQUIRES(mutex_) {
    return !task_queue_.empty() || quiescing_;
  });
  --num_idle_threads_;

  // Found something in the queue. Return the task.
  if (!task_queue_.empty()) {
    TaskFunction task = std::move(task_queue_.front());
    task_queue_.pop();
    return {std::move(task)};
  }

  // Shutdown occurred. Return empty optional.
  return llvm::None;
}

template <typename ThreadingEnvironment>
Optional<TaskFunction> BlockingWorkQueue<ThreadingEnvironment>::AddBlockingTask(
    TaskFunction task) {
  mutex_lock lock(mutex_);

  // Reject the task if we already reached our max pending tasks limit.
  if (GetNumPendingTasks() > max_num_pending_tasks_) return {std::move(task)};

  if (num_idle_threads_ > 0) {
    // Case 1: There are some idle threads. We enqueue the task to the queue and
    // then notify one idle thread.
    task_queue_.emplace(std::move(task));
    wake_do_work_cv_.notify_one();

  } else if (num_threads_ < max_num_threads_) {
    // Case 2: There are no idle threads and we are not at the thread limit. We
    // start a new thread to run the task.
    std::unique_ptr<Thread> thread = threading_environment_.StartThread(
        [this, task = std::move(task)]() mutable { DoWork(std::move(task)); });

    // Detach the thread. We rely on num_threads_ to detect thread exiting.
    ThreadingEnvironment::Detatch(thread.get());
    ++num_threads_;

  } else {
    // Case 3: There are no idle threads and we are at the thread limit. We
    // enqueue the task to the queue. When some thread finishes the current
    // task, they will check the queue to pick up the task.
    task_queue_.emplace(std::move(task));
  }

  return llvm::None;
}

// Get the number of pending tasks, including waiting tasks and currently
// running tasks.
template <typename ThreadingEnvironment>
int BlockingWorkQueue<ThreadingEnvironment>::GetNumPendingTasks() const {
  return task_queue_.size() + num_threads_;
}

}  // namespace internal
}  // namespace tfrt
#endif  // TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_BLOCKING_WORK_QUEUE_H_
