// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//===- blocking_work_queue.h ------------------------------------*- C++ -*-===//
//
// Work queue implementation based on non-blocking concurrency primitives
// optimized for IO and mostly blocking tasks.
//
// This work queue uses TaskQueue for storing pending tasks. Tasks executed
// in mostly FIFO order, which is optimal for IO tasks.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_BLOCKING_WORK_QUEUE_H_
#define TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_BLOCKING_WORK_QUEUE_H_

#include <atomic>

#include "llvm/Support/Compiler.h"
#include "task_queue.h"
#include "tfrt/host_context/task_function.h"
#include "work_queue_base.h"

namespace tfrt {
namespace internal {

template <typename ThreadingEnvironment>
class BlockingWorkQueue;

template <typename ThreadingEnvironmentTy>
struct WorkQueueTraits<BlockingWorkQueue<ThreadingEnvironmentTy>> {
  using ThreadingEnvironment = ThreadingEnvironmentTy;
  using Thread = typename ThreadingEnvironment::Thread;
  using Queue = ::tfrt::internal::TaskQueue;
};

template <typename ThreadingEnvironment>
class BlockingWorkQueue
    : public WorkQueueBase<BlockingWorkQueue<ThreadingEnvironment>> {
  using Base = WorkQueueBase<BlockingWorkQueue<ThreadingEnvironment>>;

  using Queue = typename Base::Queue;
  using Thread = typename Base::Thread;
  using PerThread = typename Base::PerThread;
  using ThreadData = typename Base::ThreadData;

 public:
  explicit BlockingWorkQueue(int num_threads);
  ~BlockingWorkQueue() = default;

  Optional<TaskFunction> AddBlockingTask(TaskFunction task);

  // Following two functions are required to correctly implement
  // MultiThreadedWorkQueue::Quiesce().

  // Enable pending task counter via acquiring Quiescing object.
  class Quiescing;
  Quiescing StartQuiescing() { return Quiescing(this); }

 private:
  template <typename WorkQueue>
  friend class WorkQueueBase;

  using Base::GetPerThread;
  using Base::IsNotifyParkedThreadRequired;

  using Base::coprimes_;
  using Base::event_count_;
  using Base::num_threads_;
  using Base::thread_data_;

  // If we are in quiescing mode, we can always execute submitted task in the
  // caller thread, because the system is anyway going to shutdown soon, and
  // even if we are running inside a non-blocking work queue, potential
  // context switch can't negatively impact system performance.
  void AddBlockingTaskWhileQuiescing(TaskFunction task);

  LLVM_NODISCARD Optional<TaskFunction> NextTask(Queue* queue);
  LLVM_NODISCARD Optional<TaskFunction> Steal(Queue* queue);
  LLVM_NODISCARD bool Empty(Queue* queue);

  // Blocking work queue requires strong guarantees for the Empty() method to
  // implement Quiesce() correctly in MultiThreadedWorkQueue, so we keep track
  // of the number of pending tasks that were submitted to this queue.
  //
  // We do this only if there is an active thread inside a
  // MultiThreadedWorkQueue::Quiesce() because the overhead of atomic
  // read-modify-write operation is too high for the regular workload.
  std::atomic<int64_t> num_quiescing_;
  std::atomic<int64_t> num_tasks_;

 public:
  class Quiescing {
   public:
    explicit Quiescing(BlockingWorkQueue* parent) : parent_(parent) {
      parent_->num_quiescing_.fetch_add(1, std::memory_order_relaxed);
    }

    ~Quiescing() {
      if (parent_ == nullptr) return;  // in moved-out state
      parent_->num_quiescing_.fetch_sub(1, std::memory_order_relaxed);
    }

    Quiescing(Quiescing&& other) : parent_(other.parent_) {
      other.parent_ = nullptr;
    }

    Quiescing& operator=(Quiescing&& other) {
      parent_ = other.parent_;
      other.parent_ = nullptr;
    }

    Quiescing(const Quiescing&) = delete;
    Quiescing& operator=(const Quiescing&) = delete;

    // Empty() returns true if all tasks added to the parent queue after
    // `*this` was created are completed.
    bool Empty() const {
      return parent_->num_tasks_.load(std::memory_order_relaxed) == 0;
    }

   private:
    BlockingWorkQueue* parent_;
  };
};

template <typename ThreadingEnvironment>
BlockingWorkQueue<ThreadingEnvironment>::BlockingWorkQueue(int num_threads)
    : WorkQueueBase<BlockingWorkQueue>(num_threads),
      num_quiescing_(0),
      num_tasks_(0) {}

template <typename ThreadingEnvironment>
Optional<TaskFunction> BlockingWorkQueue<ThreadingEnvironment>::AddBlockingTask(
    TaskFunction task) {
  // In quiescing mode we count the number of pending tasks, and are allowed to
  // execute tasks in the caller thread.
  if (num_quiescing_.load(std::memory_order_relaxed) > 0) {
    AddBlockingTaskWhileQuiescing(std::move(task));
    return llvm::None;
  }

  // If the worker queue is full, we will return task to the caller thread.
  llvm::Optional<TaskFunction> inline_task = {std::move(task)};

  PerThread* pt = GetPerThread();
  if (pt->parent == this) {
    // Worker thread of this pool, push onto the thread's queue.
    Queue& q = thread_data_[pt->thread_id].queue;
    inline_task = q.PushFront(std::move(*inline_task));
  } else {
    // A random free-standing thread (or worker of another pool).
    unsigned r = pt->rng();
    unsigned victim = FastReduce(r, num_threads_);
    unsigned inc = coprimes_[FastReduce(r, coprimes_.size())];

    for (unsigned i = 0; i < num_threads_ && inline_task.hasValue(); i++) {
      inline_task =
          thread_data_[victim].queue.PushFront(std::move(*inline_task));
      if ((victim += inc) >= num_threads_) victim -= num_threads_;
    }
  }

  if (inline_task.hasValue()) return inline_task;

  // Note: below we touch `*this` after making `task` available to worker
  // threads. Strictly speaking, this can lead to a racy-use-after-free.
  // Consider that Schedule is called from a thread that is neither main thread
  // nor a worker thread of this pool. Then, execution of `task` directly or
  // indirectly completes overall computations, which in turn leads to
  // destruction of this. We expect that such a scenario is prevented by the
  // program, that is, this is kept alive while any threads can potentially be
  // in Schedule.
  if (IsNotifyParkedThreadRequired()) event_count_.Notify(false);

  return llvm::None;
}

template <typename ThreadingEnvironment>
void BlockingWorkQueue<ThreadingEnvironment>::AddBlockingTaskWhileQuiescing(
    TaskFunction task) {
  // Prepare to add new pending task.
  num_tasks_.fetch_add(1, std::memory_order_relaxed);

  // Insert task into the random queue.
  PerThread* pt = GetPerThread();
  unsigned rnd = FastReduce(pt->rng(), num_threads_);
  Queue& q = thread_data_[rnd].queue;

  // Decrement the number of pending tasks after executing the original task.
  llvm::Optional<TaskFunction> inline_task =
      q.PushFront(TaskFunction([this, task = std::move(task)]() mutable {
        task();
        this->num_tasks_.fetch_sub(1, std::memory_order_relaxed);
      }));

  // We are allowed to execute tasks in the caller threads, because Quiesce
  // means that the system is going to shutdown soon.
  if (inline_task.hasValue()) {
    (*inline_task)();
  } else {
    if (IsNotifyParkedThreadRequired()) event_count_.Notify(false);
  }
}

template <typename ThreadingEnvironment>
LLVM_NODISCARD Optional<TaskFunction>
BlockingWorkQueue<ThreadingEnvironment>::NextTask(Queue* queue) {
  return queue->PopBack();
}

template <typename ThreadingEnvironment>
LLVM_NODISCARD Optional<TaskFunction>
BlockingWorkQueue<ThreadingEnvironment>::Steal(Queue* queue) {
  return queue->PopBack();
}

template <typename ThreadingEnvironment>
LLVM_NODISCARD bool BlockingWorkQueue<ThreadingEnvironment>::Empty(
    Queue* queue) {
  return queue->Empty();
}

}  // namespace internal
}  // namespace tfrt

#endif  // TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_BLOCKING_WORK_QUEUE_H_
