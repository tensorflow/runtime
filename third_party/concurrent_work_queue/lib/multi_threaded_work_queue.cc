// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//===- non_blocking_work_queue.cc -------------------------------*- C++ -*-===//
//
// Concurrent Work Queue implementation composed from a blocking and
// non-blocking work queues.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <thread>

#include "blocking_work_queue.h"
#include "environment.h"
#include "llvm/ADT/ArrayRef.h"
#include "non_blocking_work_queue.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/support/latch.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/string_util.h"

namespace tfrt {

class MultiThreadedWorkQueue : public ConcurrentWorkQueue {
  using ThreadingEnvironment = ::tfrt::internal::StdThreadingEnvironment;

 public:
  MultiThreadedWorkQueue(int num_threads, int max_blocking_work_queue_threads);
  ~MultiThreadedWorkQueue() override;

  std::string name() const override {
    return StrCat("Multi-threaded C++ work queue (", num_threads_, " threads)");
  }

  int GetParallelismLevel() const final { return num_threads_; }

  void AddTask(TaskFunction task) final;
  Optional<TaskFunction> AddBlockingTask(TaskFunction task) final;
  void Quiesce() final;
  void Await(ArrayRef<RCReference<AsyncValue>> values) final;

 private:
  const int num_threads_;

  internal::NonBlockingWorkQueue<ThreadingEnvironment> non_blocking_work_queue_;
  internal::BlockingWorkQueue<ThreadingEnvironment> blocking_work_queue_;
};

MultiThreadedWorkQueue::MultiThreadedWorkQueue(
    int num_threads, int max_blocking_work_queue_threads)
    : num_threads_(num_threads),
      non_blocking_work_queue_(num_threads),
      blocking_work_queue_(max_blocking_work_queue_threads) {}

MultiThreadedWorkQueue::~MultiThreadedWorkQueue() {
  // Pending tasks in the underlying queues might submit new tasks to each other
  // during destruction.
  Quiesce();
}

void MultiThreadedWorkQueue::AddTask(TaskFunction task) {
  non_blocking_work_queue_.AddTask(std::move(task));
}

Optional<TaskFunction> MultiThreadedWorkQueue::AddBlockingTask(
    TaskFunction task) {
  return blocking_work_queue_.AddBlockingTask(std::move(task));
}

void MultiThreadedWorkQueue::Quiesce() {
  // Turn on pending tasks counter inside blocking work queue.
  auto quiescing = blocking_work_queue_.StartQuiescing();

  // We call NonBlockingWorkQueue::Quiesce() first because we prefer to keep
  // caller thread busy with compute intensive tasks.
  non_blocking_work_queue_.Quiesce();

  // Wait for completion of all blocking tasks.
  blocking_work_queue_.Quiesce();

  // Check if tasks inside blocking queue added new non-blocking tasks.
  non_blocking_work_queue_.Quiesce();

  // At this point we might still have tasks in the blocking work queue, but
  // because we enabled quiescing mode earlier, we can rely on empty check as a
  // loop condition.
  while (!quiescing.Empty()) {
    // Wait for completion of all blocking tasks.
    blocking_work_queue_.Quiesce();

    // Wait for completion of all non-blocking tasks in case any new tasks
    // were submitted from the blocking queue.
    non_blocking_work_queue_.Quiesce();

    // At this point non blocking tasks potentially could submit new tasks to
    // the blocking queue, but because `quiescing.Empty()` provides a strong
    // emptiness guarantee, it's safe to rely on it in a loop condition.
  }
}

void MultiThreadedWorkQueue::Await(ArrayRef<RCReference<AsyncValue>> values) {
  // We might block on a latch waiting for the completion of all tasks, and
  // this is not allowed to do inside non blocking work queue.
  non_blocking_work_queue_.CheckCallerThread("MultiThreadedWorkQueue::Await");

  // We are done when values_remaining drops to zero.
  tfrt::latch values_remaining(values.size());

  // As each value becomes available, we decrement the count.
  for (auto& value : values) {
    value->AndThen([&values_remaining]() { values_remaining.count_down(); });
  }

  // Keep stealing tasks from non-blocking workers until we reach a point when
  // all async values are resolved or we could not steal any task.
  //
  // We steal pending tasks globally and potentially can steal a very expensive
  // task, that will unnecessarily delay the completion of this function.
  // Alternative is to immediately block on the latch.
  llvm::Optional<TaskFunction> task = non_blocking_work_queue_.Steal();
  while (task.hasValue() || !values_remaining.try_wait()) {
    if (task.hasValue()) (*task)();
    task = non_blocking_work_queue_.Steal();
  }

  // Wait until all values are resolved.
  values_remaining.wait();
}

std::unique_ptr<ConcurrentWorkQueue> CreateMultiThreadedWorkQueue(
    int num_threads, int num_blocking_threads) {
  assert(num_threads > 0 && num_blocking_threads > 0);
  return std::make_unique<MultiThreadedWorkQueue>(num_threads,
                                                  num_blocking_threads);
}

}  // namespace tfrt
