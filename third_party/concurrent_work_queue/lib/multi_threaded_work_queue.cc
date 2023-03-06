// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Concurrent Work Queue implementation composed from a blocking and
// non-blocking work queues.

#include <memory>
#include <optional>
#include <thread>

#include "blocking_work_queue.h"
#include "llvm/ADT/ArrayRef.h"
#include "non_blocking_work_queue.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/support/latch.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/string_util.h"
#include "tfrt/support/thread_environment.h"

namespace tfrt {

class MultiThreadedWorkQueue : public ConcurrentWorkQueue {
 public:
  MultiThreadedWorkQueue(int num_threads, int num_blocking_threads);
  ~MultiThreadedWorkQueue() override;

  std::string name() const override {
    return StrCat("Multi-threaded C++ work queue (", num_threads_, " threads, ",
                  num_blocking_threads_, " blocking threads)");
  }

  int GetParallelismLevel() const final { return num_threads_; }

  void AddTask(TaskFunction task) final;
  std::optional<TaskFunction> AddBlockingTask(TaskFunction task,
                                              bool allow_queuing) final;
  void Quiesce() final;
  void Await(ArrayRef<RCReference<AsyncValue>> values) final;

  bool IsInWorkerThread() const final;

 private:
  const int num_threads_;
  const int num_blocking_threads_;

  std::unique_ptr<internal::QuiescingState> quiescing_state_;
  internal::NonBlockingWorkQueue<ThreadingEnvironment> non_blocking_work_queue_;
  internal::BlockingWorkQueue<ThreadingEnvironment> blocking_work_queue_;
};

MultiThreadedWorkQueue::MultiThreadedWorkQueue(int num_threads,
                                               int num_blocking_threads)
    : num_threads_(num_threads),
      num_blocking_threads_(num_blocking_threads),
      quiescing_state_(std::make_unique<internal::QuiescingState>()),
      non_blocking_work_queue_(quiescing_state_.get(), num_threads),
      blocking_work_queue_(quiescing_state_.get(), num_blocking_threads) {}

MultiThreadedWorkQueue::~MultiThreadedWorkQueue() {
  // Pending tasks in the underlying queues might submit new tasks to each other
  // during destruction.
  Quiesce();
}

void MultiThreadedWorkQueue::AddTask(TaskFunction task) {
  non_blocking_work_queue_.AddTask(std::move(task));
}

std::optional<TaskFunction> MultiThreadedWorkQueue::AddBlockingTask(
    TaskFunction task, bool allow_queuing) {
  if (allow_queuing) {
    return blocking_work_queue_.EnqueueBlockingTask(std::move(task));
  } else {
    return blocking_work_queue_.RunBlockingTask(std::move(task));
  }
}

void MultiThreadedWorkQueue::Quiesce() {
  // Turn on pending tasks counter inside both work queues.
  auto quiescing = internal::Quiescing::Start(quiescing_state_.get());

  // We call NonBlockingWorkQueue::Quiesce() first because we prefer to keep
  // caller thread busy with compute intensive tasks.
  non_blocking_work_queue_.Quiesce();

  // Wait for completion of all blocking tasks.
  blocking_work_queue_.Quiesce();

  // At this point we might still have tasks in the work queues, but because we
  // enabled quiescing mode earlier, we can rely on empty check as a loop
  // condition.
  while (quiescing.HasPendingTasks()) {
    non_blocking_work_queue_.Quiesce();
    blocking_work_queue_.Quiesce();
  }
}

void MultiThreadedWorkQueue::Await(ArrayRef<RCReference<AsyncValue>> values) {
  // We might block on a latch waiting for the completion of all tasks, and
  // this is not allowed to do inside non blocking work queue.
  non_blocking_work_queue_.CheckCallerThread("MultiThreadedWorkQueue::Await",
                                             /*is_fatal=*/false);

  // We are done when values_remaining drops to zero.
  tfrt::latch values_remaining(values.size());

  // As each value becomes available, we decrement the count.
  for (auto& value : values) {
    value->AndThen([&values_remaining]() { values_remaining.count_down(); });
  }

  // Wait until all values are resolved.
  values_remaining.wait();
}

bool MultiThreadedWorkQueue::IsInWorkerThread() const {
  return non_blocking_work_queue_.IsInWorkerThread();
}

std::unique_ptr<ConcurrentWorkQueue> CreateMultiThreadedWorkQueue(
    int num_threads, int num_blocking_threads) {
  assert(num_threads > 0 && num_blocking_threads > 0);
  return std::make_unique<MultiThreadedWorkQueue>(num_threads,
                                                  num_blocking_threads);
}

}  // namespace tfrt
