// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements a single threaded work queue.

#include <atomic>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace {

// This class implements a work queue for single theaded clients. It spawn no
// threads and performs no synchronization. The only thread used is the host
// thread when it gets donated.
class SingleThreadedWorkQueue : public ConcurrentWorkQueue {
 public:
  SingleThreadedWorkQueue() : is_in_task_(false) {}

  std::string name() const override { return "single-threaded"; }

  void AddTask(TaskFunction work) override;
  std::optional<TaskFunction> AddBlockingTask(TaskFunction work,
                                              bool allow_queuing) override;
  void Quiesce() override;
  void Await(ArrayRef<RCReference<AsyncValue>> values) override;
  int GetParallelismLevel() const override { return 1; }

  // Given that this implementation does not spawn any threads to execute the
  // tasks it makes sense to always return `false`. However if the client uses
  // this function to check if the submitted work runs on a separate thread, to
  // decide if it is safe to do a blocking wait (e.g. on a latch), then by
  // returning false we are at risk of a dead lock.
  bool IsInWorkerThread() const override {
    return is_in_task_.load(std::memory_order_relaxed);
  }

 private:
  void Execute(TaskFunction work) {
    is_in_task_ = true;
    work();
    is_in_task_ = false;
  }

  // Current implementation uses a single mutex and condition_variable
  // for all calls to Await(). This is sub-optimal when there are many
  // outstanding calls to Await(). A more efficient, but more complex,
  // implementation can create a condition variable per call to Await()
  // and AddTask() can notify_one() on all of these condition variables.
  mutable mutex mu_;
  condition_variable cv_;
  std::vector<TaskFunction> work_items_ TFRT_GUARDED_BY(mu_);
  std::atomic<bool> is_in_task_;
};
}  // namespace

// Enqueue a block of work.
void SingleThreadedWorkQueue::AddTask(TaskFunction work) {
  mutex_lock l(mu_);
  work_items_.push_back(std::move(work));
  cv_.notify_all();
}

// We put blocking tasks and non-blocking tasks in the same queue for
// SingleThreadedWorkQueue. Note that this may cause deadlock when the blocking
// task is used to model inter-kernel dependency, e.g. one kernel is blocked
// waiting for a queue to be pushed by another kernel.
//
// Because this work queue implementation does not spawn new threads, it
// can't accept tasks that do not allow queuing.
std::optional<TaskFunction> SingleThreadedWorkQueue::AddBlockingTask(
    TaskFunction work, bool allow_queuing) {
  if (!allow_queuing) return {std::move(work)};
  mutex_lock l(mu_);
  work_items_.push_back(std::move(work));
  cv_.notify_all();
  return std::nullopt;
}

// Block until the system is quiescent (no pending work and no inflight work).
// Because we are single threaded, we *have* to use the host thread to run
// work - there is no one else to do it.
void SingleThreadedWorkQueue::Quiesce() {
  std::vector<TaskFunction> local_work_items;
  while (true) {
    // Work items can add new items to the vector, and we generally want to run
    // things in order, so make sure we explicitly pop the item off before a new
    // one is added.
    {
      mutex_lock l(mu_);
      if (work_items_.empty()) break;
      std::swap(local_work_items, work_items_);
    }
    for (auto& item : local_work_items) {
      Execute(std::move(item));
    }
    local_work_items.clear();
  }
}

void SingleThreadedWorkQueue::Await(ArrayRef<RCReference<AsyncValue>> values) {
  // We are done when values_remaining drops to zero.
  // Must be accessed while holding mu_. TFRT_GUARDED_BY does not work on local
  // variables at this point.
  int values_remaining = values.size();

  // As each value becomes available, we can decrement our counts.
  for (auto& value : values) {
    value->AndThen([this, &values_remaining]() mutable {
      mutex_lock l(mu_);
      --values_remaining;
      cv_.notify_all();
    });
  }

  // Run work items until values_remaining drops to zero.
  auto has_values = [this, &values_remaining]() mutable -> bool {
    mutex_lock l(mu_);
    return values_remaining != 0;
  };

  auto no_items_and_values_remaining = [this, &values_remaining]()
                                           TFRT_REQUIRES(mu_) -> bool {
    return values_remaining != 0 && work_items_.empty();
  };

  // Work items can add new items to the vector, and we generally want to run
  // things in order, so make sure we explicitly pop the item off before a new
  // one is added.
  std::vector<TaskFunction> local_work_items;
  int next_work_item_index = 0;

  // Run until the values get resolved.
  while (has_values()) {
    // If we've processed everything in local_work_items, then grab another
    // batch of stuff to do.  We're not done, so there must be more to do.
    if (next_work_item_index == local_work_items.size()) {
      local_work_items.clear();

      {
        mutex_lock l(mu_);
        while (no_items_and_values_remaining()) {
          cv_.wait(l);
        }
        if (values_remaining == 0) {
          break;
        }
        std::swap(local_work_items, work_items_);
      }
      next_work_item_index = 0;
    }

    // Run the next work item.
    Execute(std::move(local_work_items[next_work_item_index]));

    // Move on to the next item.
    ++next_work_item_index;
  }

  // When we resolve all the async values, we're done.  However, we could have
  // one or more elements left in the local_work_items list.  If so,
  // reconstruct the worklist in the right order (inserting local work items
  // at the head of the remaining work to do).
  if (next_work_item_index != local_work_items.size()) {
    mutex_lock l(mu_);
    work_items_.insert(work_items_.begin(),
                       std::make_move_iterator(local_work_items.begin() +
                                               next_work_item_index),
                       std::make_move_iterator(local_work_items.end()));
  }
}

std::unique_ptr<ConcurrentWorkQueue> CreateSingleThreadedWorkQueue() {
  return std::make_unique<SingleThreadedWorkQueue>();
}

}  // namespace tfrt
