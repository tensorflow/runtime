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

//===- single_threaded_work_queue.cc --------------------------------------===//
//
// This file implements a single threaded work queue.
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/concurrent_work_queue.h"

namespace tfrt {
namespace {

// This class implements a work queue for single theaded clients. It spawn no
// threads and performs no synchronization. The only thread used is the host
// thread when it gets donated.
class SingleThreadedWorkQueue : public ConcurrentWorkQueue {
 public:
  SingleThreadedWorkQueue() {}

  std::string name() const override { return "single-threaded"; }

  void AddTask(TaskFunction work) override;
  Optional<TaskFunction> AddBlockingTask(TaskFunction work,
                                         bool allow_queuing) override;
  void Quiesce() override;
  void Await(ArrayRef<RCReference<AsyncValue>> values) override;
  int GetParallelismLevel() const override { return 1; }

 private:
  void DoWork(std::function<bool()> stop_predicate);

  std::vector<TaskFunction> work_items_;
};
}  // namespace

// Enqueue a block of work. Currently thread-unsafe since std::vector::push_back
// is not thread-safe, but ConcurrentWorkQueue is intended to be used in
// non-threaded environment so synchronization is not needed.
// TODO(b/137227366): Revisit this design.
void SingleThreadedWorkQueue::AddTask(TaskFunction work) {
  work_items_.push_back(std::move(work));
}

// We put blocking tasks and non-blocking tasks in the same queue for
// SingleThreadedWorkQueue. Note that this may cause deadlock when the blocking
// task is used to model inter-kernel dependency, e.g. one kernel is blocked
// waiting for a queue to be pushed by another kernel.
//
// Because this work queue implementation does not spawn new threads, it
// can't accept tasks that do not allow queuing.
Optional<TaskFunction> SingleThreadedWorkQueue::AddBlockingTask(
    TaskFunction work, bool allow_queuing) {
  assert(allow_queuing == true);
  work_items_.push_back(std::move(work));
  return llvm::None;
}

// Block until the system is quiescent (no pending work and no inflight work).
// Because we are single threaded, we *have* to use the host thread to run
// work - there is no one else to do it.
void SingleThreadedWorkQueue::Quiesce() {
  std::vector<TaskFunction> local_work_items;
  while (!work_items_.empty()) {
    // Work items can add new items to the vector, and we generally want to run
    // things in order, so make sure we explicitly pop the item off before a new
    // one is added.
    std::swap(local_work_items, work_items_);
    for (auto& item : local_work_items) {
      item();
    }
    local_work_items.clear();
  }
}

void SingleThreadedWorkQueue::Await(ArrayRef<RCReference<AsyncValue>> values) {
  // We are done when values_remaining drops to zero.
  int values_remaining = values.size();

  // As each value becomes available, we can decrement our counts.
  for (auto& value : values)
    value->AndThen([&values_remaining]() { --values_remaining; });

  // Run work items until values_remaining drops to zero.
  DoWork([&values_remaining]() -> bool { return values_remaining == 0; });
}

void SingleThreadedWorkQueue::DoWork(std::function<bool()> stop_predicate) {
  // Work items can add new items to the vector, and we generally want to run
  // things in order, so make sure we explicitly pop the item off before a new
  // one is added.
  std::vector<TaskFunction> local_work_items;
  int next_work_item_index = 0;

  // Run until the values get resolved.
  while (!stop_predicate()) {
    // If we've processed everything in local_work_items, then grab another
    // batch of stuff to do.  We're not done, so there must be more to do.
    if (next_work_item_index == local_work_items.size()) {
      local_work_items.clear();
      std::swap(local_work_items, work_items_);
      next_work_item_index = 0;
      assert(!local_work_items.empty() &&
             "ran out of work to do, but unresolved values remain!");
    }

    // Run the next work item.
    local_work_items[next_work_item_index]();

    // Move on to the next item.
    ++next_work_item_index;
  }

  // When we resolve all the async values, we're done.  However, we could have
  // one or more elements left in the local_work_items list.  If so, reconstruct
  // the worklist in the right order (inserting local work items at the head
  // of the remaining work to do).
  if (next_work_item_index != local_work_items.size()) {
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
