/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file implements functions related to asynchronous work dispatching.

#include "tfrt/host_context/async_dispatch.h"

#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/task_function.h"

namespace tfrt {

void Await(const ExecutionContext& exec_ctx,
           ArrayRef<RCReference<AsyncValue>> values) {
  exec_ctx.host()->Await(values);
}

void EnqueueWork(const ExecutionContext& exec_ctx,
                 llvm::unique_function<void()> work) {
  auto& work_queue = exec_ctx.host()->work_queue();
  work_queue.AddTask(exec_ctx, TaskFunction(std::move(work)));
}

bool EnqueueBlockingWork(const ExecutionContext& exec_ctx,
                         llvm::unique_function<void()> work) {
  auto& work_queue = exec_ctx.host()->work_queue();
  Optional<TaskFunction> task = work_queue.AddBlockingTask(
      exec_ctx, TaskFunction(std::move(work)), /*allow_queuing=*/true);
  return !task.hasValue();
}

bool RunBlockingWork(const ExecutionContext& exec_ctx,
                     llvm::unique_function<void()> work) {
  auto& work_queue = exec_ctx.host()->work_queue();
  Optional<TaskFunction> task = work_queue.AddBlockingTask(
      exec_ctx, TaskFunction(std::move(work)), /*allow_queuing=*/false);
  return !task.hasValue();
}

void EnqueueWork(HostContext* host, llvm::unique_function<void()> work) {
  auto& work_queue = host->work_queue();
  work_queue.AddTask(TaskFunction(std::move(work)));
}

LLVM_NODISCARD bool EnqueueBlockingWork(HostContext* host,
                                        llvm::unique_function<void()> work) {
  auto& work_queue = host->work_queue();
  Optional<TaskFunction> task = work_queue.AddBlockingTask(
      TaskFunction(std::move(work)), /*allow_queuing=*/true);
  return !task.hasValue();
}

LLVM_NODISCARD bool RunBlockingWork(HostContext* host,
                                    llvm::unique_function<void()> work) {
  auto& work_queue = host->work_queue();
  Optional<TaskFunction> task = work_queue.AddBlockingTask(
      TaskFunction(std::move(work)), /*allow_queuing=*/false);
  return !task.hasValue();
}

void RunWhenReady(ArrayRef<AsyncValue*> values,
                  llvm::unique_function<void()> callee) {
  // Perform a quick scan of the arguments.  If they are all available, or if
  // any is already an error, then we can run the callee synchronously.
  SmallVector<AsyncValue*, 4> unavailable_values;
  for (auto i : values) {
    if (!i->IsAvailable()) unavailable_values.push_back(i);
  }

  // If we can synchronously call 'callee', then do it and we're done.
  if (unavailable_values.empty()) return callee();

  // If there is exactly one unavailable value, then we can just AndThen it.
  if (unavailable_values.size() == 1) {
    unavailable_values[0]->AndThen(
        [callee = std::move(callee)]() mutable { callee(); });
    return;
  }

  struct CounterAndCallee {
    std::atomic<size_t> counter;
    llvm::unique_function<void()> callee;
  };

  // Otherwise, we have multiple unavailable values.  Put a counter on the heap
  // and have each unavailable value decrement and test it.
  auto* data =
      new CounterAndCallee{{unavailable_values.size()}, std::move(callee)};

  for (auto* val : unavailable_values) {
    val->AndThen([data]() {
      // Decrement the counter unless we're the last to be here.
      if (data->counter.fetch_sub(1) != 1) return;

      // If we are the last one, then run the callee and free the data.
      data->callee();
      delete data;
    });
  }
}

void RunWhenReady(ArrayRef<RCReference<AsyncValue>> values,
                  llvm::unique_function<void()> callee) {
  auto mapped = llvm::map_range(
      values, [](const RCReference<AsyncValue>& ref) -> AsyncValue* {
        return ref.get();
      });
  SmallVector<AsyncValue*, 8> values_ptr(mapped.begin(), mapped.end());
  RunWhenReady(values_ptr, std::move(callee));
}

}  // namespace tfrt
