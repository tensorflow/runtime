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

// This file declares MapDataset class which wraps around another Dataset
// instance and transforms the element before returning it to the caller.

#ifndef TFRT_LIB_DATA_MAP_DATASET_H_
#define TFRT_LIB_DATA_MAP_DATASET_H_

#include "tfrt/data/dataset.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/function.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace data {

class MapDatasetIterator;

// MapDataset maps a user-defined function over the elements in its input
// dataset.
class MapDataset : public Dataset {
 public:
  explicit MapDataset(RCReference<Dataset> input_dataset,
                      RCArray<AsyncValue> additional_fn_args,
                      RCReference<const Function> map_fn, HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        host_(host),
        allocator_(host->allocator()),
        additional_fn_args_(std::move(additional_fn_args)),
        map_fn_(std::move(map_fn)) {}

  // This class is not copyable or movable.
  MapDataset(const MapDataset&) = delete;
  MapDataset& operator=(const MapDataset&) = delete;

  RCReference<Iterator> MakeIterator(const IteratorContext& context) override;

 private:
  // Allow iterator to rely on private data members of this dataset.
  friend class MapDatasetIterator;

  void Destroy() override {
    internal::DestroyImpl<MapDataset>(this, allocator_);
  }

  RCReference<Dataset> input_dataset_;
  HostContext* host_;
  HostAllocator* allocator_;
  RCArray<AsyncValue> additional_fn_args_;
  RCReference<const Function> map_fn_;
};

// If all AsyncValue's in `arguments` are available at the time this method is
// called, run the function using `arguments` as input and return the
// function's execution result. Otherwise, run the function when AsyncValue's
// in `arguments` are all available and return a set of IndirectAsyncValue's.
// Those IndirectAsyncValue's will be resolved later using function's execution
// results.
//
// This method is useful when the function might enqueue work into a
// threadpool and we want to make sure the work will indeed be executed in the
// specified threadpool. If we call Function::Execute() directly, the work
// might instead be executed by the thread that resolves the last un-resolved
// argument according to the 'AndThen' semantics.
//
// The function is executed inline without being explicitly enqueued to the
// threadpool. We expect the function itself to offload expensive operations to
// the threadpool.
inline llvm::SmallVector<RCReference<AsyncValue>, 4> RunFunctionWhenReady(
    const Function* function,
    llvm::SmallVector<RCReference<AsyncValue>, 4> arguments,
    const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  auto num_results = function->result_types().size();
  bool is_ready = true;
  llvm::SmallVector<AsyncValue*, 4> argument_ptrs;
  for (const auto& argument : arguments) {
    if (!argument->IsAvailable()) is_ready = false;
    argument_ptrs.push_back(argument.get());
  }
  // This is the fast path when all arguments are already available. It avoids
  // the overhead of creating IndirectAsyncValue in the slow path.
  if (is_ready) {
    SmallVector<RCReference<AsyncValue>, 4> fn_results;
    fn_results.resize(num_results);
    function->Execute(exec_ctx, argument_ptrs, fn_results);
    return fn_results;
  }

  llvm::SmallVector<RCReference<IndirectAsyncValue>, 4> results;
  llvm::SmallVector<RCReference<AsyncValue>, 4> results_copy;
  results.resize(num_results);
  results_copy.resize(num_results);
  for (size_t i = 0; i < num_results; ++i) {
    results[i] = MakeIndirectAsyncValue(host);
    results_copy[i] = results[i];
  }

  RunWhenReady(argument_ptrs, [function, arguments = std::move(arguments),
                               results = std::move(results), argument_ptrs,
                               exec_ctx]() mutable {
    auto num_results = function->result_types().size();
    SmallVector<RCReference<AsyncValue>, 4> fn_results;
    fn_results.resize(num_results);
    function->Execute(exec_ctx, argument_ptrs, fn_results);
    for (size_t i = 0; i < num_results; ++i) {
      results[i]->ForwardTo(std::move(fn_results[i]));
    }
  });

  return results_copy;
}

class MapDatasetIterator : public Iterator {
 public:
  explicit MapDatasetIterator(RCReference<MapDataset> parent_dataset,
                              const IteratorContext& context)
      : Iterator(),
        parent_dataset_(std::move(parent_dataset)),
        input_iterator_(
            parent_dataset_->input_dataset_->MakeIterator(context)) {}

  IterationResult GetNext(const ExecutionContext& exec_ctx) override;

 private:
  // This class is not copyable or movable.
  MapDatasetIterator(const MapDatasetIterator&) = delete;
  MapDatasetIterator& operator=(const MapDatasetIterator&) = delete;

  void Destroy() override {
    internal::DestroyImpl<MapDatasetIterator>(this,
                                              parent_dataset_->allocator_);
  }

  RCReference<MapDataset> parent_dataset_;
  RCReference<Iterator> input_iterator_;
};

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_MAP_DATASET_H_
