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

//===- map_dataset.h --------------------------------------------*- C++ -*-===//
//
// This file declares MapDataset class which wraps around another Dataset
// instance and transforms the element before returning it to the caller.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LIB_DATA_MAP_DATASET_H_
#define TFRT_LIB_DATA_MAP_DATASET_H_

#include "dataset.h"
#include "tfrt/host_context/function.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace data {

template <typename... T>
class MapDataset;

template <typename... T>
class MapDatasetIterator;

// MapDataset maps a user-defined function over the elements in its input
// dataset.
template <typename... InputTypes, typename... OutputTypes>
class MapDataset<std::tuple<InputTypes...>, std::tuple<OutputTypes...>>
    : public Dataset {
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

  RCReference<Iterator> MakeIterator() override;

 private:
  // Allow iterator to rely on private data members of this dataset.
  friend class MapDatasetIterator<std::tuple<InputTypes...>,
                                  std::tuple<OutputTypes...>>;

  void Destroy() override {
    internal::DestroyImpl<
        MapDataset<std::tuple<InputTypes...>, std::tuple<OutputTypes...>>>(
        this, allocator_);
  }

  RCReference<Dataset> input_dataset_;
  HostContext* host_;
  HostAllocator* allocator_;
  RCArray<AsyncValue> additional_fn_args_;
  RCReference<const Function> map_fn_;
};

// Enqueues map_fn(additional_fn_args, args) on the work queue and returns
// the result AsyncValues.
static llvm::SmallVector<RCReference<AsyncValue>, 4> EnqueueFunction(
    const Function* map_fn, RCArray<AsyncValue> additional_fn_args,
    RCArray<AsyncValue> args, const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  auto num_results = map_fn->result_types().size();

  // Placeholder for results.
  llvm::SmallVector<RCReference<IndirectAsyncValue>, 4> results;
  results.resize(num_results);
  for (size_t i = 0; i < num_results; ++i) {
    results[i] = host->MakeIndirectAsyncValue();
  }
  llvm::SmallVector<RCReference<AsyncValue>, 4> results_ref;
  for (const auto& result : results) {
    results_ref.push_back(result.CopyRef());
  }

  host->RunWhenReady(
      args.values(),
      [host, num_results, map_fn = std::move(map_fn),
       additional_fn_args = std::move(additional_fn_args),
       args = args.CopyRef(), results = std::move(results)]() mutable {
        // IDEA(donglin): We can optimize performance for small tasks by not
        // enqueuing small tasks to the threadpool. We need a way to identify
        // small tasks.
        //
        // Enqueue the map function to the threadpool to improve performance by
        // running the map function in parallel. An alternative approach to
        // increase parallelism is to compose map function with async kernels.
        // This alternative approach likely incurs higher thread context switch
        // overhead because different async kernels may be run by different
        // threads.
        //
        // NOTE: We enqueue work after the args are available. If we
        // enqueue work before the args are available, a thread from the
        // blocking threadpool might run the map function if the args is
        // computed by a thread in the blocking threadpool.
        host->EnqueueWork([host, num_results, map_fn = std::move(map_fn),
                           additional_fn_args = std::move(additional_fn_args),
                           args = std::move(args),
                           results = std::move(results)]() mutable {
          // Construct arguments for function execution. The arguments consist
          // of the 'additional_fn_args' from the MapDataset constructor,
          // followed by the values from the underlying iterator.
          SmallVector<AsyncValue*, 4> arguments;
          for (auto* additional_arg : additional_fn_args.values()) {
            arguments.push_back(additional_arg);
          }
          for (const auto& arg : args.values()) {
            arguments.push_back(arg);
          }
          SmallVector<RCReference<AsyncValue>, 4> fn_results;
          fn_results.resize(num_results);
          map_fn->Execute(arguments, fn_results, host);
          for (size_t i = 0; i < num_results; ++i) {
            results[i]->ForwardTo(std::move(fn_results[i]));
          }
        });
      });

  return results_ref;
}

template <typename... InputTypes, typename... OutputTypes>
class MapDatasetIterator<std::tuple<InputTypes...>, std::tuple<OutputTypes...>>
    : public Iterator {
 public:
  explicit MapDatasetIterator(
      RCReference<
          MapDataset<std::tuple<InputTypes...>, std::tuple<OutputTypes...>>>
          parent_dataset)
      : Iterator(),
        parent_dataset_(std::move(parent_dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()) {}

  IterationResult GetNext(const ExecutionContext& exec_ctx) override {
    auto input = input_iterator_->GetNext(exec_ctx);
    const Function* map_fn = parent_dataset_->map_fn_.get();

    auto values = std::move(input.values);
    auto eof = std::move(input.eof);

    // IDEA(donglin): consider extending RCArray to support CopyRef() without
    // doing shallow copy.
    auto additional_fn_args = parent_dataset_->additional_fn_args_.CopyRef();
    auto result =
        EnqueueFunction(map_fn, std::move(additional_fn_args),
                        RCArray<AsyncValue>(std::move(values)), exec_ctx);
    return IterationResult::Pending(std::move(result), std::move(eof));
  }

 private:
  // This class is not copyable or movable.
  MapDatasetIterator(const MapDatasetIterator&) = delete;
  MapDatasetIterator& operator=(const MapDatasetIterator&) = delete;

  void Destroy() override {
    internal::DestroyImpl<MapDatasetIterator>(this,
                                              parent_dataset_->allocator_);
  }

  RCReference<MapDataset<std::tuple<InputTypes...>, std::tuple<OutputTypes...>>>
      parent_dataset_;
  RCReference<Iterator> input_iterator_;
};

template <typename... InputTypes, typename... OutputTypes>
RCReference<Iterator> MapDataset<std::tuple<InputTypes...>,
                                 std::tuple<OutputTypes...>>::MakeIterator() {
  return TakeRef(
      host_->Construct<MapDatasetIterator<std::tuple<InputTypes...>,
                                          std::tuple<OutputTypes...>>>(
          FormRef(this)));
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_MAP_DATASET_H_
