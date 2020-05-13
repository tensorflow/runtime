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
    : public Dataset<OutputTypes...> {
 public:
  explicit MapDataset(RCReference<Dataset<InputTypes...>> input_dataset,
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

  RCReference<Iterator<OutputTypes...>> MakeIterator() override;

 private:
  // Allow iterator to rely on private data members of this dataset.
  friend class MapDatasetIterator<std::tuple<InputTypes...>,
                                  std::tuple<OutputTypes...>>;

  void Destroy() override {
    internal::DestroyImpl<
        MapDataset<std::tuple<InputTypes...>, std::tuple<OutputTypes...>>>(
        this, allocator_);
  }

  RCReference<Dataset<InputTypes...>> input_dataset_;
  HostContext* host_;
  HostAllocator* allocator_;
  RCArray<AsyncValue> additional_fn_args_;
  RCReference<const Function> map_fn_;
};

// Computes and returns map_fn(additional_fn_args, args).
template <typename... OutputTypes>
static IterationResult<OutputTypes...> RunFunction(
    const Function* map_fn, RCArray<AsyncValue> additional_fn_args,
    IterationResultUntyped input, const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  if (internal::IsConcreteAndEmpty(input)) {
    return IterationResult<OutputTypes...>::Eof(host);
  }
  auto async_result = host->template MakeUnconstructedAsyncValueRef<
      std::tuple<OutputTypes...>>();

  host->RunWhenReady(
      input.values, [host, map_fn = std::move(map_fn),
                     additional_fn_args = std::move(additional_fn_args),
                     input = input.CopyRef(),
                     async_result = async_result.CopyRef()]() mutable {
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
        host->EnqueueWork([host, map_fn = std::move(map_fn),
                           additional_fn_args = std::move(additional_fn_args),
                           args = std::move(input.values),
                           async_result = std::move(async_result)]() mutable {
          // Construct arguments for function execution. The arguments consist
          // of the 'additional_fn_args' from the MapDataset constructor,
          // followed by the values from the underlying iterator.
          SmallVector<AsyncValue*, 4> arguments;
          for (auto* additional_arg : additional_fn_args.values()) {
            arguments.push_back(additional_arg);
          }
          for (const auto& arg : args) {
            arguments.push_back(arg.get());
          }
          SmallVector<RCReference<AsyncValue>, sizeof...(OutputTypes)> results;
          results.resize(map_fn->result_types().size());
          map_fn->Execute(arguments, results, host);
          internal::VectorToTuple(std::move(results), async_result.CopyRef(),
                                  host);
        });
      });

  return IterationResult<OutputTypes...>::Pending(std::move(async_result),
                                                  std::move(input.eof));
}

template <typename... InputTypes, typename... OutputTypes>
class MapDatasetIterator<std::tuple<InputTypes...>, std::tuple<OutputTypes...>>
    : public Iterator<OutputTypes...> {
 public:
  explicit MapDatasetIterator(
      RCReference<
          MapDataset<std::tuple<InputTypes...>, std::tuple<OutputTypes...>>>
          parent_dataset)
      : Iterator<OutputTypes...>(),
        parent_dataset_(std::move(parent_dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()) {}

  IterationResult<OutputTypes...> GetNext(
      const ExecutionContext& exec_ctx) override {
    auto input = input_iterator_->GetNextUntyped(exec_ctx);
    const Function* map_fn = parent_dataset_->map_fn_.get();
    // IDEA(donglin): consider extending RCArray to support CopyRef() without
    // doing shallow copy.
    auto additional_fn_args = parent_dataset_->additional_fn_args_.CopyRef();
    return RunFunction<OutputTypes...>(map_fn, std::move(additional_fn_args),
                                       std::move(input), exec_ctx);
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
  RCReference<Iterator<InputTypes...>> input_iterator_;
};

template <typename... InputTypes, typename... OutputTypes>
RCReference<Iterator<OutputTypes...>> MapDataset<
    std::tuple<InputTypes...>, std::tuple<OutputTypes...>>::MakeIterator() {
  return TakeRef(
      host_->Construct<MapDatasetIterator<std::tuple<InputTypes...>,
                                          std::tuple<OutputTypes...>>>(
          FormRef(this)));
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_MAP_DATASET_H_
