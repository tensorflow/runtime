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

//===- interleave_dataset.h -------------------------------------*- C++ -*-===//
//
// This file declares InterleaveDataset class applies a function to its input
// to create a dataset per input element, and interleaves the results of these
// datasets.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LIB_DATA_INTERLEAVE_DATASET_H_
#define TFRT_LIB_DATA_INTERLEAVE_DATASET_H_

#include "dataset.h"
#include "tfrt/host_context/function.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace data {

template <typename... T>
class InterleaveDataset;

template <typename... T>
class InterleaveDatasetIterator;

// InterleaveDataset maps a user-defined function over the elements in its input
// dataset and interleaves the results. The user-defined function is expected to
// have a single return value of type Dataset<OutputTypes...>.
//
// The `cycle_length` and `block_length` arguments control the order in which
// elements are produced. `cycle_length` controls the number of input elements
// that are processed concurrently. If `cycle_length` is 1, this
// transformation will handle one input element at a time, and is equivalent
// to performing a flat map. In general, this transformation will apply the
// map function to `cycle_length` input elements, open iterators on the
// returned Dataset objects, and cycle through them, producing `block_length`
// consecutive elements from each iterator, and consuming the next input
// element each time it reaches the end of an iterator.
template <typename... InputTypes, typename... OutputTypes>
class InterleaveDataset<std::tuple<InputTypes...>, std::tuple<OutputTypes...>>
    : public Dataset<OutputTypes...> {
 public:
  explicit InterleaveDataset(RCReference<Dataset<InputTypes...>> input_dataset,
                             int64_t cycle_length, int64_t block_length,
                             RCReference<const Function> map_fn,
                             HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        cycle_length_(cycle_length),
        block_length_(block_length),
        host_(host),
        allocator_(host->allocator()),
        map_fn_(std::move(map_fn)) {}

  // This class is not copyable or movable.
  InterleaveDataset(const InterleaveDataset&) = delete;
  InterleaveDataset& operator=(const InterleaveDataset&) = delete;

  RCReference<Iterator<OutputTypes...>> MakeIterator() override;

 private:
  // Allow iterator to rely on private data members of this dataset.
  friend class InterleaveDatasetIterator<std::tuple<InputTypes...>,
                                         std::tuple<OutputTypes...>>;

  void Destroy() override {
    internal::DestroyImpl<InterleaveDataset<std::tuple<InputTypes...>,
                                            std::tuple<OutputTypes...>>>(
        this, allocator_);
  }

  RCReference<Dataset<InputTypes...>> input_dataset_;
  int64_t cycle_length_;
  int64_t block_length_;
  HostContext* host_;
  HostAllocator* allocator_;
  RCReference<const Function> map_fn_;
};

template <typename... InputTypes, typename... OutputTypes>
class InterleaveDatasetIterator<std::tuple<InputTypes...>,
                                std::tuple<OutputTypes...>>
    : public Iterator<OutputTypes...> {
 public:
  explicit InterleaveDatasetIterator(
      RCReference<InterleaveDataset<std::tuple<InputTypes...>,
                                    std::tuple<OutputTypes...>>>
          parent_dataset)
      : Iterator<OutputTypes...>(),
        parent_dataset_(std::move(parent_dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()),
        cycle_iterators_(parent_dataset_->cycle_length_) {}

  // This class is not copyable or movable.
  InterleaveDatasetIterator(const InterleaveDatasetIterator&) = delete;
  InterleaveDatasetIterator& operator=(const InterleaveDatasetIterator&) =
      delete;

  // Interleaves keeps cycle_length iterators open at once.
  AsyncValueRef<std::tuple<OutputTypes...>> GetNext(
      const ExecutionContext& exec_ctx) override;

 private:
  void Destroy() override {
    internal::DestroyImpl<InterleaveDatasetIterator>(
        this, parent_dataset_->allocator_);
  }

  RCReference<
      InterleaveDataset<std::tuple<InputTypes...>, std::tuple<OutputTypes...>>>
      parent_dataset_;
  RCReference<Iterator<InputTypes...>> input_iterator_;

  std::vector<RCReference<Iterator<OutputTypes...>>> cycle_iterators_;
  size_t cycle_index_ = 0;
  size_t block_index_ = 0;
  bool end_of_input_ = false;
  size_t num_open_ = 0;  // Number of open iterators

  // Advance the next block index. If the next block index exceeds the block
  // length, advance to the next iterator in the cycle.
  void AdvanceBlockIndex() {
    ++block_index_;
    if (block_index_ == parent_dataset_->block_length_) {
      AdvanceCycleIndex();
    }
  }

  // Advance to the next iterator in the cycle and reset block_index_ to 0.
  void AdvanceCycleIndex() {
    block_index_ = 0;
    cycle_index_ = (cycle_index_ + 1) % parent_dataset_->cycle_length_;
  }

  RCReference<Iterator<OutputTypes...>> MakeIteratorFromInputElement(
      AsyncValueRef<std::tuple<InputTypes...>> input_element,
      const ExecutionContext& exec_ctx);
};

template <typename... InputTypes, typename... OutputTypes>
RCReference<Iterator<OutputTypes...>> InterleaveDataset<
    std::tuple<InputTypes...>, std::tuple<OutputTypes...>>::MakeIterator() {
  return TakeRef(
      host_->Construct<InterleaveDatasetIterator<std::tuple<InputTypes...>,
                                                 std::tuple<OutputTypes...>>>(
          FormRef(this)));
}

template <typename... InputTypes, typename... OutputTypes>
AsyncValueRef<std::tuple<OutputTypes...>> InterleaveDatasetIterator<
    std::tuple<InputTypes...>,
    std::tuple<OutputTypes...>>::GetNext(const ExecutionContext& exec_ctx) {
  while (!end_of_input_ || num_open_) {  // Not at end of input

    // Case 1: cycle_index_ has an open iterator. Get the next element from
    // that iterator and advance to the next block index.
    if (cycle_iterators_[cycle_index_]) {
      // Get the next element from the iterator opened at cycle_index_.
      auto value = cycle_iterators_[cycle_index_]->GetNext(exec_ctx);

      // If we're at the end of this current iterator, advance to the next
      // iterator in the cycle.
      if (!value) {
        cycle_iterators_[cycle_index_].reset();
        --num_open_;
        AdvanceCycleIndex();
        continue;
      }
      AdvanceBlockIndex();
      return value;
    }

    // Case 2: cycle_index_ does not have an open iterator, and we've reached
    // the end of the input, therefore cannot open any more iterators. We have
    // to exhaust all the remaining open iterators.
    if (end_of_input_) {
      AdvanceCycleIndex();
      continue;
    }

    // Case 3: This iterator at the current cycle_index_ has not been created.
    // Get the next element from the input dataset and create an iterator
    // from it.
    auto input_element = input_iterator_->GetNext(exec_ctx);

    // The input iterator has been exhausted.
    if (!input_element) {
      end_of_input_ = true;
      continue;
    }

    if (!input_element.IsAvailable()) {
      // TODO(rachelim): Currently, we don't have a good way to support
      // asynchronous transformations upstream of interleave, since
      // synchronous decisions such as whether to open a new iterator depend
      // on what iterators are already open. We need to support this use case,
      // e.g. if a user has MapDataset or asynchronous I/O upstream of an
      // interleave transformation.
      return EmitErrorAsync(
          exec_ctx,
          "interleave expects its inputs to be available synchronously");
    }
    if (input_element.IsError()) {
      return AsyncValueRef<std::tuple<OutputTypes...>>(
          input_element.ReleaseRCRef());
    }

    cycle_iterators_[cycle_index_] =
        MakeIteratorFromInputElement(std::move(input_element), exec_ctx);
    ++num_open_;
  }

  // End of iteration.
  return AsyncValueRef<std::tuple<OutputTypes...>>();
}

template <typename... InputTypes, typename... OutputTypes>
RCReference<Iterator<OutputTypes...>> InterleaveDatasetIterator<
    std::tuple<InputTypes...>, std::tuple<OutputTypes...>>::
    MakeIteratorFromInputElement(
        AsyncValueRef<std::tuple<InputTypes...>> input_element,
        const ExecutionContext& exec_ctx) {
  // Translate from AsyncValue<std::tuple<T...>> to
  // SmallVector<AsyncValue<T...>*, 4>
  // TODO(rachelim): Support inputs of arbitrary arity.
  SmallVector<AsyncValue*, 4> fn_args;
  auto arg =
      exec_ctx.host()->template MakeAvailableAsyncValueRef<InputTypes...>(
          std::move(std::get<0>(input_element.get())));
  fn_args.push_back(arg.GetAsyncValue());
  SmallVector<RCReference<AsyncValue>, 1> fn_results;
  fn_results.resize(1);
  parent_dataset_->map_fn_->Execute(fn_args, fn_results, exec_ctx.host());

  // NOTE: If the inputs to this function are async, or the function is
  // executed asynchronously, this will fail.
  // TODO(rachelim): Try to support asynchronously created iterators.
  assert(fn_results[0]->IsAvailable());

  const auto& dataset =
      fn_results[0]->template get<RCReference<Dataset<OutputTypes...>>>();
  return dataset->MakeIterator();
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_INTERLEAVE_DATASET_H_
