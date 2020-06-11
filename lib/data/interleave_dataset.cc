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

//===- interleave_dataset.cc ------------------------------------*- C++ -*-===//
//
// This file implements InterleaveDataset class, which  applies a function to
// its input to create a dataset per input element, and interleaves the results
// of these datasets.
//
//===----------------------------------------------------------------------===//

#include "interleave_dataset.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// InterleaveDataset methods
//===----------------------------------------------------------------------===//
RCReference<Iterator> InterleaveDataset::MakeIterator() {
  return TakeRef(host_->Construct<InterleaveDatasetIterator>(FormRef(this)));
}

//===----------------------------------------------------------------------===//
// InterleaveDatasetIterator methods
//===----------------------------------------------------------------------===//
// TODO(b/155918211): Handle asynchrous EOF from the input_iterator_
IterationResult InterleaveDatasetIterator::GetNext(
    const ExecutionContext& exec_ctx) {
  while (!end_of_input_ || num_open_) {  // Not at end of input

    // Case 1: cycle_index_ has an open iterator. Get the next element from
    // that iterator and advance to the next block index.
    if (cycle_iterators_[cycle_index_]) {
      // Get the next element from the iterator opened at cycle_index_.
      auto result = cycle_iterators_[cycle_index_]->GetNext(exec_ctx);

      // If we're at the end of this current iterator, advance to the next
      // iterator in the cycle.
      if (internal::IsConcreteAndEmpty(result)) {
        cycle_iterators_[cycle_index_].reset();
        --num_open_;
        AdvanceCycleIndex();
        continue;
      }
      AdvanceBlockIndex();
      return result;
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
    if (internal::IsConcreteAndEmpty(input_element)) {
      end_of_input_ = true;
      continue;
    }

    auto values = std::move(input_element.values);
    for (const auto& value : values) {
      if (!value->IsAvailable()) {
        // TODO(rachelim): Currently, we don't have a good way to support
        // asynchronous transformations upstream of interleave, since
        // synchronous decisions such as whether to open a new iterator depend
        // on what iterators are already open. We need to support this use case,
        // e.g. if a user has MapDataset or asynchronous I/O upstream of an
        // interleave transformation.
        auto error = EmitErrorAsync(
            exec_ctx,
            "interleave expects its inputs to be available synchronously");
        return IterationResult::Error(std::move(error),
                                      parent_dataset_->arity_);
      }
      if (value->IsError()) {
        return IterationResult::Error(value.CopyRef(), parent_dataset_->arity_);
      }
    }

    cycle_iterators_[cycle_index_] =
        MakeIteratorFromInputElement(std::move(values), exec_ctx);
    ++num_open_;
  }

  // End of iteration.
  return IterationResult::Eof(exec_ctx.host(), parent_dataset_->arity_);
}

RCReference<Iterator> InterleaveDatasetIterator::MakeIteratorFromInputElement(
    llvm::SmallVector<RCReference<AsyncValue>, 4> input_element,
    const ExecutionContext& exec_ctx) {
  SmallVector<AsyncValue*, 4> fn_args;
  for (const auto& value : input_element) {
    fn_args.push_back(value.get());
  }
  SmallVector<RCReference<AsyncValue>, 1> fn_results;
  fn_results.resize(1);
  parent_dataset_->map_fn_->Execute(exec_ctx, fn_args, fn_results);

  // NOTE: If the inputs to this function are async, or the function is
  // executed asynchronously, this will fail.
  // TODO(rachelim): Try to support asynchronously created iterators.
  assert(fn_results[0]->IsAvailable());

  const auto& dataset = fn_results[0]->template get<RCReference<Dataset>>();
  return dataset->MakeIterator();
}

}  // namespace data
}  // namespace tfrt
