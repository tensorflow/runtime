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

IterationResult InterleaveDatasetIterator::GetNext(
    const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();

  llvm::SmallVector<RCReference<AsyncValue>, 4> result_values;
  result_values.resize(parent_dataset_->arity_);
  for (size_t i = 0; i < parent_dataset_->arity_; ++i) {
    result_values[i] = host->MakeIndirectAsyncValue();
  }
  auto result_eof = host->MakeUnconstructedAsyncValueRef<bool>();
  auto result =
      IterationResult::Pending(std::move(result_values), std::move(result_eof));
  {
    mutex_lock lock(mu_);
    output_buffer_.push(result.CopyRef());
  }

  MaybeScheduleBackgroundTask(exec_ctx, false, 0);
  return result;
}

AsyncValue* InterleaveDatasetIterator::FetchInputValues(
    const ExecutionContext& exec_ctx) {
  auto output_buffer_size = OutputBufferSize();

  // The loop attempts to fetch enough values from the intermediate iterators to
  // fill every value in the output_buffer_. It may stop earlier if e.g. an
  // intermediate iterator is not available.
  while (total_queues_size_ < output_buffer_size) {
    auto& iterator_and_queue = iterator_and_queues_[iterator_index_for_fetch_];

    if (!iterator_and_queue.is_open) {
      // We can not create any new iterator and there is no more open iterator
      // to fetch from.
      if (is_input_iterator_eof_ && num_open_iterators_ == 0) break;
      // We can not create any new iterator and there is open iterator to fetch
      // from. Move to the next iterator in the circular array.
      if (is_input_iterator_eof_) {
        iterator_index_for_fetch_ =
            (iterator_index_for_fetch_ + 1) % parent_dataset_->cycle_length_;
        continue;
      }
      // Create a new iterator at the current position in the array.
      auto input_value = input_iterator_->GetNext(exec_ctx);
      // Construct dataset = func_(input_value).
      SmallVector<AsyncValue*, 4> fn_args;
      for (const auto& value : input_value.values) {
        fn_args.push_back(value.get());
      }
      SmallVector<RCReference<AsyncValue>, 1> fn_results;
      fn_results.resize(1);
      parent_dataset_->func_->Execute(exec_ctx, fn_args, fn_results);

      iterator_and_queues_[iterator_index_for_fetch_] = IteratorAndQueue(
          std::move(input_value), std::move(fn_results[0]), true);
      num_open_iterators_++;
      continue;
    }

    auto& input_value_eof = iterator_and_queue.input_value.eof;
    // Wait for the eof of the input_value, which is used to create the
    // iterator at the current position, to be available.
    if (!input_value_eof.IsAvailable()) {
      return input_value_eof.GetAsyncValue();
    }
    // The eof of input_value has error. Propagate the error to the next value
    // in the output_buffer_ and update iterator's state.
    if (input_value_eof.IsError()) {
      auto output = DequeueOutputBuffer();
      output_buffer_size--;

      output.eof.SetError(input_value_eof.GetError());
      for (int i = 0; i < parent_dataset_->arity_; ++i) {
        output.values[i]->SetError(input_value_eof.GetError());
      }
      iterator_and_queue.is_open = false;
      num_open_iterators_--;
      continue;
    }
    // The eof of input_value is true.
    if (input_value_eof.get()) {
      is_input_iterator_eof_ = true;
      iterator_and_queue.is_open = false;
      num_open_iterators_--;
      continue;
    }

    auto& dataset = iterator_and_queue.dataset;
    // Wait for the dataset to be available.
    if (!dataset->IsAvailable()) {
      return dataset.get();
    }
    // The dataset has error. Propagate the error to the next value in
    // the output_buffer_ and update iterator's state.
    if (dataset->IsError()) {
      auto output = DequeueOutputBuffer();
      output_buffer_size--;

      output.eof.SetError(dataset->GetError());
      for (int i = 0; i < parent_dataset_->arity_; ++i) {
        output.values[i]->SetError(dataset->GetError());
      }
      iterator_and_queue.is_open = false;
      num_open_iterators_--;
      continue;
    }
    // Create the iterator from the dataset if it has not already been created.
    if (!iterator_and_queue.iterator) {
      iterator_and_queue.iterator =
          dataset->template get<RCReference<Dataset>>()->MakeIterator();
    }

    auto& iterator = iterator_and_queue.iterator;
    auto& queue = iterator_and_queue.queue;

    if (iterator_and_queue.fetched_num_in_block == 0 && !queue.empty()) {
      // iterator_index_for_fetch_ is ahead of iterator_index_for_output_ by
      // one cycle of iterators. Exit the loop so that we can forward the
      // fetched values to the outputs before fetching more values. Otherwise a
      // newly created iterator's queue can be several blocks of values behind
      // other iterators' queue which complicates FillOutputValues().
      break;
    }

    // Decide the number of values to fetch from the current iterator. It
    // should not exceed 1) the pending values in the the output_buffer_
    // and 2) the remaining values to fetch in the current block.
    auto fetched_num_in_block = iterator_and_queue.fetched_num_in_block;
    auto fetch_num =
        std::min(output_buffer_size - total_queues_size_,
                 parent_dataset_->block_length_ - fetched_num_in_block);
    // Pretch values from the current iterator into its queue and update
    // iterator's state.
    for (int i = 0; i < fetch_num; ++i) {
      queue.push(iterator->GetNext(exec_ctx));
    }
    iterator_and_queue.fetched_num_in_block += fetch_num;
    total_queues_size_ += fetch_num;
    // Move to the next iterator in the circular array if we have fetched
    // block_length_ number of values from the current iterator.
    if (iterator_and_queue.fetched_num_in_block ==
        parent_dataset_->block_length_) {
      iterator_and_queue.fetched_num_in_block = 0;
      iterator_index_for_fetch_ =
          (iterator_index_for_fetch_ + 1) % parent_dataset_->cycle_length_;
    }
  }
  return nullptr;
}

AsyncValue* InterleaveDatasetIterator::FillOutputValues(
    const ExecutionContext& exec_ctx) {
  while (total_queues_size_ > 0) {
    assert(num_open_iterators_ > 0);
    auto& iterator_and_queue = iterator_and_queues_[iterator_index_for_output_];
    // Move to the next iterator to look for the open iterator.
    if (!iterator_and_queue.is_open) {
      iterator_index_for_output_ =
          (iterator_index_for_output_ + 1) % parent_dataset_->cycle_length_;
      continue;
    }
    auto& queue = iterator_and_queue.queue;
    assert(!queue.empty());

    auto& next_result = queue.front();
    // Wait for the eof of the value at the front of the queue to be available.
    // This is needed to ensure the correct delivery order.
    if (!next_result.eof.IsAvailable()) {
      return next_result.eof.GetAsyncValue();
    }
    // The eof of the value at the front of the queue has error. Propagate the
    // error to the next value in the output_buffer_.
    if (next_result.eof.IsError()) {
      auto output = DequeueOutputBuffer();
      output.eof.SetError(next_result.eof.GetError());
      for (int i = 0; i < parent_dataset_->arity_; ++i) {
        output.values[i]->SetError(next_result.eof.GetError());
      }

      queue.pop();
      total_queues_size_--;
      iterator_and_queue.output_num_in_block++;
      if (iterator_and_queue.output_num_in_block ==
          parent_dataset_->block_length_) {
        iterator_and_queue.output_num_in_block = 0;
        iterator_index_for_output_ =
            (iterator_index_for_output_ + 1) % parent_dataset_->cycle_length_;
      }
      continue;
    }
    // The current iterator has reached end. Update iterator's state.
    if (next_result.eof.get()) {
      total_queues_size_ -= queue.size();
      iterator_and_queue.is_open = false;
      num_open_iterators_--;

      iterator_index_for_output_ =
          (iterator_index_for_output_ + 1) % parent_dataset_->cycle_length_;
      continue;
    }
    // The current iterator has not reached end. Forward the value at the front
    // of its queue to the next value in the output_buffer_.
    auto output = DequeueOutputBuffer();
    output.eof.emplace(false);
    for (int i = 0; i < parent_dataset_->arity_; ++i) {
      auto* output_value = cast<IndirectAsyncValue>(output.values[i].get());
      output_value->ForwardTo(std::move(next_result.values[i]));
    }
    queue.pop();
    total_queues_size_--;
    // Move to the next iterator in the circular array if we have fetched
    // block_length_ number of values from the queue of the current iterator.
    iterator_and_queue.output_num_in_block++;
    if (iterator_and_queue.output_num_in_block ==
        parent_dataset_->block_length_) {
      iterator_and_queue.output_num_in_block = 0;
      iterator_index_for_output_ =
          (iterator_index_for_output_ + 1) % parent_dataset_->cycle_length_;
    }
  }
  return nullptr;
}

void InterleaveDatasetIterator::MaybeScheduleBackgroundTask(
    const ExecutionContext& exec_ctx, bool is_token_owner, int callback_count) {
  {
    mutex_lock lock(mu_);
    // There is no more output value to update. Release the token if the caller
    // owns the token and then return.
    if (output_buffer_.empty()) {
      if (is_token_owner) {
        token_owned_ = false;
      }
      return;
    }
    // Return since the token is already owned by another thread.
    if (!is_token_owner && token_owned_) return;
    // Take the token if the thread does not already own the token.
    token_owned_ = true;
  }
  // Only the thread that owns the token can execute the code below. This
  // ensures in-order delivery since at most one thread can take value from the
  // input_iterator_ and update the output value in the output_buffer_.

  auto host = exec_ctx.host();
  auto callback = [exec_ctx, host, callback_count,
                   iterator = FormRef(this)]() mutable {
    if (callback_count >= MAX_RECURSIVE_CALLS) {
      host->EnqueueWork([exec_ctx, iterator = std::move(iterator)] {
        iterator->MaybeScheduleBackgroundTask(exec_ctx, true, 0);
      });
    } else {
      iterator->MaybeScheduleBackgroundTask(exec_ctx, true, callback_count + 1);
    }
  };

  auto* unavailable_async_value_ptr = FetchInputValues(exec_ctx);
  // Call MaybeScheduleBackgroundTask() again when this value is available.
  if (unavailable_async_value_ptr != nullptr) {
    unavailable_async_value_ptr->AndThen(std::move(callback));
    return;
  }

  // input_iterator_ has reached end and there is no open iterator to fetch
  // from. Mark all values in the output_buffer_ to be eof=true.
  if (total_queues_size_ == 0) {
    assert(is_input_iterator_eof_ && num_open_iterators_ == 0);
    auto error = host->MakeErrorAsyncValueRef("iterator reached end");
    auto output_buffer_size = OutputBufferSize();
    for (; output_buffer_size > 0; --output_buffer_size) {
      auto output = DequeueOutputBuffer();
      for (auto& value : output.values) {
        value->SetError(error->GetError());
      }
      output.eof.emplace(true);
    }
    // No state is kept in the stack due to tail recursion. Thus we don't need
    // to increment the callback_count.
    MaybeScheduleBackgroundTask(exec_ctx, true, callback_count);
    return;
  }

  assert(num_open_iterators_ > 0);
  unavailable_async_value_ptr = FillOutputValues(exec_ctx);
  if (unavailable_async_value_ptr != nullptr) {
    // Call MaybeScheduleBackgroundTask() again when this value is available.
    unavailable_async_value_ptr->AndThen(std::move(callback));
    return;
  }
  // No state is kept in the stack due to tail recursion. Thus we don't need to
  // increment the callback_count.
  MaybeScheduleBackgroundTask(exec_ctx, true, callback_count);
}

}  // namespace data
}  // namespace tfrt
