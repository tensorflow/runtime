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

//===- repeat_dataset.cc ----------------------------------------*- C++ -*-===//
//
// This file implements RepeatDataset class which wraps around another Dataset
// instance and repeats it a specified number of times.
//
//===----------------------------------------------------------------------===//

#include "repeat_dataset.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// RepeatDataset methods
//===----------------------------------------------------------------------===//
RCReference<Iterator> RepeatDataset::MakeIterator() {
  return TakeRef(host_->Construct<RepeatDatasetIterator>(FormRef(this)));
}

//===----------------------------------------------------------------------===//
// RepeatDatasetIterator methods
//===----------------------------------------------------------------------===//
IterationResult RepeatDatasetIterator::GetNext(
    const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  // Initialize value_count using the first value from the input_iterator_.
  if (arity_ < 0) {
    mutex_lock lock(mu_);
    assert(!token_owned_);
    auto input = input_iterator_->GetNext(exec_ctx);
    arity_ = input.values.size();
    input_buffer_.push(std::move(input));
  }
  if (parent_dataset_->count_ == 0) {
    return IterationResult::Eof(host, arity_);
  }

  llvm::SmallVector<RCReference<AsyncValue>, 4> result_values;
  result_values.resize(arity_);
  for (size_t i = 0; i < arity_; ++i) {
    result_values[i] = MakeIndirectAsyncValue(host);
  }
  auto result_eof = MakeUnconstructedAsyncValueRef<bool>(host);
  auto result =
      IterationResult::Pending(std::move(result_values), std::move(result_eof));
  {
    mutex_lock lock(mu_);
    output_buffer_.push(result.CopyRef());
  }

  MaybeScheduleBackgroundTask(exec_ctx, false, 0);
  return result;
}

void RepeatDatasetIterator::MaybeScheduleBackgroundTask(
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
  //
  // Fetch enough number of values from the `input_iterator_` to 1) satisfy the
  // values newly added in the `output_buffer_` and 2) compensate for the values
  // in the `input_and_predicate_buffer_` whose predicate value evaluates to
  // false. And schedule tasks to run the filter_fn for newly fetched values in
  // parallel.
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

  int input_fetch_num = OutputBufferSize() - input_buffer_.size();
  for (int i = 0; i < input_fetch_num; i++) {
    auto input = input_iterator_->GetNext(exec_ctx);
    assert(arity_ == input.values.size());
    input_buffer_.push(std::move(input));
  }
  // If there are multiple available values, handle them immediately to reduce
  // the number of recursive function calls and the mutex grab/release.
  while (!input_buffer_.empty() && input_buffer_.front().eof.IsAvailable()) {
    auto input = std::move(input_buffer_.front());
    input_buffer_.pop();
    HandleEofAvailableInput(std::move(input), host);
  }
  if (input_buffer_.empty()) {
    // Recursively call the function again because the output_buffer_ might have
    // more values. No state is kept in the stack due to tail recursion. Thus we
    // don't need to increment the callback_count.
    MaybeScheduleBackgroundTask(exec_ctx, true, callback_count);
    return;
  }
  // After the first value in the `input_buffer_` becomes available, the token
  // owner should update `output_buffer` as appropriate, then call
  // MaybeScheduleBackgroundTask() again to schedule more tasks if there are
  // still unfilled outputs.
  auto* input_eof_ptr = input_buffer_.front().eof.GetAsyncValue();
  input_eof_ptr->AndThen(std::move(callback));
}

void RepeatDatasetIterator::HandleEofAvailableInput(IterationResult input,
                                                    HostContext* host) {
  auto input_eof = std::move(input.eof);
  auto input_values = std::move(input.values);
  if (input_eof.IsError() || !input_eof.get()) {
    auto output = DequeueOutputBuffer();
    for (int i = 0; i < arity_; ++i) {
      auto* output_value = cast<IndirectAsyncValue>(output.values[i].get());
      output_value->ForwardTo(std::move(input_values[i]));
    }
    if (input_eof.IsError()) {
      output.eof.SetError(input_eof.GetError());
    } else {
      output.eof.emplace(false);
    }
    return;
  }
  bool can_repeat = (parent_dataset_->count_ > 0 &&
                     current_count_ + 1 < parent_dataset_->count_) ||
                    parent_dataset_->count_ < 0;
  if (can_repeat) {
    // The input_iterator_ has been exhausted and there is remaining count.
    current_count_++;
    input_iterator_ = parent_dataset_->input_dataset_->MakeIterator();
    // All the remaining elements in the buffer must be EOF because they come
    // from the exhausted iterator. Therefore we can clear the buffer.
    input_buffer_ = {};
    return;
  }
  // The input_iterator_ has been exhausted and there is no remaining count.
  auto error = MakeErrorAsyncValueRef(host, "iterator reached end");
  auto output_buffer_size = OutputBufferSize();
  for (; output_buffer_size > 0; --output_buffer_size) {
    auto output = DequeueOutputBuffer();
    for (auto& value : output.values) {
      value->SetError(error->GetError());
    }
    output.eof.emplace(true);
  }
  // All the remaining elements in the buffer must be EOF because they come
  // from the exhausted iterator. Therefore we can clear the buffer.
  input_buffer_ = {};
}

}  // namespace data
}  // namespace tfrt
