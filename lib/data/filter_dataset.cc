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

//===- filter_dataset.cc ----------------------------------------*- C++ -*-===//
//
// This file implements FilterDataset class which wraps around another Dataset
// instance and outputs those elements from the underlying dataset which satisfy
// a user-defined filter function.
//
//===----------------------------------------------------------------------===//

#include "filter_dataset.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// FilterDataset methods
//===----------------------------------------------------------------------===//
RCReference<Iterator> FilterDataset::MakeIterator() {
  return TakeRef(host_->Construct<FilterDatasetIterator>(FormRef(this)));
}

//===----------------------------------------------------------------------===//
// FilterDatasetIterator methods
//===----------------------------------------------------------------------===//

IterationResult FilterDatasetIterator::GetNext(
    const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();

  llvm::SmallVector<RCReference<AsyncValue>, 4> result_values;
  result_values.resize(parent_dataset_->arity_);
  for (size_t i = 0; i < parent_dataset_->arity_; ++i) {
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

void FilterDatasetIterator::MaybeScheduleBackgroundTask(
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

  // Fetch enough number of values from the `input_iterator_` to 1) satisfy the
  // values newly added in the `output_buffer_` and 2) compensate for the values
  // in the `input_and_predicate_buffer_` whose predicate value evaluates to
  // false. And schedule tasks to run the filter_fn for newly fetched values in
  // parallel.
  auto host = exec_ctx.host();
  int input_fetch_num = OutputBufferSize() -
                        input_and_predicate_buffer_.size() +
                        std::max(num_false_predicate_.load(), 0);
  const Function* filter_fn = parent_dataset_->filter_fn_.get();
  auto additional_fn_args =
      RCArray<AsyncValue>(llvm::SmallVector<AsyncValue*, 1>());
  for (int i = 0; i < input_fetch_num; i++) {
    auto input = input_iterator_->GetNext(exec_ctx);
    auto predicate_values =
        EnqueueFunction(filter_fn, additional_fn_args.CopyRef(),
                        RCArray<AsyncValue>(input.values), exec_ctx);
    assert(predicate_values.size() == 1);

    predicate_values[0]->AndThen(
        [predicate_values = predicate_values[0].CopyRef(),
         iterator = FormRef(this)]() mutable {
          if (!predicate_values->IsError() && !predicate_values->get<bool>()) {
            iterator->num_false_predicate_.fetch_add(1);
          }
        });
    auto predicate = IterationResult::Pending(std::move(predicate_values),
                                              input.eof.CopyRef());
    input_and_predicate_buffer_.push(
        std::make_pair(std::move(input), std::move(predicate)));
  }
  // After the first value in the `input_and_predicate_buffer_` becomes
  // available, the token owner should update `output_buffer` as appropriate,
  // then call MaybeScheduleBackgroundTask() again to schedule more tasks if
  // there are still unfilled outputs.
  assert(!input_and_predicate_buffer_.empty() &&
         "input_and_predicate_buffer should not be empty");
  auto next = std::move(input_and_predicate_buffer_.front());
  input_and_predicate_buffer_.pop();
  auto input = std::move(next.first);
  auto predicate = std::move(next.second);

  SmallVector<AsyncValue*, 4> async_value_ptrs;
  for (auto& value : input.values) async_value_ptrs.push_back(value.get());
  async_value_ptrs.push_back(input.eof.GetAsyncValue());
  async_value_ptrs.push_back(predicate.values[0].get());
  async_value_ptrs.push_back(predicate.eof.GetAsyncValue());
  host->RunWhenReady(async_value_ptrs, [exec_ctx, host, callback_count,
                                        input = std::move(input),
                                        predicate = std::move(predicate),
                                        iterator = FormRef(this)]() mutable {
    auto predicate_value = std::move(predicate.values[0]);
    auto predicate_eof = std::move(predicate.eof);
    if (predicate_eof.IsError()) {
      auto output = iterator->DequeueOutputBuffer();
      for (auto& value : output.values) {
        value->SetError(predicate_eof.GetError());
      }
      output.eof.SetError(predicate_eof.GetError());
    } else if (predicate_eof.get()) {
      // The input_iterator_ has been exhausted. Note that predicate_eof and
      // input.eof should have the same value.
      auto error = MakeErrorAsyncValueRef(host, "iterator reached end");
      while (iterator->OutputBufferSize() > 0) {
        auto output = iterator->DequeueOutputBuffer();
        for (auto& value : output.values) {
          value->SetError(error->GetError());
        }
        output.eof.emplace(true);
      }
    } else if (predicate_value->IsError()) {
      auto output = iterator->DequeueOutputBuffer();
      for (auto& value : output.values) {
        value->SetError(predicate_value->GetError());
      }
      output.eof.SetError(predicate_value->GetError());
    } else if (predicate_value->get<bool>()) {
      // The input satisfies the predicate.
      auto output = iterator->DequeueOutputBuffer();
      for (int i = 0; i < output.values.size(); ++i) {
        auto* output_value = cast<IndirectAsyncValue>(output.values[i].get());
        output_value->ForwardTo(std::move(input.values[i]));
      }
      output.eof.emplace(false);
    } else {
      iterator->num_false_predicate_.fetch_sub(1);
    }
    if (callback_count >= MAX_RECURSIVE_CALLS) {
      host->EnqueueWork([exec_ctx, iterator = std::move(iterator)] {
        iterator->MaybeScheduleBackgroundTask(exec_ctx, true, 0);
      });
    } else {
      iterator->MaybeScheduleBackgroundTask(exec_ctx, true, callback_count + 1);
    }
  });
}

}  // namespace data
}  // namespace tfrt
