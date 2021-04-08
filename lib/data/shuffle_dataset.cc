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

// This file defines the ShuffleDataset class.

#include "shuffle_dataset.h"

#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/support/philox_random.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// ShuffleDataset methods
//===----------------------------------------------------------------------===//
RCReference<Iterator> ShuffleDataset::MakeIterator(
    const IteratorContext& context) {
  return TakeRef(
      host_->Construct<ShuffleDatasetIterator>(FormRef(this), context));
}

//===----------------------------------------------------------------------===//
// ShuffleDatasetIterator methods
//===----------------------------------------------------------------------===//
IterationResult ShuffleDatasetIterator::GetNext(
    const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  // Initialize arity_ using the first value from the input_iterator_.
  if (arity_ < 0) {
    mutex_lock lock(mu_);
    assert(!token_owned_);
    auto input = input_iterator_->GetNext(exec_ctx);
    arity_ = input.values.size();
    shuffle_buffer_.push_back(std::move(input));
    num_shuffled_values_++;
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

void ShuffleDatasetIterator::MaybeScheduleBackgroundTask(
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
    is_token_owner = true;
  }

  // Only the thread that owns the token can execute the code below. This
  // ensures in-order delivery since at most one thread can take value from the
  // input_iterator_ and update the output value in the output_buffer_.
  auto host = exec_ctx.host();
  auto callback = [exec_ctx, callback_count,
                   iterator = FormRef(this)]() mutable {
    if (callback_count >= MAX_RECURSIVE_CALLS) {
      EnqueueWork(exec_ctx, [exec_ctx, iterator = std::move(iterator)] {
        iterator->MaybeScheduleBackgroundTask(exec_ctx, true, 0);
      });
    } else {
      iterator->MaybeScheduleBackgroundTask(exec_ctx, true, callback_count + 1);
    }
  };

  auto max_buffer_size = parent_dataset_->buffer_size_;
  // This optimizes the perform by skipping filling the shuffle_buffer_ if the
  // input iterator has reached end.
  CheckEof();

  while (OutputBufferSize() > 0) {
    // Fills shuffle_buffer_ with up to buffer_size_ values. It can have less
    // than buffer_size_ values only if the input_iterator_ has reached end.
    while (num_shuffled_values_ < max_buffer_size && !reached_eof_) {
      auto input = input_iterator_->GetNext(exec_ctx);
      auto* eof_async = input.eof.GetAsyncValue();
      num_shuffled_values_++;
      auto index = (start_index_ + num_shuffled_values_ - 1) % max_buffer_size;
      if (index < shuffle_buffer_.size()) {
        shuffle_buffer_[index] = std::move(input);
      } else {
        shuffle_buffer_.push_back(std::move(input));
      }
      if (eof_async->IsUnavailable()) {
        eof_async->AndThen(std::move(callback));
        return;
      }
      CheckEof();
    }

    if (num_shuffled_values_ > 0) {
      auto random_index =
          (random_() % num_shuffled_values_ + start_index_) % max_buffer_size;
      assert(random_index < shuffle_buffer_.size());
      auto input = std::move(shuffle_buffer_[random_index]);
      shuffle_buffer_[random_index] = std::move(shuffle_buffer_[start_index_]);
      start_index_ = (start_index_ + 1) % max_buffer_size;
      num_shuffled_values_--;
      HandleEofAvailableInput(std::move(input), host);
    } else {
      auto input = IterationResult::Eof(host, arity_);
      HandleEofAvailableInput(std::move(input), host);
    }
  }
  MaybeScheduleBackgroundTask(exec_ctx, true, callback_count);
}

void ShuffleDatasetIterator::HandleEofAvailableInput(IterationResult input,
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
}

inline void ShuffleDatasetIterator::CheckEof() {
  auto max_buffer_size = parent_dataset_->buffer_size_;
  if (reached_eof_ || num_shuffled_values_ == 0) return;

  auto index = (start_index_ + num_shuffled_values_ - 1) % max_buffer_size;
  assert(index < shuffle_buffer_.size());
  auto& eof = shuffle_buffer_[index].eof;
  if (eof.IsConcrete() && eof.get()) {
    reached_eof_ = true;
    auto removed_value = std::move(shuffle_buffer_[index]);
    (void)removed_value;
    num_shuffled_values_--;
  }
}

}  // namespace data
}  // namespace tfrt
