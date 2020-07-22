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

//===- io.cc ----------------------------------------------------*- C++ -*-===//
//
// This file implements PrefetchingIterator, a helper class for building IO
// iterators.
//
//===----------------------------------------------------------------------===//

#include "io.h"

namespace tfrt {
namespace data {
namespace io {

IterationResult PrefetchingIterator::GetNext(const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  llvm::SmallVector<RCReference<AsyncValue>, 1> result_values;
  result_values.resize(1);
  // The IndirectAsyncValue might be filled later by the background blocking
  // thread.
  result_values[0] = host->MakeIndirectAsyncValue();
  auto result_eof = host->MakeUnconstructedAsyncValueRef<bool>();
  auto result =
      IterationResult::Pending(std::move(result_values), std::move(result_eof));
  {
    mutex_lock lock(mu_);
    output_buffer_.push(result.CopyRef());

    // Return since another thread is already actively reading data and updating
    // the output_buffer_.
    if (token_owned_) {
      return result;
    }

    // The caller is a non-blocking thread, there is no token owner and there
    // are more values to prefetch. Schedule a blocking task to fetch values.
    if (!reached_eof_ &&
        prefetch_buffer_.size() < prefetch_threshold_ + output_buffer_.size()) {
      auto task = [iterator = FormRef(this), exec_ctx]() {
        iterator->ReadIOSource(exec_ctx);
      };
      if (exec_ctx.host()->EnqueueBlockingWork(std::move(task))) {
        // The task that is successfully scheduled in the blocking threadpool
        // owns the token.
        token_owned_ = true;
        return result;
      }
      // Caller thread has to read data from the underlying IO source because it
      // fails to schedule a blocking task to read the data.
      assert(output_buffer_.size() == 1);
      auto input = GetNextElement(exec_ctx);
      prefetch_buffer_.push(std::move(input));
    }
  }

  MaterializeOutputs(exec_ctx);
  return result;
}

void PrefetchingIterator::ReadIOSource(const ExecutionContext& exec_ctx) {
  {
    mutex_lock lock(mu_);
    assert(token_owned_);
    // The caller is the token owner, there are enough prefetched values and
    // there is no output value to update. Release the token and return.
    if (output_buffer_.empty() &&
        (prefetch_buffer_.size() >= prefetch_threshold_ || reached_eof_)) {
      token_owned_ = false;
      return;
    }
  }
  MaterializeOutputs(exec_ctx);

  // There is no other thread that owns the token. So we are free to access the
  // prefetch_buffer_ and the underlying IO source without having a lock.
  auto prefetch_num =
      max_prefetch_num_ + OutputBufferSize() - prefetch_buffer_.size();
  for (int32_t i = 0; i < prefetch_num; ++i) {
    if (exec_ctx.IsCancelled()) return;
    auto input = GetNextElement(exec_ctx);
    if (input.eof.IsConcrete() && input.eof.get()) {
      reached_eof_ = true;
      break;
    }
    prefetch_buffer_.push(std::move(input));
    auto output = DequeueOutputBuffer();
    if (output) {
      auto earliest_input = std::move(prefetch_buffer_.front());
      prefetch_buffer_.pop();
      // It is guaranteed that no other thread will attempt to dequeue value
      // from the output buffer.
      ForwardInputToOutput(std::move(earliest_input),
                           std::move(output.getValue()), exec_ctx);
    }
  }
  ReadIOSource(exec_ctx);
}

void PrefetchingIterator::ForwardInputToOutput(
    IterationResult input, IterationResult output,
    const ExecutionContext& exec_ctx) {
  assert(input.values.size() == output.values.size());
  for (int i = 0, e = output.values.size(); i < e; ++i) {
    auto* output_value = cast<IndirectAsyncValue>(output.values[i].get());
    output_value->ForwardTo(std::move(input.values[i]));
  }
  auto* input_eof = input.eof.GetAsyncValue();
  input_eof->AndThen([input_eof = std::move(input.eof),
                      output_eof = std::move(output.eof), exec_ctx] {
    if (input_eof.IsError()) {
      // Set error location and emit error before forwarding the error to
      // the output_eof.
      // TODO(donglin): If the error happens after the BEF executor has finished
      // its tasks, the LocationHandler might have been de-allocated at this
      // moment and this line can throw SIGSEGV. Fix it.
      input_eof.GetAsyncValue()->SetErrorLocationIfUnset(
          exec_ctx.location().Decode());
      exec_ctx.host()->EmitError(input_eof.GetError());
      output_eof.SetError(input_eof.GetError());
    } else {
      output_eof.emplace(input_eof.get());
    }
  });
}

void PrefetchingIterator::MaterializeOutputs(const ExecutionContext& exec_ctx) {
  // It is guaranteed that no other thread will attempt to dequeue value from
  // the output buffer concurrently.
  while (!prefetch_buffer_.empty()) {
    auto output = DequeueOutputBuffer();
    if (!output) break;
    auto input = std::move(prefetch_buffer_.front());
    prefetch_buffer_.pop();
    ForwardInputToOutput(std::move(input), std::move(output.getValue()),
                         exec_ctx);
  }
  if (reached_eof_) {
    IterationResult eof_result = IterationResult::Eof(exec_ctx.host(), 1);
    while (auto output = DequeueOutputBuffer()) {
      ForwardInputToOutput(eof_result.CopyRef(), std::move(output.getValue()),
                           exec_ctx);
    }
  }
}

}  // namespace io
}  // namespace data
}  // namespace tfrt
