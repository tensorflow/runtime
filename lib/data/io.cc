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

#include "tfrt/tracing/tracing.h"

namespace tfrt {
namespace data {
namespace io {

IterationResult PrefetchingIterator::GetNext(const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  {
    mutex_lock lock(mu_);
    // Schedule a blocking thread to fetch data if the number of prefetched
    // values has dropped below the threshold.
    if (!token_owned_ && !reached_eof_ &&
        prefetch_buffer_.size() <
            prefetch_threshold_ + output_buffer_.size() + 1) {
      auto task = [iterator = FormRef(this), exec_ctx]() {
        TFRT_TRACE_SCOPE("ReadIOSource");
        iterator->ReadIOSource(exec_ctx);
      };
      // This call can fail if the work queue is full.
      if (host->EnqueueBlockingWork(std::move(task))) {
        // The task which is successfully scheduled in the blocking threadpool
        // should own the token.
        token_owned_ = true;
      }
    }
    // Optimize the fast path. Return the first value from prefetched buffer if
    // there is no other pending output value and there is prefetched value
    // available.
    //
    // IDEA(donglin): we can further optimize the fast path by not having
    // GetNext() grab lock in the common case. This can be achieved by moving
    // values from the prefetch_buffer to another buffer that can only be
    // accessed by the caller of GetNext(), so that the GetNext() can return
    // value from this buffer directly if it is non-empty.
    if (!prefetch_buffer_.empty() && output_buffer_.empty()) {
      auto input = std::move(prefetch_buffer_.front());
      prefetch_buffer_.pop();
      // An IterationResult from GetNextElement() should only contain available
      // AsyncValues.
      assert(input.eof.IsAvailable());
      if (input.eof.IsError()) {
        input.eof.GetAsyncValue()->SetErrorLocationIfUnset(
            exec_ctx.location().Decode());
        host->EmitError(input.eof.GetError());
      }
      return input;
    }
  }

  llvm::SmallVector<RCReference<AsyncValue>, 1> result_values;
  result_values.resize(1);
  // The IndirectAsyncValue might be filled later by the background blocking
  // thread.
  result_values[0] = MakeIndirectAsyncValue(host);
  auto result_eof = MakeUnconstructedAsyncValueRef<bool>(host);
  auto result =
      IterationResult::Pending(std::move(result_values), std::move(result_eof));
  {
    mutex_lock lock(mu_);
    output_buffer_.push(result.CopyRef());
    // Caller thread has to read data from the underlying IO source because it
    // fails to enqueue a blocking task to read the data.
    if (!reached_eof_ && !token_owned_ && prefetch_buffer_.empty()) {
      // If output buffer size is not 1, it must be at least 2. Since the
      // prefetch buffer is empty, it follows that at the end of the previous
      // GetNext(...) call, output_buffer_size > prefetch_buffer_size. According
      // to the document of GetNext(...), the last GetNext(...) call should
      // have successfully enqueued a blocking task and the token should be
      // owned. This contradicts the condition here that token_owned_ == false.
      assert(output_buffer_.size() == 1);
      // Read the next element from IO source directly so that
      // output_buffer_size == prefetch_buffer_size.
      auto input = GetNextElement(exec_ctx);
      prefetch_buffer_.push(std::move(input));
    }
  }

  MaterializeOutputs(exec_ctx);
  return result;
}

void PrefetchingIterator::ReadIOSource(const ExecutionContext& exec_ctx) {
  while (true) {
    int64_t fetch_num;
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
      fetch_num =
          max_prefetch_num_ + output_buffer_.size() - prefetch_buffer_.size();
    }
    for (int32_t i = 0; i < fetch_num; ++i) {
      if (exec_ctx.IsCancelled()) return;
      // Since only one thread can own the token, we can access the underlying
      // IO source without a lock.
      auto input = GetNextElement(exec_ctx);
      if (input.eof.IsConcrete() && input.eof.get()) {
        mutex_lock lock(mu_);
        reached_eof_ = true;
        break;
      }
      bool need_to_materialize_output = false;
      {
        mutex_lock lock(mu_);
        need_to_materialize_output =
            prefetch_buffer_.empty() && !output_buffer_.empty();
        prefetch_buffer_.push(std::move(input));
      }
      // If prefetch_buffer was empty and output_buffer is not empty, it is
      // possible that data pipeline's control flow is blocked waiting for the
      // value in the output_buffer to be updated. For example, the
      // InterleaveDatasetIterator might be waiting for the EOF of the first
      // IterationResult in the output_buffer to be available before it can
      // decide whether to propagate this IterationResult to its caller.
      // Therefore we should immediately update values in the output_buffer to
      // unblock the control flow. Otherwise, the control flow is not blocked on
      // the current iterator's output and it is preferred to let the control
      // flow thread update values in the output_buffer.
      if (need_to_materialize_output) MaterializeOutputs(exec_ctx);
    }
    MaterializeOutputs(exec_ctx);
  }
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
  auto* output_eof = output.eof.GetAsyncValue();
  assert(input_eof->IsAvailable());
  if (input_eof->IsError()) {
    // Set error location and emit error before forwarding the error to the
    // output_eof.
    // TODO(donglin): If the error happens after the BEF executor has finished
    // its tasks, the LocationHandler might have been de-allocated at this
    // moment and this line can throw SIGSEGV. Fix it.
    input_eof->SetErrorLocationIfUnset(exec_ctx.location().Decode());
    exec_ctx.host()->EmitError(input.eof.GetError());
    output_eof->SetError(input_eof->GetError());
  } else {
    output_eof->emplace<bool>(input_eof->get<bool>());
  }
}

void PrefetchingIterator::MaterializeOutputs(const ExecutionContext& exec_ctx) {
  llvm::SmallVector<std::pair<IterationResult, IterationResult>, 4> pairs;
  while (true) {
    {
      mutex_lock lock(mu_);
      while (!prefetch_buffer_.empty() && !output_buffer_.empty()) {
        auto input = std::move(prefetch_buffer_.front());
        auto output = std::move(output_buffer_.front());
        prefetch_buffer_.pop();
        output_buffer_.pop();
        pairs.push_back(std::make_pair(std::move(input), std::move(output)));
      }
    }
    if (pairs.empty()) break;
    // Materialize outputs in reverse order in order to reduce the number of
    // AndThen(...) calls in the upstream datasets.
    for (int i = pairs.size() - 1; i >= 0; --i) {
      ForwardInputToOutput(std::move(pairs[i].first),
                           std::move(pairs[i].second), exec_ctx);
    }
    pairs.clear();
  }
  if (ReachedEof()) {
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
