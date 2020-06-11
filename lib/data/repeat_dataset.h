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

//===- repeat_dataset.h -----------------------------------------*- C++ -*-===//
//
// This file declares RepeatDataset class which wraps around another Dataset
// instance and repeats it a specified number of times.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DATA_REPEAT_DATASET_H_
#define TFRT_DATA_REPEAT_DATASET_H_

#include <queue>

#include "dataset.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace data {

class RepeatDatasetIterator;

// RepeatDataset wraps around another Dataset instance and repeats it for a
// specified number of times.
class RepeatDataset : public Dataset {
 public:
  explicit RepeatDataset(RCReference<Dataset> input_dataset, int64_t epochs,
                         HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        epochs_(epochs),
        host_(host),
        allocator_(host->allocator()) {
    // TODO(rachelim): Support infinite iteration.
    assert(epochs > 0);
  }

  // This class is not copyable or movable.
  RepeatDataset(const RepeatDataset&) = delete;
  RepeatDataset& operator=(const RepeatDataset&) = delete;

  RCReference<Iterator> MakeIterator() override;

 private:
  friend class RepeatDatasetIterator;

  void Destroy() override {
    internal::DestroyImpl<RepeatDataset>(this, allocator_);
  }

  RCReference<Dataset> input_dataset_;
  int64_t epochs_;
  HostContext* host_;
  HostAllocator* allocator_;
};

class RepeatDatasetIterator : public Iterator {
 public:
  explicit RepeatDatasetIterator(RCReference<RepeatDataset> dataset)
      : Iterator(),
        parent_dataset_(std::move(dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()),
        token_owned_(false) {}

  // This class is not copyable or movable.
  RepeatDatasetIterator(const RepeatDatasetIterator&) = delete;
  RepeatDatasetIterator& operator=(const RepeatDatasetIterator&) = delete;

  IterationResult GetNext(const ExecutionContext& exec_ctx) override {
    auto* host = exec_ctx.host();

    // Initialize value_count using the first value from the input_iterator_.
    if (value_count_ < 0) {
      mutex_lock lock(mu_);
      assert(value_count_ < 0);
      assert(!token_owned_);
      auto input = input_iterator_->GetNext(exec_ctx);
      value_count_ = input.values.size();
      input_buffer_.push(std::move(input));
    }

    llvm::SmallVector<RCReference<AsyncValue>, 4> result_values;
    result_values.resize(value_count_);
    for (size_t i = 0; i < value_count_; ++i) {
      result_values[i] = host->MakeIndirectAsyncValue();
    }
    auto result_eof = host->MakeUnconstructedAsyncValueRef<bool>();
    auto result = IterationResult::Pending(std::move(result_values),
                                           std::move(result_eof));
    {
      mutex_lock lock(mu_);
      output_buffer_.push(result.CopyRef());
    }

    MaybeScheduleBackgroundTask(exec_ctx, false, 0);
    return result;
  }

 private:
  void Destroy() override {
    internal::DestroyImpl<RepeatDatasetIterator>(this,
                                                 parent_dataset_->allocator_);
  }

  // This method ensures that the control flow (e.g. get value from the
  // input_iterator_) is only executed by the thread that holds the token. It
  // helps ensure in-order delivery while still allowing an unblocking
  // GetNext(...) API.
  //
  // If there is no value in the `output_buffer_`, or if the caller can not get
  // the token to run the control flow logic, this method will not schedule any
  // background task. Otherwise, this method will schedule background task as
  // appropriate. This method will recursively call itself again when the first
  // value in the `input_buffer_` becomes available.
  void MaybeScheduleBackgroundTask(const ExecutionContext& exec_ctx,
                                   bool is_token_owner, int callback_count)
      TFRT_EXCLUDES(mu_);

  void HandleEofAvailableInput(IterationResult input, HostContext* host);

  int OutputBufferSize() TFRT_EXCLUDES(mu_) {
    mutex_lock lock(mu_);
    return output_buffer_.size();
  }

  IterationResult DequeueOutputBuffer() TFRT_EXCLUDES(mu_) {
    mutex_lock lock(mu_);
    auto value = std::move(output_buffer_.front());
    output_buffer_.pop();
    return value;
  }

  RCReference<RepeatDataset> parent_dataset_;
  RCReference<Iterator> input_iterator_;

  mutex mu_;
  int value_count_ = -1;
  // A queue of IterationResult that from the `input_iterator_`.
  std::queue<IterationResult> input_buffer_;
  // A queue of IterationResult that have already been returned to the
  // GetNext(...) caller.
  std::queue<IterationResult> output_buffer_ TFRT_GUARDED_BY(mu_);
  // The current epoch number.
  int64_t epoch_ = 0;

  // This is a unique logical token for this iterator instance. It effectively
  // acts as a lock to ensure in-order delivery of results by guaranteeing that
  // at most one thread can take the next value from the input_iterator_ and
  // update values in output_buffer_.
  //
  // The thread which changes token_owned from false to true "holds" the token,
  // and can pass it on to a thread that runs the callback it schedules. The
  // token is released when token_owned_ is changed from true to false.
  bool token_owned_ TFRT_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_DATA_REPEAT_DATASET_H_
