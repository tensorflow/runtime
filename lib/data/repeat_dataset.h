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

// This file declares RepeatDataset class which wraps around another Dataset
// instance and repeats it a specified number of times.

#ifndef TFRT_DATA_REPEAT_DATASET_H_
#define TFRT_DATA_REPEAT_DATASET_H_

#include <queue>

#include "tfrt/data/dataset.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace data {

class RepeatDatasetIterator;

// RepeatDataset wraps around another Dataset instance and repeats it for a
// specified number of times.
class RepeatDataset : public Dataset {
 public:
  explicit RepeatDataset(RCReference<Dataset> input_dataset, int64_t count,
                         HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        count_(count),
        host_(host),
        allocator_(host->allocator()) {}

  // This class is not copyable or movable.
  RepeatDataset(const RepeatDataset&) = delete;
  RepeatDataset& operator=(const RepeatDataset&) = delete;

  RCReference<Iterator> MakeIterator(const IteratorContext& context) override;

 private:
  friend class RepeatDatasetIterator;

  void Destroy() override {
    internal::DestroyImpl<RepeatDataset>(this, allocator_);
  }

  RCReference<Dataset> input_dataset_;
  int64_t count_;
  HostContext* host_;
  HostAllocator* allocator_;
};

class RepeatDatasetIterator : public Iterator {
 public:
  explicit RepeatDatasetIterator(RCReference<RepeatDataset> dataset,
                                 const IteratorContext& context)
      : Iterator(),
        parent_dataset_(std::move(dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator(context)),
        context_(context),
        token_owned_(false) {}

  // This class is not copyable or movable.
  RepeatDatasetIterator(const RepeatDatasetIterator&) = delete;
  RepeatDatasetIterator& operator=(const RepeatDatasetIterator&) = delete;

  IterationResult GetNext(const ExecutionContext& exec_ctx) override;

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
    assert(!output_buffer_.empty());
    auto value = std::move(output_buffer_.front());
    output_buffer_.pop();
    return value;
  }

  RCReference<RepeatDataset> parent_dataset_;
  RCReference<Iterator> input_iterator_;
  const IteratorContext context_;

  mutex mu_;
  int arity_ = -1;
  // A queue of IterationResult that from the `input_iterator_`.
  std::queue<IterationResult> input_buffer_;
  // A queue of IterationResult that have already been returned to the
  // GetNext(...) caller.
  std::queue<IterationResult> output_buffer_ TFRT_GUARDED_BY(mu_);
  // The number of times the underlying iterator is repeated.
  int64_t current_count_ = 0;

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
