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

// This file declares the ShuffleDataset class which wraps around another
// Dataset instance and shuffles its values before outputting those values.

#ifndef TFRT_DATA_SHUFFLE_DATASET_H_
#define TFRT_DATA_SHUFFLE_DATASET_H_

#include "tfrt/data/dataset.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/philox_random.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace data {

class ShuffleDatasetIterator;

// ShuffleDataset wraps around another Dataset instance and shuffles its values
// before outputting those values.
class ShuffleDataset : public Dataset {
 public:
  explicit ShuffleDataset(RCReference<Dataset> input_dataset,
                          int64_t buffer_size, int64_t seed, int64_t seed2,
                          HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        buffer_size_(buffer_size),
        seed_(seed),
        seed2_(seed2),
        host_(host),
        allocator_(host->allocator()) {
    assert(buffer_size_ > 0);
  }

  // This class is not copyable or movable.
  ShuffleDataset(const ShuffleDataset&) = delete;
  ShuffleDataset& operator=(const ShuffleDataset&) = delete;

  RCReference<Iterator> MakeIterator(const IteratorContext& context) override;

 private:
  friend class ShuffleDatasetIterator;

  void Destroy() override {
    internal::DestroyImpl<ShuffleDataset>(this, allocator_);
  }

  RCReference<Dataset> input_dataset_;
  int64_t buffer_size_;
  int64_t seed_;
  int64_t seed2_;
  HostContext* host_;
  HostAllocator* allocator_;
};

class ShuffleDatasetIterator : public Iterator {
 public:
  explicit ShuffleDatasetIterator(RCReference<ShuffleDataset> dataset,
                                  const IteratorContext& context)
      : Iterator(),
        parent_dataset_(std::move(dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator(context)),
        random_(parent_dataset_->seed_, parent_dataset_->seed2_) {}

  // This class is not copyable or movable.
  ShuffleDatasetIterator(const ShuffleDatasetIterator&) = delete;
  ShuffleDatasetIterator& operator=(const ShuffleDatasetIterator&) = delete;

  IterationResult GetNext(const ExecutionContext& exec_ctx) override;

 private:
  void Destroy() override {
    internal::DestroyImpl<ShuffleDatasetIterator>(this,
                                                  parent_dataset_->allocator_);
  }

  // This method ensures that the control flow (e.g. get value from the
  // input_iterator_) is only executed by the thread that holds the token. It
  // helps ensure in-order delivery while still allowing an unblocking
  // GetNext(...) API.
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

  // If the last non-empty value in the circular buffer `num_shuffled_values_`
  // has reached EOF, remove this value from the circular buffer and set
  // `reached_eof_` to true.
  void CheckEof();

  RCReference<ShuffleDataset> parent_dataset_;
  RCReference<Iterator> input_iterator_;
  random::PhiloxRandom random_;

  mutex mu_;
  // The number of values in the IterationResult returned by this iterator.
  int arity_ = -1;
  // True iff the input_iterator_ has reached EOF.
  bool reached_eof_ = false;
  // A circular buffer of IterationResult from the `input_iterator_`.
  std::vector<IterationResult> shuffle_buffer_;
  // Number of non-empty values in the circular shuffle_buffer_.
  size_t num_shuffled_values_ = 0;
  // The index of the first value in the circular shuffle_buffer_.
  size_t start_index_ = 0;
  // A queue of IterationResult that have already been returned to the
  // GetNext(...) caller.
  std::queue<IterationResult> output_buffer_ TFRT_GUARDED_BY(mu_);

  // This is a unique logical token for this iterator instance. It effectively
  // acts as a lock to ensure in-order delivery of results by guaranteeing that
  // at most one thread can take the next value from the input_iterator_ and
  // update values in output_buffer_.
  //
  // The thread which changes token_owned from false to true "holds" the token,
  // and can pass it on to a thread that runs the callback it schedules. The
  // token is released when token_owned_ is changed from true to false.
  bool token_owned_ TFRT_GUARDED_BY(mu_) = false;
};

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_DATA_SHUFFLE_DATASET_H_
