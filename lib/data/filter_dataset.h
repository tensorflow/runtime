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

// This file declares FilterDataset class which wraps around another Dataset
// instance and outputs those elements from the underlying dataset which satisfy
// a user-defined filter function.

#ifndef TFRT_LIB_DATA_FILTER_DATASET_H_
#define TFRT_LIB_DATA_FILTER_DATASET_H_

#include <queue>

#include "map_dataset.h"
#include "tfrt/data/dataset.h"
#include "tfrt/host_context/function.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace data {

class FilterDatasetIterator;

// FilterDataset takes elements from the underlying dataset and outputs those
// elements which satisfy a user-defined filter function.
class FilterDataset : public Dataset {
 public:
  explicit FilterDataset(RCReference<Dataset> input_dataset,
                         RCReference<const Function> filter_fn,
                         HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        host_(host),
        allocator_(host->allocator()),
        arity_(filter_fn->num_arguments()),
        filter_fn_(std::move(filter_fn)) {}

  // This class is not copyable or movable.
  FilterDataset(const FilterDataset&) = delete;
  FilterDataset& operator=(const FilterDataset&) = delete;

  RCReference<Iterator> MakeIterator(const IteratorContext& context) override;

 private:
  // Allow iterator to rely on private data members of this dataset.
  friend class FilterDatasetIterator;

  void Destroy() override {
    internal::DestroyImpl<FilterDataset>(this, allocator_);
  }

  RCReference<Dataset> input_dataset_;
  HostContext* host_;
  HostAllocator* allocator_;
  const int64_t arity_;
  // The function should take value from the `input_dataset_` as input and
  // then output a boolean value.
  RCReference<const Function> filter_fn_;
};

class FilterDatasetIterator : public Iterator {
 public:
  explicit FilterDatasetIterator(RCReference<FilterDataset> parent_dataset,
                                 const IteratorContext& context)
      : Iterator(),
        parent_dataset_(std::move(parent_dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator(context)),
        num_false_predicate_(0),
        token_owned_(false) {}

  IterationResult GetNext(const ExecutionContext& exec_ctx) override;

 private:
  // This class is not copyable or movable.
  FilterDatasetIterator(const FilterDatasetIterator&) = delete;
  FilterDatasetIterator& operator=(const FilterDatasetIterator&) = delete;

  void Destroy() override {
    internal::DestroyImpl<FilterDatasetIterator>(this,
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
  // value in the `input_and_predicate_buffer_` becomes available.
  void MaybeScheduleBackgroundTask(const ExecutionContext& exec_ctx,
                                   bool is_token_owner, int callback_count)
      TFRT_EXCLUDES(mu_);

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

  RCReference<FilterDataset> parent_dataset_;
  RCReference<Iterator> input_iterator_;
  mutex mu_;
  // A queue of IterationResult pairs. The first value of each pair is a value
  // from the `input_iterator_`. The second value of each pair is the result of
  // applying `filter_fn_` to the first value.
  std::queue<std::pair<IterationResult, IterationResult>>
      input_and_predicate_buffer_;
  // A queue of IterationResult that have already been returned to the
  // GetNext(...) caller.
  std::queue<IterationResult> output_buffer_ TFRT_GUARDED_BY(mu_);
  // The number of IterationResult in the input_and_predicate_buffer_ whose
  // predicate value is false. `num_false_predicate_` might be negative if
  // fetch_sub(...) is invoked before fetch_add(...).
  std::atomic<int32_t> num_false_predicate_;
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

#endif  // TFRT_LIB_DATA_FILTER_DATASET_H_
