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

//===- interleave_dataset.h -------------------------------------*- C++ -*-===//
//
// This file declares InterleaveDataset class, which applies a function to its
// input to create a dataset per input element, and interleaves the results of
// these datasets.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LIB_DATA_INTERLEAVE_DATASET_H_
#define TFRT_LIB_DATA_INTERLEAVE_DATASET_H_

#include <queue>

#include "tfrt/data/dataset.h"
#include "tfrt/host_context/function.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace data {

class InterleaveDatasetIterator;

// InterleaveDataset applies a user-defined function over the values from its
// input dataset to construct intermediate iterators, then interleaves the
// values from those intermediate iterators. The user-defined function is
// expected to return a single value of type Dataset.
//
// If there is any error from the input data, or if there is error from calling
// func_(input), the error will be output via GetNext(...).
//
// The `cycle_length` and `block_length` arguments control the order in which
// elements are produced. `cycle_length` controls the number of input elements
// that are processed concurrently. If `cycle_length` is 1, this
// transformation will handle one input element at a time, and is equivalent
// to performing a flat map. In general, this transformation will apply the
// map function to `cycle_length` input elements, open iterators on the
// returned Dataset objects, and cycle through them, producing `block_length`
// consecutive elements from each iterator, and consuming the next input
// element each time it reaches the end of an iterator.
class InterleaveDataset : public Dataset {
 public:
  explicit InterleaveDataset(RCReference<Dataset> input_dataset,
                             int64_t cycle_length, int64_t block_length,
                             RCReference<const Function> func, int64_t arity,
                             HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        cycle_length_(cycle_length),
        block_length_(block_length),
        prefetch_iterator_num_(cycle_length),
        arity_(arity),
        host_(host),
        allocator_(host->allocator()),
        func_(std::move(func)) {
    assert(cycle_length_ > 0);
    assert(block_length_ > 0);
  }

  // This class is not copyable or movable.
  InterleaveDataset(const InterleaveDataset&) = delete;
  InterleaveDataset& operator=(const InterleaveDataset&) = delete;

  RCReference<Iterator> MakeIterator() override;

 private:
  // Allow iterator to rely on private data members of this dataset.
  friend class InterleaveDatasetIterator;

  void Destroy() override {
    internal::DestroyImpl<InterleaveDataset>(this, allocator_);
  }

  RCReference<Dataset> input_dataset_;
  const int64_t cycle_length_;
  const int64_t block_length_;
  // Pre-initialize up to `cycle_length` intermediate iterators, in addition to
  // initializing the intermediate iterators already requested by the current
  // cycle.
  const int64_t prefetch_iterator_num_;
  const int64_t arity_;
  HostContext* host_;
  HostAllocator* allocator_;
  RCReference<const Function> func_;
};

class InterleaveDatasetIterator : public Iterator {
 public:
  explicit InterleaveDatasetIterator(
      RCReference<InterleaveDataset> parent_dataset)
      : Iterator(),
        parent_dataset_(std::move(parent_dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()),
        token_owned_(false) {
    for (int i = 0; i < parent_dataset_->cycle_length_; ++i) {
      iterator_and_queues_.push_back(
          IteratorAndQueue::DummyValue((parent_dataset_->host_)));
    }
  }

  // This class is not copyable or movable.
  InterleaveDatasetIterator(const InterleaveDatasetIterator&) = delete;
  InterleaveDatasetIterator& operator=(const InterleaveDatasetIterator&) =
      delete;

  IterationResult GetNext(const ExecutionContext& exec_ctx) override;

 private:
  // This struct contains the state for an intermediate iterator. The state is
  // needed by InterleaveDatasetIterator to return results in the expected order
  // and still be able to fetch multiple values from the intermeidate iterators
  // without knowing whether these iterators have reached eof.
  struct IteratorAndQueue {
    IteratorAndQueue(IterationResult r, RCReference<AsyncValue> d, bool o)
        : input_value(std::move(r)), dataset(std::move(d)), is_open(o) {}

    // Creates an IteratorAndQueue with input_value.eof=true and is_open=false.
    static IteratorAndQueue DummyValue(HostContext* host) {
      return IteratorAndQueue(IterationResult::Eof(host, 0),
                              RCReference<AsyncValue>(), false);
    }

    // The value from the input_iterator_ which can be used create the dataset.
    IterationResult input_value;
    // The first value prefetched from the iterator. This is needed to let the
    // intermediate iterator starts prefetching right after the iterator is
    // constructed.
    AsyncValueRef<IterationResult> prefetched_value;
    // The dataset by calling func_(input_value).
    RCReference<AsyncValue> dataset;
    // The intermediate iterator created from the above dataset. Note that it
    // can not be AsyncValueRef<Iterator> because Iterator is an abstract class.
    AsyncValueRef<RCReference<Iterator>> iterator;
    // A queue of results from the above intermediate iterator.
    std::queue<IterationResult> queue;
    // The number of results fetched in the current fetch-block of this iterator
    // This means fetched_num_in_block < block_length.
    size_t fetched_num_in_block = 0;
    // The number of results output from the current output-block of this
    // iterator. This means output_num_in_block < block_length.
    size_t output_num_in_block = 0;
    // Whether more values can be fetched from this IteratorAndQueue.
    // This is true iff all the conditions below hold:
    // 1) input_value.eof is false without error.
    // 2) dataset is created without error.
    // 3) Either the iterator has not reached end or the queue is not empty.
    bool is_open;
  };

  void Destroy() override {
    internal::DestroyImpl<InterleaveDatasetIterator>(
        this, parent_dataset_->allocator_);
  }

  // This method ensures that the control flow (e.g. get value from the
  // input_iterator_) is only executed by the thread that holds the token. It
  // helps ensure in-order delivery while still allowing an unblocking
  // GetNext(...) API.
  //
  // If there is no value in the `output_buffer_*`, or if the caller can not get
  // the token to run the control flow logic, this method will not schedule any
  // background task. Otherwise, this method will schedule background task as
  // appropriate.
  void MaybeScheduleBackgroundTask(const ExecutionContext& exec_ctx,
                                   bool is_token_owner, int callback_count)
      TFRT_EXCLUDES(mu_);

  // If the input iterator has not reached end, prefetch enough values from it
  // and transform those values into intermediate iterators until
  // num_open_iterators_ == cycle_length_ + prefetch_iterator_num_.
  void PreInitializeIntermediateIterators(const ExecutionContext& exec_ctx)
      TFRT_EXCLUDES(mu_);

  // This method contains the following logic:
  // 1) Fetch value from the input_iterator_ and call dataset = func_(value).
  // 2) Create iterator from the dataset when the dataset is available.
  // 3) Identify the correct intermediate iterator to fetch from based on the
  //    block/cycle config. Fetch value from this iterator into its queue.
  // 4) If there is any error from the input_iterator_, or if there is error
  //    from calling func_(input), the method propagates the error to the next
  //    value in the output_buffer_*.
  //
  // If there is any unavailable async value that stops this method from
  // fetching values, e.g. input value is not available to create
  // dataset, the method returns an pointer to that AsyncValue. Otherwise it
  // returns a nullptr when there is no more value to fetch.
  AsyncValue* FetchInputValues(const ExecutionContext& exec_ctx)
      TFRT_EXCLUDES(mu_);

  // This method identifies the correct intermediate iterator that should be
  // used fill the next value in output_buffer_* and keeps forwarding fetched
  // values to the values in the output_buffer_* as much as possible.
  //
  // It repeats the following steps until either it encounters a value with
  // unavailable eof or when total_queues_size_ == 0:
  // 1) If the eof of the value at the front of its queue is unavailable, the
  //    method returns an pointer to that eof's AsyncValue.
  // 2) If the eof of the value at the front of its queue has error, the method
  //    propagates the error to the next value in the output_buffer_* and pops
  //    both values out of their queues.
  // 3) If the eof of the value at the front of its queue is true, the method
  //    marks set is_open=false for this iterator and identifies the next
  //    iterator in the cycle.
  // 4) If the eof of the value at the front of its queue is false, the method
  //    forwards this value to the next value in the output_buffer_* and pops
  //    both values out of their queues.
  AsyncValue* FillOutputValues(const ExecutionContext& exec_ctx)
      TFRT_EXCLUDES(mu_);

  // Return the total number of values in the output buffers.
  int OutputBufferSize() TFRT_EXCLUDES(mu_) {
    mutex_lock lock(mu_);
    return output_buffer_back_.size() + output_buffer_front_.size();
  }

  // Return the next value in the output buffer. Values in the
  // output_buffer_front_ should be returned before those values in
  // the output_buffer_back_.
  IterationResult DequeueOutputBuffer() TFRT_EXCLUDES(mu_) {
    if (output_buffer_front_.empty()) {
      mutex_lock lock(mu_);
      std::swap(output_buffer_front_, output_buffer_back_);
    }
    assert(!output_buffer_front_.empty());
    auto value = std::move(output_buffer_front_.front());
    output_buffer_front_.pop();
    return value;
  }

  RCReference<InterleaveDataset> parent_dataset_;
  RCReference<Iterator> input_iterator_;
  bool is_input_iterator_eof_ = false;

  // List of intermediate iterators and their states. The positions of those
  // intermediate iterators have been fixed in the iterator cycle and thus we
  // can get values from them.
  std::vector<IteratorAndQueue> iterator_and_queues_;
  // Buffer of prefetched intermediate iterators. The iterator in this buffer
  // should be moved to iterator_and_queues_ before we can get values from it.
  std::queue<IteratorAndQueue> prefetched_iterators_;
  // Total number of iterators in prefetched_iterators_ and iterator_and_queues_
  // whose is_open is true.
  size_t num_open_iterators_ = 0;
  // iterator_and_queues_[iterator_index_for_fetch_] contains the next
  // iterator to fetch from.
  size_t iterator_index_for_fetch_ = 0;
  // iterator_and_queues_[iterator_index_for_output_] contains the value to fill
  // the next value in output_buffer_*.
  size_t iterator_index_for_output_ = 0;
  // Total size of queues across all iterators whose is_open is true.
  size_t total_queues_size_ = 0;

  mutex mu_;
  // A queue of unavailable IterationResult enqueued by the caller of GetNext().
  // This queue must be accessed with the mutex because it can be accessed by
  // both GetNext() caller and the blocking threadpool.
  std::queue<IterationResult> output_buffer_back_ TFRT_GUARDED_BY(mu_);
  // A queue of unavailable IterationResult that are moved from
  // output_buffer_back_. This queue can be accessed without the mutex because
  // only the token owner can access it.
  std::queue<IterationResult> output_buffer_front_;
  // This is a unique logical token for this iterator instance. It effectively
  // acts as a lock to ensure in-order delivery of results by guaranteeing that
  // at most one thread can take the next value from the input_iterator_ and
  // update values in output_buffer_front_.
  //
  // The thread which changes token_owned from false to true "holds" the token,
  // and can pass it on to a thread that runs the callback it schedules. The
  // token is released when token_owned_ is changed from true to false.
  bool token_owned_ TFRT_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_INTERLEAVE_DATASET_H_
