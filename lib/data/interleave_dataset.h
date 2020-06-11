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

#include "dataset.h"
#include "tfrt/host_context/function.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace data {

class InterleaveDatasetIterator;

// InterleaveDataset maps a user-defined function over the elements in its input
// dataset and interleaves the results. The user-defined function is expected to
// have a single return value of type Dataset.
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
                             RCReference<const Function> map_fn, int64_t arity,
                             HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        cycle_length_(cycle_length),
        block_length_(block_length),
        arity_(arity),
        host_(host),
        allocator_(host->allocator()),
        map_fn_(std::move(map_fn)) {}

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
  const int64_t arity_;
  HostContext* host_;
  HostAllocator* allocator_;
  RCReference<const Function> map_fn_;
};

class InterleaveDatasetIterator : public Iterator {
 public:
  explicit InterleaveDatasetIterator(
      RCReference<InterleaveDataset> parent_dataset)
      : Iterator(),
        parent_dataset_(std::move(parent_dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()),
        cycle_iterators_(parent_dataset_->cycle_length_) {}

  // This class is not copyable or movable.
  InterleaveDatasetIterator(const InterleaveDatasetIterator&) = delete;
  InterleaveDatasetIterator& operator=(const InterleaveDatasetIterator&) =
      delete;

  IterationResult GetNext(const ExecutionContext& exec_ctx) override;

 private:
  void Destroy() override {
    internal::DestroyImpl<InterleaveDatasetIterator>(
        this, parent_dataset_->allocator_);
  }

  // Advance the next block index. If the next block index exceeds the block
  // length, advance to the next iterator in the cycle.
  void AdvanceBlockIndex() {
    ++block_index_;
    if (block_index_ == parent_dataset_->block_length_) {
      AdvanceCycleIndex();
    }
  }

  // Advance to the next iterator in the cycle and reset block_index_ to 0.
  void AdvanceCycleIndex() {
    block_index_ = 0;
    cycle_index_ = (cycle_index_ + 1) % parent_dataset_->cycle_length_;
  }

  RCReference<Iterator> MakeIteratorFromInputElement(
      llvm::SmallVector<RCReference<AsyncValue>, 4> input_element,
      const ExecutionContext& exec_ctx);

  RCReference<InterleaveDataset> parent_dataset_;
  RCReference<Iterator> input_iterator_;

  std::vector<RCReference<Iterator>> cycle_iterators_;
  size_t cycle_index_ = 0;
  size_t block_index_ = 0;
  bool end_of_input_ = false;
  size_t num_open_ = 0;  // Number of open iterators.
};

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_INTERLEAVE_DATASET_H_
