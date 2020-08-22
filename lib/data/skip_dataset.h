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

//===- skip_dataset.h -------------------------------------------*- C++ -*-===//
//
// This file declares SkipDataset class which wraps around another Dataset
// instance and skips a specified number of elements from that dataset.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DATA_SKIP_DATASET_H_
#define TFRT_DATA_SKIP_DATASET_H_

#include "tfrt/data/dataset.h"

namespace tfrt {
namespace data {

class SkipDatasetIterator;

// RepeatDataset wraps around another Dataset instance and skips `count`
// elements from that dataset.
class SkipDataset : public Dataset {
 public:
  explicit SkipDataset(RCReference<Dataset> input_dataset, int64_t count,
                       HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        count_(count),
        host_(host),
        allocator_(host->allocator()) {}

  // This class is not copyable or movable.
  SkipDataset(const SkipDataset&) = delete;
  SkipDataset& operator=(const SkipDataset&) = delete;

  RCReference<Iterator> MakeIterator() override;

 private:
  friend class SkipDatasetIterator;

  void Destroy() override {
    internal::DestroyImpl<SkipDataset>(this, allocator_);
  }

  RCReference<Dataset> input_dataset_;
  int64_t count_;
  HostContext* host_;
  HostAllocator* allocator_;
};

class SkipDatasetIterator : public Iterator {
 public:
  explicit SkipDatasetIterator(RCReference<SkipDataset> dataset)
      : Iterator(),
        parent_dataset_(std::move(dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()) {}

  // This class is not copyable or movable.
  SkipDatasetIterator(const SkipDatasetIterator&) = delete;
  SkipDatasetIterator& operator=(const SkipDatasetIterator&) = delete;

  IterationResult GetNext(const ExecutionContext& exec_ctx) override;

 private:
  void Destroy() override {
    internal::DestroyImpl<SkipDatasetIterator>(this,
                                               parent_dataset_->allocator_);
  }

  RCReference<SkipDataset> parent_dataset_;
  RCReference<Iterator> input_iterator_;
  // The skip operation is only needed for the first GetNext() call.
  bool is_skip_done_ = false;
};

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_DATA_SKIP_DATASET_H_
