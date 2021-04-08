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

// This file declares RangeDataset class which yields a step-separated range of
// values.

#ifndef TFRT_LIB_DATA_RANGE_DATASET_H_
#define TFRT_LIB_DATA_RANGE_DATASET_H_

#include "tfrt/data/dataset.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace data {

class RangeDatasetIterator;

// RangeDataset yields a step-separated range of values from start (inclusive)
// to stop (exclusive).
class RangeDataset : public Dataset {
 public:
  explicit RangeDataset(int64_t start, int64_t stop, int64_t step,
                        DType element_type, HostContext* host)
      : start_(start),
        stop_(stop),
        step_(step),
        element_type_(element_type),
        host_(host),
        allocator_(host->allocator()) {}

  // This class is not copyable or movable.
  RangeDataset(const RangeDataset&) = delete;
  RangeDataset& operator=(const RangeDataset&) = delete;

  RCReference<Iterator> MakeIterator(const IteratorContext& context) override;

 private:
  friend class RangeDatasetIterator;

  void Destroy() override {
    internal::DestroyImpl<RangeDataset>(this, allocator_);
  }

  const int64_t start_;
  const int64_t stop_;
  const int64_t step_;
  const DType element_type_;
  HostContext* host_;
  HostAllocator* allocator_;
};

class RangeDatasetIterator : public Iterator {
 public:
  explicit RangeDatasetIterator(RCReference<RangeDataset> dataset)
      : Iterator(), dataset_(std::move(dataset)), next_(dataset_->start_) {}

  // This class is not copyable or movable.
  RangeDatasetIterator(const RangeDatasetIterator&) = delete;
  RangeDatasetIterator& operator=(const RangeDatasetIterator&) = delete;

  IterationResult GetNext(const ExecutionContext& exec_ctx) override;

 private:
  void Destroy() override {
    internal::DestroyImpl<RangeDatasetIterator>(this, dataset_->allocator_);
  }

  RCReference<RangeDataset> dataset_;
  int64_t next_;
};

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_RANGE_DATASET_H_
