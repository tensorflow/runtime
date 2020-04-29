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

//===- range_dataset.h ------------------------------------------*- C++ -*-===//
//
// This file declares RangeDataset class which yields a step-separated range of
// values.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LIB_DATA_RANGE_DATASET_H_
#define TFRT_LIB_DATA_RANGE_DATASET_H_

#include "dataset.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace data {

template <typename T>
class RangeDatasetIterator;

// RangeDataset yields a step-separated range of values from start (inclusive)
// to stop (exclusive).
template <typename T>
class RangeDataset : public Dataset<T> {
 public:
  explicit RangeDataset(T start, T stop, T step, HostContext* host)
      : start_(start),
        stop_(stop),
        step_(step),
        host_(host),
        allocator_(host->allocator()) {}

  // This class is not copyable or movable.
  RangeDataset(const RangeDataset&) = delete;
  RangeDataset& operator=(const RangeDataset&) = delete;

  RCReference<Iterator<T>> MakeIterator() override;

 private:
  friend class RangeDatasetIterator<T>;

  void Destroy() override {
    internal::DestroyImpl<RangeDataset<T>>(this, allocator_);
  }

  const T start_;
  const T stop_;
  const T step_;
  HostContext* host_;
  HostAllocator* allocator_;
};

template <typename T>
class RangeDatasetIterator : public Iterator<T> {
 public:
  explicit RangeDatasetIterator(RCReference<RangeDataset<T>> dataset)
      : Iterator<T>(dataset->host_),
        dataset_(std::move(dataset)),
        next_(dataset_->start_) {}

  // This class is not copyable or movable.
  RangeDatasetIterator(const RangeDatasetIterator&) = delete;
  RangeDatasetIterator& operator=(const RangeDatasetIterator&) = delete;

  AsyncValueRef<std::tuple<T>> GetNext(
      const ExecutionContext& exec_ctx) override {
    auto* host = IteratorBase::host_;
    bool has_next = (dataset_->step_ > 0 && next_ < dataset_->stop_) ||
                    (dataset_->step_ < 0 && next_ > dataset_->stop_);
    if (!has_next) {
      return AsyncValueRef<std::tuple<T>>();
    }
    auto result = next_;
    next_ += dataset_->step_;
    return host->template MakeConcreteAsyncValueRef<std::tuple<T>>(
        std::make_tuple(result));
  }

 private:
  void Destroy() override {
    internal::DestroyImpl<RangeDatasetIterator>(this, dataset_->allocator_);
  }

  RCReference<RangeDataset<T>> dataset_;
  T next_;
};

template <typename T>
RCReference<Iterator<T>> RangeDataset<T>::MakeIterator() {
  return TakeRef(host_->Construct<RangeDatasetIterator<T>>(FormRef(this)));
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_RANGE_DATASET_H_
