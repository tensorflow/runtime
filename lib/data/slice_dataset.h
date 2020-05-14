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

//===- slice_dataset.h ------------------------------------------*- C++ -*-===//
//
// This file declares SliceDataset class which wraps around a vector of
// elements.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DATA_SLICE_DATASET_H_
#define TFRT_DATA_SLICE_DATASET_H_

#include "dataset.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {
namespace data {

template <typename T>
class SliceDatasetIterator;

// SliceDataset allows caller to access elements in the underlying std::vector.
// TODO(rachelim): Support variadic parameters for SliceDataset.
template <typename T>
class SliceDataset : public Dataset<T> {
 public:
  explicit SliceDataset(std::vector<T> data, HostContext* host)
      : data_(std::move(data)), host_(host), allocator_(host->allocator()) {}

  // This class is not copyable or movable.
  SliceDataset(const SliceDataset&) = delete;
  SliceDataset& operator=(const SliceDataset&) = delete;

  RCReference<Iterator<T>> MakeIterator() override;

 private:
  friend class SliceDatasetIterator<T>;

  void Destroy() override {
    internal::DestroyImpl<SliceDataset<T>>(this, allocator_);
  }

  std::vector<T> data_;
  HostContext* host_;
  HostAllocator* allocator_;
};

template <typename T>
class SliceDatasetIterator : public Iterator<T> {
 public:
  explicit SliceDatasetIterator(RCReference<SliceDataset<T>> parent_dataset,
                                typename std::vector<T>::iterator iterator,
                                typename std::vector<T>::iterator end)
      : Iterator<T>(),
        parent_dataset_(std::move(parent_dataset)),
        iterator_(std::move(iterator)),
        end_(std::move(end)) {}

  IterationResultUntyped GetNextUntyped(
      const ExecutionContext& exec_ctx) override {
    auto* host = exec_ctx.host();
    if (iterator_ == end_) {
      return IterationResultUntyped::Eof(host, 1);
    }

    SmallVector<RCReference<AsyncValue>, 4> values;
    values.push_back(host->MakeAvailableAsyncValueRef<T>(*iterator_));
    iterator_++;
    return IterationResultUntyped::Values(std::move(values), host);
  }

 private:
  // This class is not copyable or movable.
  SliceDatasetIterator(const SliceDatasetIterator&) = delete;
  SliceDatasetIterator& operator=(const SliceDatasetIterator&) = delete;

  void Destroy() override {
    internal::DestroyImpl<SliceDatasetIterator>(this,
                                                parent_dataset_->allocator_);
  }

  RCReference<SliceDataset<T>> parent_dataset_;
  typename std::vector<T>::iterator iterator_;
  typename std::vector<T>::iterator end_;
};

// Add template specialization for DenseHostTensor because DenseHostTensor does
// not have copy constructor. This implementation passes DenseHostTensor by
// reference.
template <>
inline IterationResultUntyped
SliceDatasetIterator<DenseHostTensor>::GetNextUntyped(
    const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  if (iterator_ == end_) {
    return IterationResultUntyped::Eof(host, 1);
  }

  SmallVector<RCReference<AsyncValue>, 4> values;
  values.push_back(
      host->MakeAvailableAsyncValueRef<DenseHostTensor>(iterator_->CopyRef()));
  iterator_++;
  return IterationResultUntyped::Values(std::move(values), host);
}

template <typename T>
RCReference<Iterator<T>> SliceDataset<T>::MakeIterator() {
  return TakeRef(host_->Construct<SliceDatasetIterator<T>>(
      FormRef(this), data_.begin(), data_.end()));
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_DATA_SLICE_DATASET_H_
