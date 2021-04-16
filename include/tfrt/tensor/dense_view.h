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

// This file defines class DenseView and class template DenseTensorView.

#ifndef TFRT_TENSOR_DENSE_VIEW_H_
#define TFRT_TENSOR_DENSE_VIEW_H_

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {

template <typename T, size_t Rank>
class DenseTensorView;

// DenseView is a view to any densely laid out data with a shape.
class DenseView {
 public:
  explicit DenseView(DType dtype, ArrayRef<int64_t> shape, const void* data)
      : metadata_(dtype, shape), data_(data) {}

  const TensorMetadata& metadata() const { return metadata_; }

  DType dtype() const { return metadata_.dtype; }

  const TensorShape& shape() const { return metadata_.shape; }

  const void* data() const { return data_; }

  template <typename T>
  ArrayRef<T> GetFlat() const {
    assert(dtype() == GetDType<T>());
    return llvm::makeArrayRef<T>(static_cast<const T*>(data()),
                                 shape().GetNumElements());
  }

  template <typename T, size_t Rank>
  DenseTensorView<T, Rank> GetTensor() const {
    assert(dtype() == GetDType<T>());
    assert(shape().GetRank() == Rank);
    return DenseTensorView<T, Rank>(this);
  }

 private:
  TensorMetadata metadata_;
  const void* data_;
};

// DenseTensorView is a view to any densely laid out data in a tensor format.
template <typename T, size_t Rank>
class DenseTensorView {
 public:
  DType dtype() const { return GetDType<T>(); }

  const FixedRankShape<Rank>& shape() const { return shape_; }

  ArrayRef<T> GetFlat() const { return data_; }

  const T* data() const { return data_.data(); }

  template <typename... Index>
  const T& GetElementAt(Index... index) const {
    static_assert(sizeof...(Index) == Rank, "Incorrect number of indices");
    return GetFlat()[CalculateOffset(index...)];
  }

 private:
  template <typename... Index>
  size_t CalculateOffset(Index... index) const {
    return CalculateOffsetInternal<0>(index...).first;
  }

  template <size_t Dim>
  std::pair<size_t, size_t> CalculateOffsetInternal() const {
    static_assert(Dim == Rank, "Rank mismatch");
    return {/*offset=*/0, /*stride=*/1};
  }

  template <size_t Dim, typename... Index>
  std::pair<size_t, size_t> CalculateOffsetInternal(
      int64_t index, Index... remaining_indices) const {
    static_assert(Dim < Rank, "Rank mismatch");

    auto pair = CalculateOffsetInternal<Dim + 1>(remaining_indices...);
    auto offset = pair.first;
    auto stride = pair.second;

    auto dim_size = shape()[Dim];
    assert(index < dim_size);
    return {index * stride + offset, dim_size * stride};
  }

  DenseTensorView(const DenseView* dense_view) : shape_(dense_view->shape()) {
    assert(dense_view->dtype() == GetDType<T>());
    assert(dense_view->shape().GetRank() == Rank);
    data_ = llvm::makeArrayRef(static_cast<const T*>(dense_view->data()),
                               shape_.GetNumElements());
  }

  FixedRankShape<Rank> shape_;
  ArrayRef<T> data_;

  friend class DenseView;
};

}  // namespace tfrt

#endif  // TFRT_TENSOR_DENSE_VIEW_H_
