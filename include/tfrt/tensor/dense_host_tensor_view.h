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

// This file defines the DHTIndexableView class template.

#ifndef TFRT_TENSOR_DENSE_HOST_TENSOR_VIEW_H_
#define TFRT_TENSOR_DENSE_HOST_TENSOR_VIEW_H_

#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/host_context/host_buffer.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {

class HostContext;

// DHTArrayView<DType> projects a DenseHostTensor as an array of type
// DType. This is useful for element-wise tensor operations, such as tensor add.
//
// The underlying DenseHostTensor must out live DHTArrayView. The underlying
// DenseHostTensor cannot be changed via DHTArrayView. Concurrent changes to the
// underlying tensor data via DenseHostTensor itself or through another view
// class are visible through DHTArrayView.
template <typename DType>
class DHTArrayView {
 public:
  // Used by Argument<> to get the underlying type.
  using UnderlyingT = DenseHostTensor;

  // Used by ::testing::ElementsAre to get the underlying type.
  using value_type = DType;

  DHTArrayView() = default;

  /*implicit*/ DHTArrayView(const DenseHostTensor* dht)
      : data_(dht->data<DType>()), num_elements_(dht->NumElements()) {}

  DHTArrayView(const DType* data, size_t num_elements)
      : data_(data), num_elements_(num_elements) {}

  // Raw access to data. Typically used when dispatching to external libraries
  // (like Eigen or libxssm).
  size_t NumElements() const { return num_elements_; }

  // The pointer to the data. If there is no elements, the returned pointer is
  // undefined.
  const DType* data() const { return data_; }

  ArrayRef<DType> Elements() const {
    return ArrayRef<DType>(data(), NumElements());
  }

  using const_iterator = typename ArrayRef<DType>::iterator;
  const_iterator begin() const { return Elements().begin(); }
  const_iterator end() const { return Elements().end(); }

  const DType& operator[](size_t idx) const { return data()[idx]; }

 protected:
  const DType* data_ = nullptr;
  size_t num_elements_ = 0;
};

template <typename DType>
raw_ostream& operator<<(raw_ostream& os, const DHTArrayView<DType>& t);

// Mutable version of DHTArrayView. The underlying DenseHostTensor may be
// changed via this view class. Concurrent changes to the
// underlying tensor data via DenseHostTensor itself or through another view
// class are visible through MutableDHTArrayView.
template <typename DType>
class MutableDHTArrayView : public DHTArrayView<DType> {
 public:
  MutableDHTArrayView() = default;

  /*implicit*/ MutableDHTArrayView(DenseHostTensor* dht)
      : DHTArrayView<DType>(dht) {}

  MutableDHTArrayView(DType* data, size_t num_elements)
      : DHTArrayView<DType>(data, num_elements) {}

  // Sets all values to 'v'. Useful for operations that have some obvious
  // initializer (usually 0 or 1).
  void Fill(const DType& v) {
    std::fill(data(), data() + this->NumElements(), v);
  }

  // The pointer to the data. If there is no elements, the returned pointer is
  // undefined.
  DType* data() const {
    return const_cast<DType*>(this->DHTArrayView<DType>::data());
  }

  MutableArrayRef<DType> Elements() const {
    return MutableArrayRef<DType>(data(), this->NumElements());
  }

  // Allow direct iteration and manipulation of the view as if it were its
  // elements.
  using iterator = typename MutableArrayRef<DType>::iterator;
  iterator begin() const { return Elements().begin(); }
  iterator end() const { return Elements().end(); }

  using DHTArrayView<DType>::operator[];
  DType& operator[](size_t idx) const { return data()[idx]; }
};

template <size_t Rank, typename... Dims>
std::array<ssize_t, Rank> CoordFromDims(Dims... dims) {
  static_assert(sizeof...(Dims) == Rank,
                "invalid number of values in coordinate.");
  return std::array<ssize_t, Rank>{static_cast<ssize_t>(dims)...};
}

// Returns the offset of the given coordinate in the underlying storage. If the
// coordinates are of smaller rank than the shape, the coordinates are used as a
// prefix and the missing trailing dimensions are filled with zeros.
template <size_t ShapeRank, size_t CoordRank>
size_t OffsetOf(const FixedRankShape<ShapeRank>& fixed_shape,
                const std::array<ssize_t, CoordRank>& coord) {
  static_assert(CoordRank <= ShapeRank,
                "coordinates must be within shape rank");
  size_t offset = 0;
  size_t stride = 1;
  for (int i = ShapeRank - 1; i >= 0; --i) {
    if (i < CoordRank) {
      assert(coord[i] < fixed_shape[i]);
      offset += stride * coord[i];
    }
    stride *= fixed_shape[i];
  }
  return offset;
}

// DHTIndexableView<DType, Rank> projects a DenseHostTensor into a view that
// allows efficient access of tensor elements by their coordinates. Compared to
// DHTArrayView<T>, DHTIndexableView<T, Rank> maintains an internal
// FixedRankShape to enable efficient element indexing.
//
// Similar to DHTArrayView<T>, the underlying DenseHostTensor must out live
// DHTIndexableView. The underlying DenseHostTensor can not be changed via
// DHTIndexableView. Concurrent changes to the underlying tensor data via
// DenseHostTensor itself or through another view class are visible through
// DHTIndexableView.
template <typename DType, size_t Rank>
class DHTIndexableView {
 public:
  // Used by Argument<> to get the underlying type.
  using UnderlyingT = DenseHostTensor;

  // Used by ::testing::ElementsAre to get the underlying type.
  using value_type = DType;

  using FixedShapeType = FixedRankShape<Rank>;
  using CoordType = std::array<ssize_t, Rank>;

  DHTIndexableView() = default;

  /*implicit*/ DHTIndexableView(const DenseHostTensor* dht)
      : data_(dht->data<DType>()),
        num_elements_(dht->NumElements()),
        fixed_shape_(dht->shape()) {}

  DHTIndexableView(const DType* data, const FixedShapeType& shape)
      : data_(data),
        num_elements_(shape.GetNumElements()),
        fixed_shape_(shape) {}

  template <typename... Dims>
  explicit DHTIndexableView(const DType* data, Dims... dims)
      : DHTIndexableView(
            data, FixedShapeType(CoordFromDims<Rank, Dims...>(dims...))) {}

  // The fixed shape of this tensor.
  const FixedShapeType& FixedShape() const { return fixed_shape_; }

  // Raw access to data. Typically used when dispatching to external libraries
  // (like Eigen or libxssm).
  size_t NumElements() const { return num_elements_; }

  // The pointer to the data. If there is no elements, the returned pointer is
  // undefined.
  const DType* data() const { return data_; }

  ArrayRef<DType> Elements() const {
    return ArrayRef<DType>(data(), NumElements());
  }

  using const_iterator = typename ArrayRef<DType>::iterator;
  const_iterator begin() const { return Elements().begin(); }
  const_iterator end() const { return Elements().end(); }

  // Returns reference to element at the given coordinate.
  const DType& operator[](const CoordType& coord) const {
    return this->data()[OffsetOf(fixed_shape_, coord)];
  }

  // Convenience wrapper around operator[]. Specify the Coord as a list of index
  // arguments rather than a 'Coord'. Makes writing nested for loops much
  // easier.
  template <typename... Dims>
  const DType& ElementAt(Dims... dims) const {
    return (*this)[CoordFromDims<Rank, Dims...>(dims...)];
  }

  // Chip is a special kind of slice. It indexes into the view at the given
  // coordinate prefix and returns a view onto the remaining dimensions.
  // It is similar to indexing into a numpy array, e.g. for a 5D ndarray A, the
  // slice A[1, 3] would return a 3D view.
  template <typename... PrefixDims, size_t PrefixRank = sizeof...(PrefixDims),
            size_t ChippedRank = Rank - PrefixRank>
  DHTIndexableView<DType, ChippedRank> Chip(PrefixDims... dims) const {
    static_assert(PrefixRank > 0, "prefix dimensions cannot be empty");
    static_assert(PrefixRank <= Rank, "prefix dimensions must be within rank");
    auto coord = CoordFromDims<PrefixRank, PrefixDims...>(dims...);
    FixedRankShape<ChippedRank> chipped_shape;
    for (int i = 0; i < ChippedRank; i++) {
      chipped_shape[i] = fixed_shape_[i + PrefixRank];
    }
    return DHTIndexableView<DType, Rank - PrefixRank>(
        &this->data()[OffsetOf(fixed_shape_, coord)], chipped_shape);
  }

 private:
  const DType* data_ = nullptr;
  size_t num_elements_ = 0;
  FixedShapeType fixed_shape_;
};

// Mutable version of DHTIndexableView. The underlying DenseHostTensor may be
// changed via this view class. Concurrent changes to the underlying tensor data
// via DenseHostTensor itself or through another view class are visible through
// MutableDHTIndexableView.
template <typename DType, size_t Rank>
class MutableDHTIndexableView : public DHTIndexableView<DType, Rank> {
 public:
  using FixedShapeType = FixedRankShape<Rank>;
  using CoordType = std::array<ssize_t, Rank>;

  MutableDHTIndexableView() = default;

  /*implicit*/ MutableDHTIndexableView(DenseHostTensor* dht)
      : DHTIndexableView<DType, Rank>(dht) {}

  MutableDHTIndexableView(DType* data, const FixedShapeType& shape)
      : DHTIndexableView<DType, Rank>(data, shape) {}

  template <typename... Dims>
  explicit MutableDHTIndexableView(DType* data, Dims... dims)
      : DHTIndexableView<DType, Rank>(data, dims...) {}

  // The pointer to the data. If there is no elements, the returned pointer is
  // undefined.
  DType* data() const {
    return const_cast<DType*>(DHTIndexableView<DType, Rank>::data());
  }

  MutableArrayRef<DType> Elements() const {
    return MutableArrayRef<DType>(data(), this->NumElements());
  }

  using iterator = typename MutableArrayRef<DType>::iterator;
  iterator begin() const { return Elements().begin(); }
  iterator end() const { return Elements().end(); }

  // Returns reference to element at the given coordinate.
  DType& operator[](const CoordType& coord) const {
    return this->data()[OffsetOf(this->FixedShape(), coord)];
  }

  // Convenience wrapper around operator[]. Specify the Coord as a list of index
  // arguments rather than a 'Coord'. Makes writing nested for loops much
  // easier.
  template <typename... Dims>
  DType& ElementAt(Dims... dims) const {
    return (*this)[CoordFromDims<Rank, Dims...>(dims...)];
  }

  // Chip is a special kind of slice. It indexes into the view at the given
  // coordinate prefix and returns a view onto the remaining dimensions.
  // It is similar to indexing into a numpy array, e.g. for a 5D ndarray A, the
  // slice A[1, 3] would return a 3D view.
  template <typename... PrefixDims, size_t PrefixRank = sizeof...(PrefixDims),
            size_t ChippedRank = Rank - PrefixRank>
  MutableDHTIndexableView<DType, ChippedRank> Chip(PrefixDims... dims) const {
    static_assert(PrefixRank > 0, "prefix dimensions cannot be empty");
    static_assert(PrefixRank <= Rank, "prefix dimensions must be within rank");
    auto coord = CoordFromDims<PrefixRank, PrefixDims...>(dims...);
    FixedRankShape<ChippedRank> chipped_shape;
    for (int i = 0; i < ChippedRank; i++) {
      chipped_shape[i] = this->FixedShape()[i + PrefixRank];
    }
    return MutableDHTIndexableView<DType, Rank - PrefixRank>(
        &this->data()[OffsetOf(this->FixedShape(), coord)], chipped_shape);
  }
};

}  // namespace tfrt

#endif  // TFRT_TENSOR_DENSE_HOST_TENSOR_VIEW_H_
