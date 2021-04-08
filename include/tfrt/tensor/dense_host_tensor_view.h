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

  /*implicit*/ DHTArrayView(const DenseHostTensor* dht) : dht_(*dht) {
    assert(GetDType<DType>() == dht->dtype() && "Incorrect dtype for tensor");
  }

  // The shape of this tensor.
  const TensorShape& Shape() const { return dht_.shape(); }

  // Raw access to data. Typically used when dispatching to external libraries
  // (like Eigen or libxssm).
  size_t NumElements() const { return dht_.DataSizeInBytes() / sizeof(DType); }

  // The pointer to the data. If there is no elements, the returned pointer is
  // undefined.
  const DType* data() const { return static_cast<const DType*>(dht_.data()); }

  ArrayRef<DType> Elements() const {
    return ArrayRef<DType>(data(), NumElements());
  }

  using const_iterator = typename ArrayRef<DType>::iterator;
  const_iterator begin() const { return Elements().begin(); }
  const_iterator end() const { return Elements().end(); }

  const DType& operator[](size_t idx) const { return data()[idx]; }

 protected:
  const DenseHostTensor& dht_;
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
  /*implicit*/ MutableDHTArrayView(DenseHostTensor* dht)
      : DHTArrayView<DType>(dht) {}

  // Sets all values to 'v'. Useful for operations that have some obvious
  // initializer (usually 0 or 1).
  void Fill(const DType& v) {
    std::fill(data(), data() + this->NumElements(), v);
  }
  using DHTArrayView<DType>::data;

  // The pointer to the data. If there is no elements, the returned pointer is
  // undefined.
  DType* data() {
    return const_cast<DType*>(static_cast<const DType*>(this->dht_.data()));
  }

  using DHTArrayView<DType>::Elements;
  MutableArrayRef<DType> Elements() {
    return MutableArrayRef<DType>(data(), this->NumElements());
  }

  // Allow direct iteration and manipulation of the view as if it were its
  // elements.
  using iterator = typename MutableArrayRef<DType>::iterator;

  using DHTArrayView<DType>::begin;
  using DHTArrayView<DType>::end;
  iterator begin() { return Elements().begin(); }
  iterator end() { return Elements().end(); }

  using DHTArrayView<DType>::operator[];
  DType& operator[](size_t idx) { return data()[idx]; }
};

template <size_t Rank, typename... Dims>
static std::array<size_t, Rank> CoordFromDims(Dims... dims) {
  static_assert(sizeof...(Dims) == Rank,
                "invalid number of values in coordinate.");
  return std::array<size_t, Rank>{static_cast<size_t>(dims)...};
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
class DHTIndexableView : public DHTArrayView<DType> {
 public:
  using FixedShapeType = FixedRankShape<Rank>;
  using CoordType = std::array<size_t, Rank>;

  /*implicit*/ DHTIndexableView(const DenseHostTensor* dht)
      : DHTArrayView<DType>(dht), fixed_shape_(dht->shape()) {}

  // The fixed shape of this tensor.
  const FixedShapeType& FixedShape() const { return fixed_shape_; }

  // Returns reference to element at the given coordinate.
  const DType& operator[](const CoordType& coord) const {
    return this->data()[OffsetOf(coord)];
  }

  // Convenience wrapper around operator[]. Specify the Coord as a list of index
  // arguments rather than a 'Coord'. Makes writing nested for loops much
  // easier.
  template <typename... Dims>
  const DType& ElementAt(Dims... dims) const {
    return (*this)[CoordFromDims<Rank, Dims...>(dims...)];
  }

 private:
  // Returns the offset of the given coordinate in the underlying storage.
  size_t OffsetOf(const CoordType& coord) const {
    size_t offset = 0;
    size_t stride = 1;
    for (int i = Rank - 1; i >= 0; --i) {
      assert(coord[i] < fixed_shape_[i]);
      offset += stride * coord[i];
      stride *= fixed_shape_[i];
    }
    return offset;
  }

  FixedShapeType fixed_shape_;
};

// Mutable version of DHTIndexableView. The underlying DenseHostTensor may be
// changed via this view class. Concurrent changes to the underlying tensor data
// via DenseHostTensor itself or through another view class are visible through
// MutableDHTIndexableView.
template <typename DType, size_t Rank>
class MutableDHTIndexableView : public MutableDHTArrayView<DType> {
 public:
  using FixedShapeType = FixedRankShape<Rank>;
  using CoordType = std::array<size_t, Rank>;

  /*implicit*/ MutableDHTIndexableView(DenseHostTensor* dht)
      : MutableDHTArrayView<DType>(dht), fixed_shape_(dht->shape()) {}

  // The fixed shape of this tensor.
  const FixedShapeType& FixedShape() const { return fixed_shape_; }

  // Returns reference to element at the given coordinate.
  DType& operator[](const CoordType& coord) {
    return this->data()[OffsetOf(coord)];
  }
  const DType& operator[](const CoordType& coord) const {
    return this->data()[OffsetOf(coord)];
  }

  // Convenience wrapper around operator[]. Specify the Coord as a list of index
  // arguments rather than a 'Coord'. Makes writing nested for loops much
  // easier.
  template <typename... Dims>
  const DType& ElementAt(Dims... dims) const {
    return (*this)[CoordFromDims<Rank, Dims...>(dims...)];
  }

  template <typename... Dims>
  DType& ElementAt(Dims... dims) {
    return (*this)[CoordFromDims<Rank, Dims...>(dims...)];
  }

 private:
  // Returns the offset of the given coordinate in the underlying storage.
  size_t OffsetOf(const CoordType& coord) const {
    size_t offset = 0;
    size_t stride = 1;
    for (int i = Rank - 1; i >= 0; --i) {
      assert(coord[i] < fixed_shape_[i]);
      offset += stride * coord[i];
      stride *= fixed_shape_[i];
    }
    return offset;
  }

  FixedShapeType fixed_shape_;
};

}  // namespace tfrt

#endif  // TFRT_TENSOR_DENSE_HOST_TENSOR_VIEW_H_
