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

// This file implements TFRT HostTensor conversion to an Eigen TensorMap that
// could be used in Eigen expressions.

#ifndef TFRT_BACKENDS_COMMON_COMPAT_EIGEN_TENSOR_TYPES_H_
#define TFRT_BACKENDS_COMMON_COMPAT_EIGEN_TENSOR_TYPES_H_

#define EIGEN_USE_THREADS

#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/dense_view.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace tfrt {
namespace compat {

//===----------------------------------------------------------------------===//
// Typedefs for Eigen tensors.
//===----------------------------------------------------------------------===//

// Ranked tensor of scalar type T.
template <typename T, size_t Rank = 1, typename IndexType = Eigen::Index>
using EigenTensor =
    Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor, IndexType>,
                     Eigen::Aligned>;

template <typename T, size_t Rank = 1, typename IndexType = Eigen::Index>
using EigenConstTensor =
    Eigen::TensorMap<const Eigen::Tensor<T, Rank, Eigen::RowMajor, IndexType>,
                     Eigen::Aligned>;

// Rank-2 tensor (matrix) of scalar type T.
template <typename T, typename IndexType = Eigen::Index>
using EigenMatrix = EigenTensor<T, 2, IndexType>;

template <typename T, typename IndexType = Eigen::Index>
using EigenConstMatrix = EigenConstTensor<T, 2, IndexType>;

//===----------------------------------------------------------------------===//
// Conversion functions from TFRT tensors to Eigen tensors.
//===----------------------------------------------------------------------===//

template <size_t Rank, typename IndexType = Eigen::Index>
Eigen::DSizes<IndexType, Rank> AsEigenDSizes(
    const FixedRankShape<Rank>& shape) {
  Eigen::DSizes<IndexType, Rank> dims;
  if (Rank > 0) {
    std::copy(shape.begin(), shape.end(), &dims.front());
  }
  return dims;
}

template <size_t Rank, typename IndexType = Eigen::Index>
Eigen::DSizes<IndexType, Rank> AsEigenDSizes(const TensorShape& shape) {
  Eigen::DSizes<IndexType, Rank> dims;
  if (Rank == 1) {
    dims[0] = shape.GetNumElements();
  } else if (Rank > 0) {
    shape.GetDimensions(llvm::makeMutableArrayRef(&dims.front(), Rank));
  }
  return dims;
}

template <int Rank>
FixedRankShape<Rank> AsShape(const Eigen::DSizes<Eigen::Index, Rank> dsizes) {
  FixedRankShape<Rank> shape;
  for (int dim = 0; dim < Rank; ++dim) {
    shape[dim] = dsizes[dim];
  }
  return shape;
}

template <typename T, size_t Rank>
EigenTensor<T, Rank> AsEigenTensor(MutableDHTIndexableView<T, Rank> tensor) {
  return EigenTensor<T, Rank>(tensor.data(),
                              AsEigenDSizes(tensor.FixedShape()));
}

template <typename T, size_t Rank, size_t TargetRank>
EigenTensor<T, TargetRank> AsEigenTensor(
    MutableDHTIndexableView<T, Rank> tensor,
    const FixedRankShape<TargetRank>& target_shape) {
  assert(tensor.FixedShape().GetNumElements() == target_shape.GetNumElements());
  return EigenTensor<T, TargetRank>(tensor.data(), AsEigenDSizes(target_shape));
}

template <typename T, size_t Rank>
EigenConstTensor<T, Rank> AsEigenConstTensor(
    const DenseTensorView<T, Rank>& tensor) {
  return EigenConstTensor<T, Rank>(tensor.data(),
                                   AsEigenDSizes(tensor.shape()));
}

template <typename T, size_t Rank>
EigenConstTensor<T, Rank> AsEigenConstTensor(DHTIndexableView<T, Rank> tensor) {
  return EigenConstTensor<T, Rank>(tensor.data(),
                                   AsEigenDSizes(tensor.FixedShape()));
}

template <typename T, size_t Rank, size_t TargetRank>
EigenConstTensor<T, TargetRank> AsEigenConstTensor(
    DHTIndexableView<T, Rank> tensor,
    const FixedRankShape<TargetRank>& target_shape) {
  assert(tensor.FixedShape().GetNumElements() == target_shape.GetNumElements());
  return EigenConstTensor<T, TargetRank>(tensor.data(),
                                         AsEigenDSizes(target_shape));
}

template <typename T>
EigenTensor<T, 1> AsEigenTensor(MutableDHTArrayView<T> tensor) {
  return EigenTensor<T, 1>(tensor.data(), tensor.NumElements());
}

template <typename T, size_t TargetRank>
EigenTensor<T, TargetRank> AsEigenTensor(
    MutableDHTArrayView<T> tensor,
    const FixedRankShape<TargetRank>& target_shape) {
  assert(tensor.NumElements() == target_shape.GetNumElements());
  return EigenTensor<T, TargetRank>(tensor.data(), AsEigenDSizes(target_shape));
}

template <typename T>
EigenConstTensor<T, 1> AsEigenConstTensor(DHTArrayView<T> tensor) {
  return EigenConstTensor<T, 1>(tensor.data(), tensor.NumElements());
}

template <typename T, size_t TargetRank>
EigenConstTensor<T, TargetRank> AsEigenConstTensor(
    DHTArrayView<T> tensor, const FixedRankShape<TargetRank>& target_shape) {
  assert(tensor.NumElements() == target_shape.GetNumElements());
  return EigenConstTensor<T, TargetRank>(tensor.data(),
                                         AsEigenDSizes(target_shape));
}

}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_COMPAT_EIGEN_TENSOR_TYPES_H_
