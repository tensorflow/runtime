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

//===- contraction_output_kernel.h ------------------------------*- C++ -*-===//
//
// Eigen Tensor contraction output kernel is a mechanism to fuse any
// element-wise operations into the Tensor contraction expression.
//
// All tensor contractions in Eigen are computed as a matrix multiplication. For
// each block of the output matrix, Eigen can call an output kernel before
// "finalizing" the output block. Given that output blocks typically fit into L1
// cache, any simple element-wise operations are practically free.
//
// Depending on the type of the original Tensor contraction expression, it's
// possible to fuse more complicated operations, e.g. derive batch, spatial or
// channel coordinates from the output block offsets.
//
// If you want to do a simple element-wise operation (e.g. apply sqrt to all
// values) you do not need to understand the details described below (no need to
// recover spatial position of the output block in the post-contraction tensor).
//
// IMPORTANT: Eigen TensorContraction swaps lhs with rhs, and changes layout
// from RowMajor (DenseHostTensor layout) to ColMajor (default in Eigen), and
// computes matrix multiplication using these tensors.
//
// (1) Spatial Convolution (see spatial_convolution.h):
//
//   TensorContraction output matrix (before reshape) has a ColMajor layout, and
//   has dimensions:
//    - 0 (rows): output_channels (contiguous in memory)
//    - 1 (cols): all other dimensions
//
//   First element in every column is:
//     [batch ??, height ??, width ??, out_channel = i]
//
//   We do not know the values of 'batch', 'height', and 'width' here (if we
//   know original dimensions, they can be computed from 'j').
//
//   Each column of an output block is a continuous slice along the output
//   channel dimension, so we can use it to efficiently compute any
//   transformation that depends only on a channel value (e.g. add channel
//   bias).
//
// (2) Matrix Multiplication:
//
//   For the `MxK * KxN` matrix multiplication, output matrix has a `MxN`
//   dimensions. Each column in output block is a slice of the innermost
//   dimension of the output matrix starting at offset 'i'.
//
//   Example: For matrix multiplication [8x32] * [32x64], each output block
//   column will correspond to a MatMul output row of size 64 (because
//   DenseHostTensor uses row major storage order).
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_CONTRACTION_OUTPUT_KERNEL_H_
#define TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_CONTRACTION_OUTPUT_KERNEL_H_

#include "tfrt/common/compat/eigen/tensor_types.h"

namespace tfrt {
namespace compat {

namespace internal {

// Type alias for the tensor contraction output mapper.
template <typename T>
using ContractionOutputMapper =
    Eigen::internal::blas_data_mapper<T, Eigen::Index, Eigen::ColMajor>;

}  // namespace internal

// Returns input expression without any transformations.
struct Identity {
  template <typename XprType>
  static auto apply(XprType expr) -> XprType {
    return expr;
  };
};

// Applies `Relu` to the passed input expression.
struct Relu {
  template <typename XprType>
  static auto apply(XprType expr)
      -> decltype(expr.cwiseMax(std::declval<typename XprType::Scalar>())) {
    return expr.cwiseMax(static_cast<typename XprType::Scalar>(0));
  };
};

// Adds bias to the output block inner dimension. Optionally applies activation
// function specified by `Activation` type parameter.
template <typename T, typename Activation = Identity>
class BiasAddOutputKernel {
  using Index = Eigen::Index;
  using Vec = Eigen::Tensor<T, 1, Eigen::RowMajor, Index>;

 public:
  explicit BiasAddOutputKernel(const EigenConstTensor<T, 1>& bias)
      : bias_data_(bias.data()), bias_size_(bias.size()) {}

  EIGEN_ALWAYS_INLINE void operator()(
      const internal::ContractionOutputMapper<T>& output_mapper,
      const Eigen::TensorContractionParams& params, Index i, Index j,
      Index num_rows, Index num_cols) const {
    // There is no guarantee that `bias + offset` will be properly aligned.
    using Bias = Eigen::TensorMap<const Vec, Eigen::Unaligned>;
    using OutputChannels = Eigen::TensorMap<Vec, Eigen::Unaligned>;

    assert(params.swapped_arguments &&
           "Unexpected contraction output kernel parameters");
    assert(i + num_rows <= bias_size_ &&
           "Output block inner dimension is larger than the bias vector");

    const T* bias_base = bias_data_ + i;
    const Bias bias(bias_base, num_rows);

    for (Index col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      OutputChannels output(output_base, num_rows);
      const auto expr = output + bias;
      output = Activation::template apply<decltype(expr)>(expr);
    }
  }

 private:
  const T* bias_data_;
  const Index bias_size_;
};

template <typename T, typename Activation = Identity>
class BatchNormOutputKernel {
  using Index = Eigen::Index;
  using Vec = Eigen::Tensor<T, 1, Eigen::RowMajor, Index>;

 public:
  BatchNormOutputKernel(const EigenConstTensor<T, 1>& scale,
                        const EigenConstTensor<T, 1>& offset,
                        const EigenConstTensor<T, 1>& estimated_mean,
                        const EigenConstTensor<T, 1>& estimated_variance,
                        float epsilon)
      : scale_data_(scale.data()),
        offset_data_(offset.data()),
        estimated_mean_data_(estimated_mean.data()),
        estimated_variance_data_(estimated_variance.data()) {
    scaling_factor_ =
        (estimated_variance + static_cast<T>(epsilon)).rsqrt() * scale;
  }

  EIGEN_ALWAYS_INLINE void operator()(
      const internal::ContractionOutputMapper<T>& output_mapper,
      const Eigen::TensorContractionParams& params, Index i, Index j,
      Index num_rows, Index num_cols) const {
    // There is no guarantee that any of the batch normalization parameters
    // tensors will be aligned at the given offset.
    using ScalingFactor = Eigen::TensorMap<const Vec, Eigen::Unaligned>;
    using Offset = Eigen::TensorMap<const Vec, Eigen::Unaligned>;
    using Mean = Eigen::TensorMap<const Vec, Eigen::Unaligned>;
    using OutputChannels = Eigen::TensorMap<Vec, Eigen::Unaligned>;

    assert(params.swapped_arguments &&
           "Unexpected contraction output kernel parameters");
    assert(i + num_rows <= scaling_factor_.size() &&
           "Output block inner dimension is larger than the scaling factor");

    const T* scaling_factor_base = scaling_factor_.data() + i;
    const T* offset_base = offset_data_ + i;
    const T* mean_base = estimated_mean_data_ + i;

    const ScalingFactor scaling_factor(scaling_factor_base, num_rows);
    const Offset offset(offset_base, num_rows);
    const Mean mean(mean_base, num_rows);

    for (Index col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      OutputChannels output(output_base, num_rows);

      auto scaled = (output - mean) * scaling_factor;
      auto shifted = scaled + offset;

      output = Activation::template apply<decltype(shifted)>(shifted);
    }
  }

 private:
  const T* scale_data_;
  const T* offset_data_;
  const T* estimated_mean_data_;
  const T* estimated_variance_data_;

  // Precomputed expression:
  //   scaling_factor = (estimated_variance + epsilon).rsqrt() * scale
  Eigen::Tensor<T, 1, Eigen::RowMajor> scaling_factor_;
};

}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_CONTRACTION_OUTPUT_KERNEL_H_
