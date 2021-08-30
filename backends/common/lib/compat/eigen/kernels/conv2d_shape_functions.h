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

// Collection of helper shape functions for Conv2D kernels.

#ifndef TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_CONV2D_SHAPE_FUNCTIONS_H_
#define TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_CONV2D_SHAPE_FUNCTIONS_H_

#include <array>

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/common/compat/eigen/kernels/shape_functions.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace compat {

struct Conv2DParams {
  PaddingType padding_type;

  std::array<Index, 4> paddings;   // input paddings
  std::array<Index, 2> strides;    // input strides
  std::array<Index, 2> dilations;  // kernel dilation rates

  FixedRankShape<4> input_shape;
  FixedRankShape<4> kernel_shape;
  FixedRankShape<4> output_shape;
};

// Computes convolution parameters from the padding type and input/kernel
// shapes. Returns error if input shapes does not match expectations.
llvm::Expected<Conv2DParams> ComputeConv2DParams(
    const FixedRankShape<4>& input_shape, const FixedRankShape<4>& kernel_shape,
    string_view padding, std::array<Index, 2> strides);

}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_CONV2D_SHAPE_FUNCTIONS_H_
