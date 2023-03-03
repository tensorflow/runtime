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

// Collection of helper functions to compute and validate tensor shapes.

#ifndef TFRT_BACKENDS_COMMON_COMPAT_EIGEN_KERNELS_SHAPE_FUNCTIONS_H_
#define TFRT_BACKENDS_COMMON_COMPAT_EIGEN_KERNELS_SHAPE_FUNCTIONS_H_

#include <optional>

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace compat {

enum class PaddingType { kSame, kValid, kExplicit };

struct Padding {
  Index padding_before;
  Index padding_after;
};

struct WindowedOutputDimension {
  Index output_size;  // Computed output dimension.
  Padding padding;    // Computed padding for corresponding input dimension.
};

// Parses padding type from a string view: [same, valid, explicit].
llvm::Expected<PaddingType> ParsePaddingType(string_view padding_type);

// ComputeWindowedOutputDimension() computes output dimension size and padding
// for corresponding input dimension from the input and filter size, stride and
// dilation rate.
//
// If padding type is explicit, `explicit_padding` must have a value, otherwise
// it must be empty.
llvm::Expected<WindowedOutputDimension> ComputeWindowedOutputDimension(
    Index input_size, Index filter_size, Index stride, Index dilation,
    PaddingType padding_type, std::optional<Padding> explicit_padding);

// Returns Error::success() if `lhs_shape` matches `rhs_shape` and error
// otherwise.
llvm::Error CheckShapeMatch(string_view lhs_name, const TensorShape& lhs_shape,
                            string_view rhs_name, const TensorShape& rhs_shape);

template <size_t Rank>
llvm::Error CheckShapeMatch(string_view lhs_name,
                            const FixedRankShape<Rank>& lhs_shape,
                            string_view rhs_name,
                            const FixedRankShape<Rank>& rhs_shape) {
  return CheckShapeMatch(lhs_name, lhs_shape.ToTensorShape(), rhs_name,
                         rhs_shape.ToTensorShape());
}

// Returns Error::success() if `lhs_dim` matches `rhs_dim` and error otherwise.
llvm::Error CheckDimensionMatch(string_view lhs_dim_name, Index lhs_dim,
                                string_view rhs_dim_name, Index rhs_dim);

}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_COMPAT_EIGEN_KERNELS_SHAPE_FUNCTIONS_H_
