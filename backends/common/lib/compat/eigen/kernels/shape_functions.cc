// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Collection of helper functions to compute and validate tensor shapes.

#include "tfrt/common/compat/eigen/kernels/shape_functions.h"

#include <cassert>
#include <optional>

#include "tfrt/support/string_util.h"

namespace tfrt {
namespace compat {

// Parses padding type from a string view: [same, valid, explicit].
llvm::Expected<PaddingType> ParsePaddingType(string_view padding_type) {
  if (padding_type.lower() == "same") {
    return PaddingType::kSame;
  } else if (padding_type.lower() == "valid") {
    return PaddingType::kValid;
  } else if (padding_type.lower() == "explicit") {
    return PaddingType::kExplicit;
  }

  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "Unknown padding type %s",
                                 padding_type.data());
}

llvm::Expected<WindowedOutputDimension> ComputeWindowedOutputDimension(
    Index input_size, Index filter_size, Index stride, Index dilation,
    PaddingType padding_type, std::optional<Padding> explicit_padding) {
  if (stride <= 0) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Stride must be > 0, but got %d", stride);
  }
  if (dilation <= 0) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Dilation must be > 0, but got %d",
                                   dilation);
  }

  if (padding_type == PaddingType::kExplicit && !explicit_padding.has_value()) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "For explicit padding, padding values must be defined");
  }
  if (padding_type != PaddingType::kExplicit && explicit_padding.has_value()) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Explicit padding values must be empty if "
                                   "padding type is not explicit");
  }

  // Compute effective filter size taking into account dilation rate.
  const Index effective_filter_size = (filter_size - 1) * dilation + 1;

  WindowedOutputDimension output_dimensions{0, {0, 0}};

  switch (padding_type) {
    case PaddingType::kExplicit:
      output_dimensions.output_size =
          (input_size + explicit_padding->padding_before +
           explicit_padding->padding_after - effective_filter_size + stride) /
          stride;
      output_dimensions.padding = explicit_padding.value();
      break;

    case PaddingType::kValid:
      output_dimensions.output_size =
          (input_size - effective_filter_size + stride) / stride;
      output_dimensions.padding = {0, 0};
      break;

    case PaddingType::kSame:
      output_dimensions.output_size = (input_size + stride - 1) / stride;

      const Index padding_needed =
          std::max(Index{0}, (output_dimensions.output_size - 1) * stride +
                                 effective_filter_size - input_size);

      // For odd values of total padding, add more padding at the 'right'
      // side of the given dimension.
      output_dimensions.padding = {padding_needed / 2,
                                   padding_needed - padding_needed / 2};
      break;
  }

  if (output_dimensions.output_size < 0) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "Computed output size would be negative: %d (input_size: %d, "
        "filter_size: %d, stride: %d, dilation: %d)",
        output_dimensions.output_size, input_size, filter_size, stride,
        dilation);
  }

  return output_dimensions;
}

llvm::Error CheckShapeMatch(string_view lhs_name, const TensorShape& lhs_shape,
                            string_view rhs_name,
                            const TensorShape& rhs_shape) {
  if (lhs_shape != rhs_shape) {
    llvm::Twine lhs = lhs_name + " is " + StrCat(lhs_shape);
    llvm::Twine rhs = rhs_name + " is " + StrCat(rhs_shape);

    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "Tensor shapes do not match: " + lhs + " and " + rhs);
  }

  return llvm::Error::success();
}

llvm::Error CheckDimensionMatch(string_view lhs_dim_name, Index lhs_dim,
                                string_view rhs_dim_name, Index rhs_dim) {
  assert(lhs_dim >= 0 && rhs_dim >= 0);

  if (lhs_dim != rhs_dim) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "Dimensions does not match: %s is %d and %s is %d", lhs_dim_name.data(),
        lhs_dim, rhs_dim_name.data(), rhs_dim);
  }

  return llvm::Error::success();
}

}  // namespace compat
}  // namespace tfrt
