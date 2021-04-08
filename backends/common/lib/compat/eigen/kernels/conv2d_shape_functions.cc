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

// Collection of helper shape functions for Conv2D kernels.

#include "conv2d_shape_functions.h"

#include <array>

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace compat {

// Computes convolution parameters from the padding type and input/kernel
// shapes. Returns error if input shapes does not match expectations.
llvm::Expected<Conv2DParams> ComputeConv2DParams(
    const FixedRankShape<4>& input_shape, const FixedRankShape<4>& kernel_shape,
    string_view padding, std::array<ssize_t, 2> strides) {
  // Padding must be a valid string.
  auto padding_type = ParsePaddingType(padding);
  if (!padding_type) return padding_type.takeError();

  // Input channels dimension size must match kernel dimension.
  auto channels_error = CheckDimensionMatch("input channels", input_shape[3],
                                            "kernel depth", kernel_shape[2]);
  if (channels_error) return std::move(channels_error);

  // TODO(ezhulenev): Add support for dilations.
  std::array<ssize_t, 2> dilations = {1, 1};

  auto output_height = ComputeWindowedOutputDimension(
      input_shape[1], kernel_shape[0], strides[0], dilations[0], *padding_type,
      /*explicit_padding=*/llvm::None);
  if (!output_height) return output_height.takeError();

  auto output_width = ComputeWindowedOutputDimension(
      input_shape[2], kernel_shape[1], strides[1], dilations[1], *padding_type,
      /*explicit_padding=*/llvm::None);
  if (!output_width) return output_width.takeError();

  // Expected output shape.
  const FixedRankShape<4> output_shape({input_shape[0],              // batch
                                        output_height->output_size,  // height
                                        output_width->output_size,   // width
                                        kernel_shape[3]});           // channels

  // Computed input paddings.
  const std::array<ssize_t, 4> paddings = {
      output_height->padding.padding_before,  // padding top
      output_height->padding.padding_after,   // padding bottom
      output_width->padding.padding_before,   // padding left
      output_width->padding.padding_after};   // padding right

  Conv2DParams params{*padding_type, paddings,     strides,     dilations,
                      input_shape,   kernel_shape, output_shape};

  return params;
}

}  // namespace compat
}  // namespace tfrt
