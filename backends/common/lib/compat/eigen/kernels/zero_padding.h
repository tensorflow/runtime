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

// Pad input tensor with zeroes.

#ifndef TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_ZERO_PADDING_H_
#define TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_ZERO_PADDING_H_

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {
namespace compat {

using ::Eigen::Index;
using ::tfrt::compat::AsEigenConstTensor;
using ::tfrt::compat::AsEigenTensor;

template <typename T>
static AsyncValueRef<Chain> TfPadImpl(
    const DenseHostTensor& input, const int32_t height_1,
    const int32_t height_2, const int32_t width_1, const int32_t width_2,
    DenseHostTensor* output, const ExecutionContext& exec_ctx) {
  // input_shape has format (batch_size, height, width, in_channel_num).
  DHTIndexableView<T, 4> input_view(&input);
  MutableDHTIndexableView<T, 4> output_view(output);
  const auto& input_shape = input_view.FixedShape();
  const auto& output_shape = output_view.FixedShape();

  // output_shape has format (batch_size, height, width, out_channel_num).
  const FixedRankShape<4> expected_output_shape(
      {input_shape[0], input_shape[1] + height_1 + height_2,
       input_shape[2] + width_1 + width_2, input_shape[3]});

  if (output_shape != expected_output_shape) {
    return EmitErrorAsync(exec_ctx,
                          StrCat("ZeroPadding output shape ", output_shape,
                                 " does not match the expected output shape ",
                                 expected_output_shape));
  }
  const std::pair<Index, Index> padding_heights = {height_1, height_2};
  const std::pair<Index, Index> padding_widths = {width_1, width_2};
  const Eigen::array<std::pair<Index, Index>, 4> paddings = {
      {0, 0}, padding_heights, padding_widths, {0, 0}};

  auto input_t = AsEigenConstTensor(input_view);
  auto output_t = AsEigenTensor(output_view);
  auto expr = input_t.pad(paddings);

  return AsyncAssign(
      exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>(),
      std::move(output_t), expr, KeepBuffers::alive(&input, output));
}

}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_ZERO_PADDING_H_
