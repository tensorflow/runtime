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

// Batch normalization kernels implemented with Eigen.

#ifndef TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_BATCH_NORM_H_
#define TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_BATCH_NORM_H_

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace compat {

using ::Eigen::Index;
using ::tfrt::compat::AsEigenConstTensor;
using ::tfrt::compat::AsEigenTensor;
using ::tfrt::compat::EigenConstTensor;

template <typename T>
static AsyncValueRef<Chain> FusedBatchNormV3Impl(
    const DenseHostTensor& input, const DenseHostTensor& scale,
    const DenseHostTensor& bias, const DenseHostTensor& mean,
    const DenseHostTensor& variance, DenseHostTensor* output, T epsilon,
    const ExecutionContext& exec_ctx) {
  // shape_input has format (batch_size, height, width, channel_num).
  DHTIndexableView<T, 4> input_view(&input);
  DHTArrayView<T> scale_view(&scale);
  DHTArrayView<T> bias_view(&bias);
  DHTArrayView<T> mean_view(&mean);
  DHTArrayView<T> variance_view(&variance);
  MutableDHTIndexableView<T, 4> output_view(output);

  const auto& shape_input = input_view.FixedShape();
  const auto& shape_output = output_view.FixedShape();

  if (scale.NumElements() != shape_input[3] ||
      bias.NumElements() != shape_input[3] ||
      mean.NumElements() != shape_input[3] ||
      variance.NumElements() != shape_input[3]) {
    return EmitErrorAsync(exec_ctx,
                          "parameter does not match the input channel number");
  }

  if (shape_input != shape_output) {
    return EmitErrorAsync(exec_ctx, "unexpected output shape");
  }

  const Index depth = shape_input[3];
  const Index rest_size = shape_input[0] * shape_input[1] * shape_input[2];

  Eigen::DSizes<Index, 2> rest_by_depth(rest_size, depth);
  Eigen::IndexList<Index, Eigen::type2index<1>> rest_by_one;
  rest_by_one.set(0, rest_size);
  Eigen::IndexList<Eigen::type2index<1>, Index> one_by_depth;
  one_by_depth.set(1, depth);
  Eigen::IndexList<Index, Eigen::type2index<1>> depth_by_one;
  depth_by_one.set(0, depth);

  // Reshape and broadcast vectors of [depth] to [rest_size, depth] tensor.
  auto to_rest_by_depth = [&](const auto& vec) -> auto {
    return vec.reshape(one_by_depth).broadcast(rest_by_one);
  };

  auto input_t = AsEigenConstTensor(input_view);
  auto scale_t = AsEigenConstTensor(scale_view);
  auto bias_t = AsEigenConstTensor(bias_view);
  auto mean_t = AsEigenConstTensor(mean_view);
  auto variance_t = AsEigenConstTensor(variance_view);

  auto expr_0 = (input_t.reshape(rest_by_depth) - to_rest_by_depth(mean_t)) /
                to_rest_by_depth((variance_t + epsilon).sqrt().eval());
  auto expr = to_rest_by_depth(scale_t) * expr_0 + to_rest_by_depth(bias_t);

  auto output_t_0 = AsEigenTensor(output_view);
  auto output_t = output_t_0.reshape(rest_by_depth);

  return AsyncAssign(
      exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>(),
      std::move(output_t), std::move(expr),
      KeepBuffers::alive(&input, &scale, &bias, &mean, &variance, output));
}

}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_BATCH_NORM_H_
