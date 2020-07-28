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

//===- conv2d_grad_filter.cc -------------------------------------*- C++-*-===//
//
// Conv2D filter gradient.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include "conv2d_shape_functions.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/kernels/shape_functions.h"
#include "tfrt/common/compat/eigen/spatial_convolution.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace compat {

namespace {

// Spatial convolution parameters for filter gradient computation.
struct Conv2DGradFilterParams {
  std::array<ssize_t, 4> paddings;
  std::array<ssize_t, 2> strides;
  std::array<ssize_t, 2> dilations;
};

Conv2DGradFilterParams ComputeConv2DGradFilterParams(
    const Conv2DParams& params) {
  // Convolution input height and width accounting for padding.
  const ssize_t conv_input_height =
      params.input_shape[1] + params.paddings[0] + params.paddings[1];
  const ssize_t conv_input_width =
      params.input_shape[2] + params.paddings[2] + params.paddings[3];

  // Calculate effective input dimension sizes (elements that contributed to the
  // output), we might have to add negative bottom/right padding to exclude some
  // of the input values that did not contribute to the output.
  const ssize_t eff_input_height =
      params.strides[0] * (params.output_shape[1] - 1) + params.kernel_shape[0];
  const ssize_t eff_input_width =
      params.strides[1] * (params.output_shape[2] - 1) + params.kernel_shape[1];

  // Adjust paddings to remove excess from the input.
  std::array<ssize_t, 4> eff_paddings = params.paddings;
  eff_paddings[1] -= (conv_input_height - eff_input_height);
  eff_paddings[3] -= (conv_input_width - eff_input_width);

  // IMPORTANT: Strides become dilations and vice versa.
  return {eff_paddings, /*strides=*/params.dilations,
          /*dilations=*/params.strides};
}

}  // namespace

template <typename T>
static void Conv2DGradFilter(ArgumentView<DHTIndexableView<T, 4>> output_grad,
                             ArgumentView<DHTIndexableView<T, 4>> input,
                             ArgumentView<DHTIndexableView<T, 4>> kernel_grad,
                             Argument<Chain> chain_in, Result<Chain> chain_out,
                             StringAttribute padding,
                             ArrayAttribute<ssize_t> strides,
                             KernelErrorHandler handler,
                             const ExecutionContext& exec_ctx,
                             AsyncKernelFrame* frame) {
  const FixedRankShape<4> input_shape = input->FixedShape();
  const FixedRankShape<4> kernel_shape = kernel_grad->FixedShape();
  const FixedRankShape<4> output_shape = output_grad->FixedShape();

  // Compute convolution parameters from input and kernel shapes.
  auto params = ComputeConv2DParams(input_shape, kernel_shape, padding.get(),
                                    {strides[0], strides[1]});

  // Check that computed output shape matches gradient shape.
  TFRT_RETURN_IF_ERROR(handler, params.takeError());
  TFRT_RETURN_IF_ERROR(
      handler, CheckShapeMatch("output gradient shape", output_shape,
                               "computed output shape", params->output_shape));

  const auto shuffle = [](const FixedRankShape<4>& shape, auto shfl) {
    return FixedRankShape<4>(
        {shape[shfl[0]], shape[shfl[1]], shape[shfl[2]], shape[shfl[3]]});
  };

  // Input: [N, IH, IW, IC].
  auto input_t = AsEigenConstTensor(input.get());

  // Output grad: [N, OH, OW, OC].
  auto output_grad_t = AsEigenConstTensor(output_grad.get());

  // Filter gradient: [FH, FW, IC, OC].
  auto filter_grad_t = AsEigenConstTensor(kernel_grad.get());

  // Input shuffled: [IC, IH, IW, N].
  std::array<ssize_t, 4> input_shuffle = {3, 1, 2, 0};
  auto input_shuffled = input_t.shuffle(input_shuffle).eval();

  // Output grad shuffled: [OH, OW, N, OC].
  std::array<ssize_t, 4> output_grad_shuffle = {1, 2, 0, 3};
  auto output_grad_shuffled = output_grad_t.shuffle(output_grad_shuffle).eval();

  // Convolution: [IC, IH, IW, N] X [OH, OW, N, OC] -> [IC, FH, FW, OC].
  auto grad_params = ComputeConv2DGradFilterParams(*params);
  // clang-format off
  auto convolution = SpatialConvolution(
      input_shuffled, shuffle(input_shape, input_shuffle),
      output_grad_shuffled, shuffle(output_shape, output_grad_shuffle),
      /*strides=*/grad_params.strides,
      /*paddings=*/grad_params.paddings,
      /*dilations=*/grad_params.dilations);
  // clang-format on

  // Convolution shuffled: [FH, FW, IC, OC] (filter gradient).
  Eigen::array<ssize_t, 4> convolution_shuffle = {1, 2, 0, 3};
  auto convolution_shuffled = convolution.shuffle(convolution_shuffle);

  auto on_done = [chain = chain_out.Allocate(),
                  frame = RAIIKernelFrame(*frame)]() { chain.emplace(); };

  AsyncAssign(exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>(),
              std::move(filter_grad_t), std::move(convolution_shuffled),
              std::move(on_done));
}

}  // namespace compat

void RegisterConv2DGradFilterKernels(KernelRegistry* registry) {
  registry->AddKernel("eigen.conv2d.grad.filter.f32",
                      TFRT_KERNEL(compat::Conv2DGradFilter<float>));
}
}  // namespace tfrt
