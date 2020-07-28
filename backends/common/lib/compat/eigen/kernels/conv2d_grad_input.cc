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

//===- conv2d_grad_input.cc --------------------------------------*- C++-*-===//
//
// Conv2D input gradient.
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

namespace tfrt {
namespace compat {

namespace {

// Spatial convolution parameters for filter gradient computation.
struct Conv2DGradInputParams {
  std::array<ssize_t, 4> paddings;
  std::array<ssize_t, 2> strides;
  std::array<ssize_t, 2> dilations;
  std::array<ssize_t, 2> inflations;
};

Conv2DGradInputParams ComputeConv2DGradInputParams(const Conv2DParams& params) {
  // Compute effective filter size taking into account dilation rate.
  const std::array<ssize_t, 2> eff_filter_size = {
      (params.kernel_shape[0] - 1) * params.dilations[0] + 1,
      (params.kernel_shape[1] - 1) * params.dilations[1] + 1};

  // Compute paddings for the output gradient tensor.
  const ssize_t pad_top = eff_filter_size[0] - 1 - params.paddings[0];
  const ssize_t pad_bottom = params.input_shape[1] -
                             (params.output_shape[1] - 1) * params.strides[0] -
                             2 - pad_top + eff_filter_size[0];

  const ssize_t pad_left = eff_filter_size[1] - 1 - params.paddings[2];
  const ssize_t pad_right = params.input_shape[2] -
                            (params.output_shape[2] - 1) * params.strides[1] -
                            2 - pad_left + eff_filter_size[1];

  assert(pad_top >= 0 && pad_bottom >= 0);
  assert(pad_left >= 0 && pad_right >= 0);

  return {/*paddings=*/{pad_top, pad_bottom, pad_left, pad_right},
          /*strides=*/{1, 1},
          /*dilations=*/params.dilations,
          /*inflations=*/params.strides};
}

}  // namespace

template <typename T>
static void Conv2DGradInput(ArgumentView<DHTIndexableView<T, 4>> output_grad,
                            ArgumentView<DHTIndexableView<T, 4>> kernel,
                            ArgumentView<DHTIndexableView<T, 4>> input_grad,
                            Argument<Chain> chain_in, Result<Chain> chain_out,
                            StringAttribute padding,
                            ArrayAttribute<ssize_t> strides,
                            KernelErrorHandler handler,
                            const ExecutionContext& exec_ctx,
                            AsyncKernelFrame* frame) {
  const FixedRankShape<4> input_shape = input_grad->FixedShape();
  const FixedRankShape<4> filter_shape = kernel->FixedShape();
  const FixedRankShape<4> output_shape = output_grad->FixedShape();

  // Compute convolution parameters from input and kernel shapes.
  auto params = ComputeConv2DParams(input_shape, filter_shape, padding.get(),
                                    {strides[0], strides[1]});

  // Check that computed output shape matches gradient shape.
  TFRT_RETURN_IF_ERROR(handler, params.takeError());
  TFRT_RETURN_IF_ERROR(
      handler,
      CheckShapeMatch("output gradient shape", output_grad->FixedShape(),
                      "computed output shape", params->output_shape));

  const auto shuffle = [](const FixedRankShape<4>& shape, auto shfl) {
    return FixedRankShape<4>(
        {shape[shfl[0]], shape[shfl[1]], shape[shfl[2]], shape[shfl[3]]});
  };

  // Output grad: [N, OH, OW, OC].
  auto output_grad_t = AsEigenConstTensor(output_grad.get());

  // Filter: [FH, FW, IC, OC].
  auto filter_t = AsEigenConstTensor(kernel.get());

  // Input gradient: [N, IH, IW, IC].
  auto input_grad_t = AsEigenConstTensor(input_grad.get());

  // Filter reversed and shuffled: [FH^, FW^, OC, IC].
  std::array<bool, 4> filter_reverse = {true, true, false, false};
  std::array<ssize_t, 4> filter_shuffle = {0, 1, 3, 2};
  auto filter_shuffled =
      filter_t.reverse(filter_reverse).shuffle(filter_shuffle);

  // TODO(ezhulenev): Optimize gradient computation for 1x1 filter.

  // Convolution: [N, OH, OW, OC] X [FH, FW, OC, IC]-> [N, IH, IW, IC].
  auto grad_params = ComputeConv2DGradInputParams(*params);
  // clang-format off
  auto convolution =
      SpatialConvolution(output_grad_t, output_shape,
                         filter_shuffled, shuffle(filter_shape, filter_shuffle),
                         /*strides=*/grad_params.strides,
                         /*paddings=*/grad_params.paddings,
                         /*dilations=*/grad_params.dilations,
                         /*inflations=*/grad_params.inflations);
  // clang-format on

  auto on_done = [chain = chain_out.Allocate(),
                  frame = RAIIKernelFrame(*frame)]() { chain.emplace(); };

  AsyncAssign(exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>(),
              std::move(input_grad_t), std::move(convolution),
              std::move(on_done));
}

}  // namespace compat

void RegisterConv2DGradInputKernels(KernelRegistry* registry) {
  registry->AddKernel("eigen.conv2d.grad.input.f32",
                      TFRT_KERNEL(compat::Conv2DGradInput<float>));
}
}  // namespace tfrt
