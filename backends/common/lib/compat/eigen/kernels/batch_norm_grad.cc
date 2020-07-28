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

//===- batch_norm_grad.cc ----------------------------------------*- C++-*-===//
//
// Batch normalization gradient kernels implemented with Eigen.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/kernels/shape_functions.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace compat {

template <typename T>
static void BatchNormGrad(
    // clang-format off
    // Inputs --------------------------------------------------------------- //
    ArgumentView<DHTIndexableView<T, 4>> output_grad,
    ArgumentView<DHTIndexableView<T, 4>> input,
    ArgumentView<DHTArrayView<T>> gamma,  // scale
    ArgumentView<DHTArrayView<T>> moving_mean,
    ArgumentView<DHTArrayView<T>> moving_variance,
    Argument<Chain> chain_in,
    // Outputs -------------------------------------------------------------- //
    ArgumentView<MutableDHTIndexableView<T, 4>> input_grad,
    ArgumentView<MutableDHTArrayView<T>> gamma_grad,  // scale_grad
    ArgumentView<MutableDHTArrayView<T>> beta_grad,  // offset_grad
    Result<Chain> input_grad_chain,
    Result<Chain> gamma_grad_chain,
    Result<Chain> beta_grad_chain,
    // Attributes ----------------------------------------------------------- //
    Attribute<float> epsilon,
    // Execution context ---------------------------------------------------- //
    KernelErrorHandler handler, const ExecutionContext& exec_ctx,
    AsyncKernelFrame* frame) {
  // clang-format on

  // Note: the following formulas are used to compute the gradients for
  // back propagation:
  //
  // output_grad = scale * rsqrt(variance + epsilon) *
  //               (output_grad - mean(output_grad) - (x - mean(x)) *
  //                mean(output_grad * (x - mean(x))) / (variance + epsilon))
  //
  // gamma_grad  = sum(output_grad *
  //                  (x - mean(x)) * rsqrt(variance + epsilon))
  //
  // beta_grad   = sum(output_grad)

  TFRT_RETURN_IF_ERROR(
      handler, CheckShapeMatch("output_grad shape", output_grad->Shape(),
                               "input_grad shape", input_grad->Shape()));
  TFRT_RETURN_IF_ERROR(
      handler, CheckShapeMatch("input shape", input->Shape(),
                               "input_grad shape", input_grad->Shape()));

  // Data format: (batch_size, height, width, num_channels) [NHWC]
  const FixedRankShape<4>& input_grad_shape = input_grad->FixedShape();

  const auto depth = input_grad_shape[3];  // num channels
  const auto channels_shape = TensorShape{depth};

  TFRT_RETURN_IF_ERROR(
      handler, CheckShapeMatch("channels dimension size", channels_shape,
                               "gamma shape", gamma->Shape()));
  TFRT_RETURN_IF_ERROR(
      handler, CheckShapeMatch("channels dimension size", channels_shape,
                               "mean shape", moving_mean->Shape()));
  TFRT_RETURN_IF_ERROR(
      handler, CheckShapeMatch("channels dimension size", channels_shape,
                               "variance shape", moving_variance->Shape()));
  TFRT_RETURN_IF_ERROR(
      handler, CheckShapeMatch("channels dimension size", channels_shape,
                               "gamma_grad shape", gamma_grad->Shape()));
  TFRT_RETURN_IF_ERROR(
      handler, CheckShapeMatch("channels dimension size", channels_shape,
                               "beta_grad shape", beta_grad->Shape()));

  // Flatten all outer dimensions of input{grad}/output_grad.
  const ssize_t rest_size = output_grad->NumElements() / depth;
  Eigen::DSizes<Eigen::Index, 2> rest_by_depth(rest_size, depth);

  // Resize all vectors into 2d Tensors.
  Eigen::DSizes<Eigen::Index, 2> one_by_depth(1, depth);

  // Compute reductions (sum and mean) over outer dimension (after flattening).
  Eigen::DSizes<int, 1> reduce_dims(0);

  // Broadcast 1d vectors to the input/output shape.
  Eigen::DSizes<int, 2> bcast_spec(rest_size, 1);

  auto& ctx = exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>();

  // Reshape input/output arguments into [rest_size, depth] tensors.
  const FixedRankShape<2> rest_by_depth_s = AsShape(rest_by_depth);

  auto output_grad_t = AsEigenConstTensor(output_grad.get(), rest_by_depth_s);
  auto input_t = AsEigenConstTensor(input.get(), rest_by_depth_s);
  auto input_grad_t = AsEigenTensor(input_grad.get(), rest_by_depth_s);

  // Reshape input vectors into [1, depth] tensors.
  const FixedRankShape<2> one_by_depth_s = AsShape(one_by_depth);

  auto gamma_t = AsEigenConstTensor(gamma.get(), one_by_depth_s);
  auto mean_t = AsEigenConstTensor(moving_mean.get(), one_by_depth_s);
  auto variance_t = AsEigenConstTensor(moving_variance.get(), one_by_depth_s);

  // Output gradients of [depth] shape.
  auto gamma_grad_t = AsEigenTensor(gamma_grad.get());
  auto beta_grad_t = AsEigenTensor(beta_grad.get());

  T rest_size_inv = static_cast<T>(1.0f / static_cast<T>(rest_size));

  auto coef0 = (variance_t + epsilon.get()).rsqrt();            // [1, depth]
  auto coef1 = (gamma_t * coef0).eval().broadcast(bcast_spec);  // [rest, depth]

  auto input_centered = (input_t - mean_t.broadcast(bcast_spec));
  auto input_scaled = input_centered * (coef0.eval().broadcast(bcast_spec));

  // Allocate output chains for all results, because they must be not null
  // before we construct RAIIKernelFrame below.
  auto input_grad_ready = input_grad_chain.Allocate();
  auto gamma_grad_ready = gamma_grad_chain.Allocate();
  auto beta_grad_ready = beta_grad_chain.Allocate();

  //=== gamma/scale gradient ----------------------------------------------===//
  auto gamma_grad_expr = (output_grad_t * input_scaled).sum(reduce_dims);

  AsyncAssign(ctx, std::move(gamma_grad_t), std::move(gamma_grad_expr),
              [chain = std::move(gamma_grad_ready),
               frame = RAIIKernelFrame(*frame)]() { chain.emplace(); });

  //=== beta/offset gradient ----------------------------------------------===//
  auto output_grad_sum = output_grad_t.sum(reduce_dims);

  AsyncAssign(ctx, std::move(beta_grad_t), output_grad_sum,
              [chain = std::move(beta_grad_ready),
               frame = RAIIKernelFrame(*frame)]() { chain.emplace(); });

  //=== input gradient ----------------------------------------------------===//
  auto output_grad_sum_one_by_depth =
      output_grad_sum.eval().reshape(one_by_depth);
  auto output_grad_mean_one_by_depth =
      output_grad_sum_one_by_depth * rest_size_inv;
  auto output_grad_mean = output_grad_mean_one_by_depth.broadcast(bcast_spec);

  auto output_grad_centered = output_grad_t - output_grad_mean;

  auto coef2 =
      (coef0.square() *
       (output_grad_t * input_centered).mean(reduce_dims).reshape(one_by_depth))
          .eval()
          .broadcast(bcast_spec);

  auto input_grad_expr =
      coef1 * (output_grad_centered - input_centered * coef2);

  AsyncAssign(ctx, std::move(input_grad_t), std::move(input_grad_expr),
              [chain = std::move(input_grad_ready),
               frame = RAIIKernelFrame(*frame)]() { chain.emplace(); });
}

}  // namespace compat

void RegisterBatchNormGradKernels(KernelRegistry* registry) {
  registry->AddKernel("eigen.batch_norm.grad.f32",
                      TFRT_KERNEL(compat::BatchNormGrad<float>));
}

}  // namespace tfrt
