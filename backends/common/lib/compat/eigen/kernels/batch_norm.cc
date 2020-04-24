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

//===- batch_norm.cc ---------------------------------------------*- C++-*-===//
//
// Batch normalization kernels implemented with Eigen.
//
//===----------------------------------------------------------------------===//

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
static void BatchNorm(ArgumentView<MutableDHTIndexableView<T, 4>> input,
                      ArgumentView<MutableDHTArrayView<T>> gamma,  // scale
                      ArgumentView<MutableDHTArrayView<T>> beta,   // offset
                      ArgumentView<MutableDHTArrayView<T>> moving_mean,
                      ArgumentView<MutableDHTArrayView<T>> moving_variance,
                      Argument<Chain> chain_in, Result<Chain> chain_out,
                      Attribute<float> epsilon, KernelErrorHandler handler,
                      HostContext* host, KernelFrame* frame) {
  // shape_input has format (batch_size, height, width, channel_num).
  const auto& shape_input = input->FixedShape();

  if (gamma->NumElements() != shape_input[3] ||
      beta->NumElements() != shape_input[3] ||
      moving_mean->NumElements() != shape_input[3] ||
      moving_variance->NumElements() != shape_input[3]) {
    handler.ReportError("BatchNorm parameter dimension does not match",
                        " the input channel number ", shape_input[3]);
    return;
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

  auto input_t = AsEigenTensor(input.get());
  auto gamma_t = AsEigenConstTensor(gamma.get());
  auto beta_t = AsEigenConstTensor(beta.get());
  auto mean_t = AsEigenConstTensor(moving_mean.get());
  auto variance_t = AsEigenConstTensor(moving_variance.get());

  auto expr_0 = (input_t.reshape(rest_by_depth) - to_rest_by_depth(mean_t)) /
                to_rest_by_depth((variance_t + epsilon.get()).sqrt().eval());

  auto expr = to_rest_by_depth(gamma_t) * expr_0 + to_rest_by_depth(beta_t);

  auto out = input_t.reshape(rest_by_depth);

  AsyncAssign(host->GetOrCreateSharedContext<EigenHostContext>(),
              std::move(out), std::move(expr),
              [chain = chain_out.Allocate(),
               frame = RAIIKernelFrame(*frame)]() { chain.emplace(); });
}

}  // namespace compat

void RegisterBatchNormKernels(KernelRegistry* registry) {
  registry->AddKernel("eigen.batch_norm.f32",
                      TFRT_KERNEL(compat::BatchNorm<float>));
}

}  // namespace tfrt
