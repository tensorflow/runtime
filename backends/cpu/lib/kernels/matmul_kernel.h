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

//===- matmul_kernel.h ------------------------------------------*- C++ -*-===//
//
// MatMul + Fusion kernel implementation (fusion added via output kernel).
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_MATMUL_KERNEL_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_MATMUL_KERNEL_H_

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {
namespace cpu {

// General matrix multiplication kernel:
//   C = alpha * AB + beta * C
//
// Link: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
template <typename T, typename OutputKernel>
static AsyncValueRef<Chain> MatMul(T alpha, const DenseHostTensor& a,
                                   const DenseHostTensor& b, T beta,
                                   DenseHostTensor* c,
                                   OutputKernel output_kernel,
                                   const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  DHTIndexableView<T, 2> a_view(&a);
  DHTIndexableView<T, 2> b_view(&b);
  MutableDHTIndexableView<T, 2> c_view(c);

  // Contraction dimension.
  Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dim({1, 0});

  auto in0 = compat::AsEigenConstTensor(a_view);
  auto in1 = compat::AsEigenConstTensor(b_view);
  auto out = compat::AsEigenTensor(c_view);

  const auto& ctx = host->GetOrCreateSharedContext<compat::EigenHostContext>();
  auto buffers = compat::KeepBuffers::alive(&a, &b, c);

  auto contract_expr = in0.contract(in1, contract_dim, output_kernel);

  if (alpha == 1.0 && beta == 0.0) {
    // Expression: C = AB
    return AsyncAssign(ctx, std::move(out), std::move(contract_expr),
                       std::move(buffers));

  } else if (alpha == 1.0) {
    // Expression: C = AB + beta * C
    auto expr = contract_expr + out.constant(beta) * out;
    return AsyncAssign(ctx, std::move(out), std::move(expr),
                       std::move(buffers));

  } else {
    // Expression: C = alpha * AB + beta * C
    auto expr = out.constant(alpha) * contract_expr + out.constant(beta) * out;
    return AsyncAssign(ctx, std::move(out), std::move(expr),
                       std::move(buffers));
  }
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_MATMUL_KERNEL_H_
