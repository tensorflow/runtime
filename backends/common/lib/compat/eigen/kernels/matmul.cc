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

//===- matmul.cc -------------------------------------------------*- C++-*-===//
//
// Matrix multiplication kernels implemented with Eigen.
//
//===----------------------------------------------------------------------===//

#include "tfrt/common/compat/eigen/contraction_kernel.h"
#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {
namespace compat {

using ::Eigen::Index;
using ::tfrt::compat::AsEigenConstTensor;
using ::tfrt::compat::AsEigenTensor;

// General matrix multiplication kernel:
//   C = alpha * AB + beta * C
//
// Link: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
template <typename T>
void MatMul(Argument<T> alpha, ArgumentView<DHTIndexableView<T, 2>> a,
            ArgumentView<DHTIndexableView<T, 2>> b, Argument<T> beta,
            ArgumentView<MutableDHTIndexableView<T, 2>> c,
            Argument<Chain> chain_in, Result<Chain> chain_out,
            KernelErrorHandler handler, const ExecutionContext& exec_ctx,
            AsyncKernelFrame* frame) {
  const auto& shape_a = a->FixedShape();
  const auto& shape_b = b->FixedShape();
  const auto& shape_c = c->FixedShape();
  if (shape_a[1] != shape_b[0]) {
    handler.ReportError("MatMul input tensors inner dimension mismatch: ",
                        shape_a, " vs. ", shape_b);
    return;
  }
  if (shape_c[0] != shape_a[0] || shape_c[1] != shape_b[1]) {
    handler.ReportError("MatMul output shape ", shape_c,
                        " does not match product shape of inputs: ", shape_a,
                        " * ", shape_b);
    return;
  }

  // Contraction dimension.
  Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dim({1, 0});

  auto on_done = [chain = chain_out.Allocate(),
                  frame = RAIIKernelFrame(*frame)]() { chain.emplace(); };

  auto in0 = AsEigenConstTensor(a.get());
  auto in1 = AsEigenConstTensor(b.get());
  auto out = AsEigenTensor(c.get());

  const EigenHostContext& cpu =
      exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>();

  if (alpha.get() == 1.0 && beta.get() == 0.0) {
    auto expr = in0.contract(in1, contract_dim);
    AsyncAssign(cpu, std::move(out), std::move(expr), std::move(on_done));

  } else if (alpha.get() == 1.0) {
    auto expr =
        in0.contract(in1, contract_dim) + out.constant(beta.get()) * out;
    AsyncAssign(cpu, std::move(out), std::move(expr), std::move(on_done));

  } else {
    auto expr = out.constant(alpha.get()) * in0.contract(in1, contract_dim) +
                out.constant(beta.get()) * out;
    AsyncAssign(cpu, std::move(out), std::move(expr), std::move(on_done));
  }
}

}  // namespace compat

void RegisterMatMulKernels(KernelRegistry* registry) {
  registry->AddKernel("eigen.matmul.f32", TFRT_KERNEL(compat::MatMul<float>));
  registry->AddKernel("eigen.matmul.i32", TFRT_KERNEL(compat::MatMul<int32_t>));
}

}  // namespace tfrt
