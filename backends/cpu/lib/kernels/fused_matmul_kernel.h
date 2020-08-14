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

//===- fused_matmul_kernel.h ------------------------------------*- C++ -*-===//
//
// MatMul + Fusion kernel implementation (fusion added via output kernel).
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_FUSED_MATMUL_KERNEL_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_FUSED_MATMUL_KERNEL_H_

#include "./matmul_kernel.h"
#include "tfrt/common/compat/eigen/contraction_output_kernel.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/common/compat/eigen/eigen_evaluator.h"
#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace cpu {

template <typename T, typename EigenEvaluator, typename FuseInputsRange>
typename EigenEvaluator::DependencyToken FusedMatMul(
    const DenseHostTensor& a, const DenseHostTensor& b, DenseHostTensor* output,
    FuseInputsRange fusion_inputs, bool transpose_a, bool transpose_b,
    AggregateAttr fused_ops_attr, const ExecutionContext& exec_ctx) {
  EigenEvaluator eigen{exec_ctx.host()};

  // TODO(ezhulenev): Add support for transposed operands.
  if (transpose_a || transpose_b) {
    return eigen.MakeError("Transpose is not supported");
  }

  // Parse the MatMul fusion config.
  SmallVector<string_view, 4> fused_ops(fused_ops_attr.GetNumElements());
  for (int i = 0; i < fused_ops_attr.GetNumElements(); ++i) {
    fused_ops[i] = fused_ops_attr.GetAttribute(i).cast<StringAttr>().GetValue();
  }

  if (fused_ops.empty()) {
    return eigen.MakeError("FusedMatMul must specify fused operations");
  }

  // Match the fusion to Eigen contraction output kernel.
  auto match_fusion = [&](std::initializer_list<string_view> ops) -> bool {
    return fused_ops.size() == ops.size() &&
           std::equal(fused_ops.begin(), fused_ops.end(), ops.begin());
  };

  // Dispatch fusion to correct Eigen contraction output kernel.
  // Validate BiasAdd operands.
  if (fused_ops[0] == "BiasAdd") {
    auto& bias = fusion_inputs[0];

    if (bias.shape().GetRank() != 1)
      return eigen.MakeError("Bias tensor must a vector");

    const ssize_t inner_dim = output->shape().GetDimensionSize(1);
    if (bias.NumElements() != inner_dim)
      return eigen.MakeError("The number of bias elements ", bias.NumElements(),
                             " doesn't match output inner dimension ",
                             inner_dim);
  }

  // Fusion: BiasAdd
  if (match_fusion({"BiasAdd"})) {
    using OutputKernel = compat::BiasAddOutputKernel<T>;
    DHTArrayView<T> bias_view(&fusion_inputs[0]);
    OutputKernel output_kernel(compat::AsEigenConstTensor(bias_view));
    return cpu::MatMul<T>(1.0, a, b, 0.0, output, std::move(output_kernel),
                          eigen);
  }

  // Fusion: BiasAdd + Relu
  if (match_fusion({"BiasAdd", "Relu"})) {
    using OutputKernel = compat::BiasAddOutputKernel<T, compat::Relu>;
    DHTArrayView<T> bias_view(&fusion_inputs[0]);
    OutputKernel output_kernel(compat::AsEigenConstTensor(bias_view));

    return cpu::MatMul<T>(1.0, a, b, 0.0, output, std::move(output_kernel),
                          eigen);
  }

  return eigen.MakeError("Unsupported fusion type");
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_FUSED_MATMUL_KERNEL_H_
