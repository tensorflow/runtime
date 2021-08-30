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

// MatMul + Fusion kernel implementation (fusion added via output kernel).

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
namespace {

template <typename OutputKernel, typename T, typename EigenEvaluator>
typename EigenEvaluator::DependencyToken FusedMatMulInternal(
    const DenseHostTensor& a, const DenseHostTensor& b, DenseHostTensor* output,
    const DenseHostTensor& fusion_input, bool transpose_a, bool transpose_b,
    EigenEvaluator eigen) {
  DHTArrayView<T> bias_view(&fusion_input);
  OutputKernel output_kernel(compat::AsEigenConstTensor(bias_view));
  return cpu::MatMul<T>(1.0, a, b, 0.0, output, transpose_a, transpose_b,
                        std::move(output_kernel), eigen);
}

}  // namespace

template <typename T, typename EigenEvaluator, typename FuseInputsRange>
typename EigenEvaluator::DependencyToken FusedMatMul(
    const DenseHostTensor& a, const DenseHostTensor& b, DenseHostTensor* output,
    FuseInputsRange fusion_inputs, bool transpose_a, bool transpose_b,
    AggregateAttr fused_ops_attr, const ExecutionContext& exec_ctx) {
  static_assert(std::is_same<std::decay_t<decltype(fusion_inputs[0])>,
                             DenseHostTensor>::value,
                "fusion_inputs must be a range of DenseHostTensor");

  EigenEvaluator eigen{exec_ctx.host()};

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

    const Index inner_dim = output->shape().GetDimensionSize(1);
    if (bias.NumElements() != inner_dim)
      return eigen.MakeError("The number of bias elements ", bias.NumElements(),
                             " doesn't match output inner dimension ",
                             inner_dim);
  }

  // Fusion: BiasAdd
  if (match_fusion({"BiasAdd"})) {
    return FusedMatMulInternal<compat::BiasAddOutputKernel<T>, T>(
        a, b, output, fusion_inputs[0], transpose_a, transpose_b, eigen);
  }

  // Fusion: BiasAdd + Relu
  if (match_fusion({"BiasAdd", "Relu"})) {
    return FusedMatMulInternal<compat::BiasAddOutputKernel<T, compat::Relu>, T>(
        a, b, output, fusion_inputs[0], transpose_a, transpose_b, eigen);
  }

  // Fusion: BiasAdd + Relu6
  if (match_fusion({"BiasAdd", "Relu6"})) {
    return FusedMatMulInternal<compat::BiasAddOutputKernel<T, compat::Relu6>,
                               T>(a, b, output, fusion_inputs[0], transpose_a,
                                  transpose_b, eigen);
  }

  // Fusion: BiasAdd + Elu
  if (match_fusion({"BiasAdd", "Elu"})) {
    return FusedMatMulInternal<compat::BiasAddOutputKernel<T, compat::Elu>, T>(
        a, b, output, fusion_inputs[0], transpose_a, transpose_b, eigen);
  }

  return eigen.MakeError("Unsupported fusion type");
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_FUSED_MATMUL_KERNEL_H_
