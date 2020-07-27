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

//===- matmul_fusion_ops.cc - -----------------------------------*- C++ -*-===//
//
// Tensorflow MatMul fusion operations.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_OPS_TF_CWISE_UNARY_OPS_H_
#define TFRT_BACKENDS_CPU_OPS_TF_CWISE_UNARY_OPS_H_

#include "matmul_fusion_ops.h"

#include <algorithm>
#include <initializer_list>

#include "../../kernels/matmul_kernel.h"
#include "tfrt/common/compat/eigen/contraction_output_kernel.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/common/ops/tf/metadata_functions.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {
namespace {

static AsyncValueRef<DenseHostTensor> TfFusedMatMulOp(
    const DenseHostTensor& a, const DenseHostTensor& b,
    RepeatedArguments<DenseHostTensor> fusion_inputs, const OpAttrsRef& attrs,
    const TensorMetadata& output_md, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto output = DenseHostTensor::CreateUninitialized(output_md, host);
  if (!output) {
    return EmitErrorAsync(exec_ctx, "out of memory allocating result");
  }

  bool transpose_a = attrs.GetAsserting<bool>("transpose_a");
  bool transpose_b = attrs.GetAsserting<bool>("transpose_b");

  // TODO(ezhulenev): Add support for transposed operands.
  if (transpose_a || transpose_b) {
    return EmitErrorAsync(exec_ctx, "Transpose is not supported");
  }

  // Parse the MatMul fusion config.
  auto fused_ops_attr = attrs.GetAsserting<AggregateAttr>("fused_ops");
  SmallVector<string_view, 4> fused_ops(fused_ops_attr.GetNumElements());
  for (int i = 0; i < fused_ops_attr.GetNumElements(); ++i) {
    fused_ops[i] = fused_ops_attr.GetAttribute(i).cast<StringAttr>().GetValue();
  }

  if (fused_ops.empty()) {
    return EmitErrorAsync(exec_ctx,
                          "FusedMatMul must specify fused operations");
  }

  // Match the fusion to Eigen contraction output kernel.
  auto match_fusion = [&](std::initializer_list<string_view> ops) -> bool {
    return fused_ops.size() == ops.size() &&
           std::equal(fused_ops.begin(), fused_ops.end(), ops.begin());
  };

  // Dispatch fusion to correct Eigen contraction output kernel.
  auto dispatch_type = [&](auto type_tag) -> Expected<AsyncValueRef<Chain>> {
    using T = decltype(type_tag);

    // Validate BiasAdd operands.
    if (fused_ops[0] == "BiasAdd") {
      auto& bias = fusion_inputs[0];

      if (bias.shape().GetRank() != 1)
        return MakeStringError("Bias tensor must a vector");

      const ssize_t inner_dim = output_md.shape.GetDimensionSize(1);
      if (bias.NumElements() != inner_dim)
        return MakeStringError(
            "The number of bias elements ", bias.NumElements(),
            " doesn't match output inner dimension ", inner_dim);
    }

    // Fusion: BiasAdd
    if (match_fusion({"BiasAdd"})) {
      using OutputKernel = compat::BiasAddOutputKernel<T>;
      DHTArrayView<T> bias_view(&fusion_inputs[0]);
      OutputKernel output_kernel(compat::AsEigenConstTensor(bias_view));
      return cpu::MatMul<float>(1.0, a, b, 0.0, output.getPointer(),
                                std::move(output_kernel), exec_ctx);
    }

    // Fusion: BiasAdd + Relu
    if (match_fusion({"BiasAdd", "Relu"})) {
      using OutputKernel = compat::BiasAddOutputKernel<T, compat::Relu>;
      DHTArrayView<T> bias_view(&fusion_inputs[0]);
      OutputKernel output_kernel(compat::AsEigenConstTensor(bias_view));

      return cpu::MatMul<float>(1.0, a, b, 0.0, output.getPointer(),
                                std::move(output_kernel), exec_ctx);
    }

    return MakeStringError("Unsupported fusion type");
  };

  // Dispatch based on the input data type.
  auto dispatch = [&]() -> Expected<AsyncValueRef<Chain>> {
    switch (a.dtype().kind()) {
      default:
        return MakeStringError("Unsupported dtype: ", a.dtype());
      case DType::F32:
        return dispatch_type(float{});
    }
  };

  // Failed to dispatch fusion expression.
  auto expected_chain = dispatch();
  if (auto err = expected_chain.takeError())
    return EmitErrorAsync(exec_ctx, std::move(err));

  return ForwardValue(output.getValue(), std::move(*expected_chain), host);
}

}  // namespace

void RegisterTfMatmulFusionCpuOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf._FusedMatMul", TFRT_CPU_OP(TfFusedMatMulOp),
                     CpuOpFlags::NoSideEffects,
                     {"transpose_a", "transpose_b", "fused_ops"});
}

}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_OPS_TF_CWISE_UNARY_OPS_H_
