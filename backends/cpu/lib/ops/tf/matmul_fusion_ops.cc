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

// Tensorflow MatMul fusion operations.

#include "matmul_fusion_ops.h"

#include <algorithm>
#include <initializer_list>

#include "../../kernels/fused_matmul_kernel.h"
#include "tfrt/common/compat/eigen/eigen_evaluator.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/tensor_serialize_utils.h"
#include "type_dispatch.h"

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
  auto fused_ops_attr = attrs.GetAsserting<AggregateAttr>("fused_ops");

  // Dispatch based on the input data type.
  auto unsupported = [&](DType dtype) -> AsyncValueRef<Chain> {
    return EmitErrorAsync(exec_ctx, StrCat("Unsupported input dtype: ", dtype));
  };

  auto dispatch = [&](auto type_tag) -> AsyncValueRef<Chain> {
    using T = decltype(type_tag);
    return cpu::FusedMatMul<T, compat::AsyncEigenEvaluator>(
        a, b, &*output, fusion_inputs, transpose_a, transpose_b, fused_ops_attr,
        exec_ctx);
  };

  // TODO(ezhulenev): Keep these types consistent with graph rewrite that
  // does fusion (kernel matcher pass).
  internal::TypeDispatch<float, int32_t> type_dispatch(a.dtype());
  return ForwardValue(output.getValue(), type_dispatch(dispatch, unsupported),
                      host);
}

}  // namespace

void RegisterTfMatmulFusionCpuOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf._FusedMatMul", TFRT_CPU_OP(TfFusedMatMulOp),
                     CpuOpFlags::NoSideEffects,
                     {"transpose_a", "transpose_b", "fused_ops"});
}

}  // namespace tfrt
