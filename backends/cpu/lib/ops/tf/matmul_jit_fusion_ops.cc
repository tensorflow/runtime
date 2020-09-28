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

//===- matmul_jit_fusion_ops.cc - -------------------------------*- C++ -*-===//
//
// Tensorflow MatMul Jit fusion operations.
//
//===----------------------------------------------------------------------===//

#include "matmul_jit_fusion_ops.h"

#include "../../kernels/matmul_kernel.h"
#include "tfrt/common/compat/eigen/eigen_evaluator.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/cpu/jit/contraction_output_kernel.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "type_dispatch.h"

namespace tfrt {
namespace {

static AsyncValueRef<DenseHostTensor> TfJitFusedMatMulOp(
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

  // Collect output kernel names from the attribute.
  auto fusion = attrs.GetAsserting<AggregateAttr>("fusion");
  SmallVector<string_view, 4> fusion_kernels(fusion.GetNumElements());
  for (int i = 0; i < fusion.GetNumElements(); ++i) {
    fusion_kernels[i] = fusion.GetAttributeOfType<StringAttr>(i).GetValue();
  }

  // Collect output block and additional arguments dtypes.
  const DType output_dtype = a.dtype();
  SmallVector<DType, 8> additional_args_dtypes;
  for (DenseHostTensor& dht : fusion_inputs) {
    additional_args_dtypes.push_back(dht.dtype());
  }

  // Compile fusion into the contraction output kernel.
  auto compiled_kernel = cpu::jit::GetCompiledContractionOutputKernel(
      host, fusion_kernels, attrs, output_dtype, additional_args_dtypes);
  if (auto err = compiled_kernel.takeError()) {
    return EmitErrorAsync(exec_ctx,
                          StrCat("Failed to compiled output kernel: ", err));
  }

  // Convert fusion inputs to compiled kernel arguments.
  SmallVector<const DenseHostTensor*, 8> additional_args;
  for (DenseHostTensor& fusion_input : fusion_inputs) {
    additional_args.push_back(&fusion_input);
  }

  // Verify that additional arguments are compatible with the compiled kernel.
  if (auto err = cpu::jit::VerifyCompiledContractionOutoutKernelArgs(
          *compiled_kernel, output_dtype, additional_args)) {
    return EmitErrorAsync(
        exec_ctx, StrCat("Illegal output kernel additional arguments: ", err));
  }

  // Dispatch to the correct data type expression.
  auto unsupported = [&](DType dtype) -> AsyncValueRef<Chain> {
    return EmitErrorAsync(exec_ctx, StrCat("Unsupported input dtype: ", dtype));
  };

  auto dispatch = [&](auto type_tag) -> AsyncValueRef<Chain> {
    using T = decltype(type_tag);
    cpu::jit::ContractionOutputKernel<float> output_kernel(*compiled_kernel,
                                                           additional_args);
    return cpu::MatMul<T>(1.0, a, b, 0.0, &*output, transpose_a, transpose_b,
                          std::move(output_kernel),
                          compat::AsyncEigenEvaluator(host));
  };

  internal::TypeDispatch<float> type_dispatch(a.dtype());
  return ForwardValue(output.getValue(), type_dispatch(dispatch, unsupported),
                      host);
}

}  // namespace

void RegisterTfMatmulJitFusionCpuOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf._JitFusedMatMul", TFRT_CPU_OP(TfJitFusedMatMulOp),
                     CpuOpFlags::NoSideEffects,
                     {"transpose_a", "transpose_b", "fusion"});
}

}  // namespace tfrt
