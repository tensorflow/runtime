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

// Tensorflow MatMul operations.

#include "matmul_ops.h"

#include <algorithm>
#include <complex>
#include <initializer_list>

#include "../../kernels/matmul_kernel.h"
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

using compat::AsyncEigenEvaluator;

static AsyncValueRef<DenseHostTensor> TfMatMulOp(
    const DenseHostTensor& a, const DenseHostTensor& b, const OpAttrsRef& attrs,
    const TensorMetadata& output_md, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto output = DenseHostTensor::CreateUninitialized(output_md, host);
  if (!output) {
    return EmitErrorAsync(exec_ctx, "out of memory allocating result");
  }

  bool transpose_a = attrs.GetAsserting<bool>("transpose_a");
  bool transpose_b = attrs.GetAsserting<bool>("transpose_b");

  AsyncEigenEvaluator evaluator(exec_ctx.host());

  // Dispatch based on the input data type.
  auto unsupported = [&](DType dtype) -> AsyncValueRef<Chain> {
    return EmitErrorAsync(exec_ctx, StrCat("Unsupported input dtype: ", dtype));
  };

  auto dispatch = [&](auto type_tag) -> AsyncValueRef<Chain> {
    using T = decltype(type_tag);
    return cpu::MatMul<T>(1.0, a, b, 0.0, &*output, transpose_a, transpose_b,
                          Eigen::NoOpOutputKernel(), evaluator);
  };

  internal::TypeDispatch<float, double, int32_t, int64_t, uint32_t, uint64_t,
                         std::complex<float>, std::complex<double>>
      type_dispatch(a.dtype());
  return ForwardValue(output.getValue(), type_dispatch(dispatch, unsupported));
}

}  // namespace

void RegisterTfMatmulCpuOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf.MatMul", TFRT_CPU_OP(TfMatMulOp),
                     CpuOpFlags::NoSideEffects, {"transpose_a", "transpose_b"});
}

}  // namespace tfrt
