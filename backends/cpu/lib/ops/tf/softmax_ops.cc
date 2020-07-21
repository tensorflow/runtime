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

//===- softmax_ops.cc - -----------------------------------------*- C++ -*-===//
//
// Softmax and LogSoftmax Tensorflow operations.
//
//===----------------------------------------------------------------------===//

#include "softmax_ops.h"

#include "../../kernels/softmax_kernel.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {
namespace {

template <bool log>
static AsyncValueRef<DenseHostTensor> TfSoftmaxOp(
    const DenseHostTensor& logits, const TensorMetadata& output_md,
    const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto dest = DenseHostTensor::CreateUninitialized(output_md, host);
  if (!dest) {
    return EmitErrorAsync(exec_ctx, "out of memory allocating result");
  }

  AsyncValueRef<Chain> chain;
  switch (logits.dtype().kind()) {
    default:
      chain = EmitErrorAsync(exec_ctx, "unsupported dtype");
      break;
#define DTYPE_FLOAT(ENUM)                                             \
  case DType::ENUM: {                                                 \
    chain = ::tfrt::cpu::Softmax<EigenTypeForDTypeKind<DType::ENUM>>( \
        logits, log, dest.getPointer(), exec_ctx);                    \
  } break;
#include "tfrt/dtype/dtype.def"  // NOLINT
  }

  return ForwardValue(dest.getValue(), std::move(chain), host);
}

}  // namespace

void RegisterTfSofmaxCpuOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf.Softmax", TFRT_CPU_OP(TfSoftmaxOp<false>),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tf.LogSoftmax", TFRT_CPU_OP(TfSoftmaxOp<true>),
                     CpuOpFlags::NoSideEffects);
}

}  // namespace tfrt
