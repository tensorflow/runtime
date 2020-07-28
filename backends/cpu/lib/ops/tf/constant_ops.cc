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

//===- constant_ops.cc - ----------------------------------------*- C++ -*-===//
//
// Tensorflow constant operation.
//
//===----------------------------------------------------------------------===//

#include "constant_ops.h"

#include <sys/types.h>

#include <cstdint>

#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/host_tensor.h"
#include "tfrt/tensor/scalar_host_tensor.h"

namespace tfrt {
namespace {

static AsyncValueRef<HostTensor> TfZerosLike(const HostTensor& input,
                                             const TensorMetadata& output_md,
                                             const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  switch (output_md.dtype.kind()) {
    default:
      return EmitErrorAsync(exec_ctx, "unsupported dtype");
      break;
#define DTYPE_NUMERIC(ENUM)                                       \
  case DType::ENUM: {                                             \
    using T = EigenTypeForDTypeKind<DType::ENUM>;                 \
    return host->MakeAvailableAsyncValueRef<ScalarHostTensor<T>>( \
        output_md, static_cast<T>(0.0));                          \
  }
#include "tfrt/dtype/dtype.def"  // NOLINT
  }
}

}  // namespace

void RegisterTfConstantCpuOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf.ZerosLike", TFRT_CPU_OP(TfZerosLike),
                     CpuOpFlags::NoSideEffects);
}

}  // namespace tfrt
