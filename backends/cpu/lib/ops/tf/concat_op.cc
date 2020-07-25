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

//===- concat_op.cc - -------------------------------------------*- C++ -*-===//
//
// Tensorflow Concat operations.
//
//===----------------------------------------------------------------------===//

#include "concat_op.h"

#include <cstdint>

#include "../../kernels/concat_kernel.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace {

static AsyncValueRef<DenseHostTensor> TfConcatOp(
    RepeatedArguments<DenseHostTensor> args, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  // Parse concatenation axis.
  Expected<int64_t> expected_axis = cpu::ConcatAxis(args[args.size() - 1]);
  if (auto err = expected_axis.takeError())
    return EmitErrorAsync(exec_ctx, std::move(err));

  // Compute the actual axis from a negative value.
  const int rank = args[0].shape().GetRank();
  int64_t axis = *expected_axis < 0 ? *expected_axis + rank : *expected_axis;

  // Compute the output tensor shape.
  llvm::SmallVector<const DenseHostTensor*, 4> inputs;
  for (int i = 0; i < args.size() - 1; ++i) inputs.push_back(&args[i]);
  auto expected_output_md = cpu::ConcatTensorMetadata(inputs, axis);
  if (auto err = expected_output_md.takeError())
    return EmitErrorAsync(exec_ctx, std::move(err));

  // Allocate output tensor.
  auto dest = DenseHostTensor::CreateUninitialized(*expected_output_md, host);
  if (!dest) {
    return EmitErrorAsync(exec_ctx, "out of memory allocating result");
  }

  // Call concat kernel.
  AsyncValueRef<Chain> chain;

  switch (args[0].dtype().kind()) {
    default:
      chain = EmitErrorAsync(exec_ctx,
                             StrCat("Unsupported dtype: ", args[0].dtype()));
      break;
#define DTYPE_NUMERIC(ENUM)                                               \
  case DType::ENUM: {                                                     \
    using T = EigenTypeForDTypeKind<DType::ENUM>;                         \
    chain = ::tfrt::cpu::ConcatKernel<T>(inputs, axis, dest.getPointer(), \
                                         exec_ctx);                       \
  } break;
#include "tfrt/dtype/dtype.def"  // NOLINT
  }

  return ForwardValue(dest.getValue(), std::move(chain), host);
}

}  // namespace

void RegisterTfConcatCpuOp(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf.ConcatV2", TFRT_CPU_OP(TfConcatOp),
                     CpuOpFlags::NoSideEffects);
}

}  // namespace tfrt
