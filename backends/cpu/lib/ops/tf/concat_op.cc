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

static Expected<DenseHostTensor> TfConcatOp(
    RepeatedArguments<DenseHostTensor> inputs,
    const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto axis = cpu::ConcatAxis(inputs[inputs.size() - 1]);
  if (!axis) return axis.takeError();

  // The last DHT in inputs is the axis. Make a range view to cover all but the
  // last DHT inputs as args.
  auto args = views::Counted(inputs.begin(), inputs.size() - 1);

  auto output_md = cpu::ConcatMetadataKernel(args, *axis);
  if (!output_md) return output_md.takeError();

  // Allocate output tensor.
  auto dest = DenseHostTensor::CreateUninitialized(*output_md, host);
  if (!dest) {
    return MakeStringError("out of memory allocating result");
  }

  // Call concat kernel.
  switch (args[0].dtype().kind()) {
    default:
      return MakeStringError("Unsupported dtype: ", args[0].dtype());
      break;
    case DType::I1: {
      using T = EigenTypeForDTypeKind<DType::I1>;
      auto error = ::tfrt::cpu::ConcatKernel<T>(args, *axis, dest.getPointer());
      if (error) return std::move(error);
      break;
    }
#define DTYPE_NUMERIC(ENUM)                                                    \
  case DType::ENUM: {                                                          \
    using T = EigenTypeForDTypeKind<DType::ENUM>;                              \
    auto error = ::tfrt::cpu::ConcatKernel<T>(args, *axis, dest.getPointer()); \
    if (error) return std::move(error);                                        \
    break;                                                                     \
  }
#include "tfrt/dtype/dtype.def"  // NOLINT
  }

  return std::move(*dest);
}  // namespace

}  // namespace

void RegisterTfConcatCpuOp(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf.ConcatV2", TFRT_CPU_OP(TfConcatOp),
                     CpuOpFlags::NoSideEffects);
}

}  // namespace tfrt
