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

//===- shape_ops.cc - -------------------------------------------*- C++ -*-===//
//
// Tensorflow shape related operations.
//
//===----------------------------------------------------------------------===//

#include "shape_ops.h"

#include <algorithm>

#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace {

static AsyncValueRef<DenseHostTensor> TfShapeOp(
    const DenseHostTensor& input, const TensorMetadata& output_md,
    const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  auto dest = DenseHostTensor::MakeConstructedAsyncValueRef(output_md, host);
  if (!dest) {
    return EmitErrorAsync(exec_ctx, "out of memory allocating result");
  }

  SmallVector<ssize_t, 4> dims(input.shape().GetRank());
  input.shape().GetDimensions(dims);

  if (output_md.dtype.kind() == DType::I32) {
    MutableDHTArrayView<int32_t> view(&dest.get());
    std::transform(dims.begin(), dims.end(), view.data(),
                   [](ssize_t dim) { return static_cast<int32_t>(dim); });

  } else if (output_md.dtype.kind() == DType::I64) {
    MutableDHTArrayView<int64_t> view(&dest.get());
    std::transform(dims.begin(), dims.end(), view.data(),
                   [](ssize_t dim) { return static_cast<int64_t>(dim); });
  } else {
    return EmitErrorAsync(exec_ctx, "Unsupported output data type");
  }

  dest.SetStateConcrete();
  return dest;
}

}  // namespace

void RegisterTfShapeCpuOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf.Shape", TFRT_CPU_OP(TfShapeOp),
                     CpuOpFlags::NoSideEffects);
}

}  // namespace tfrt
