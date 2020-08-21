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
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace {

//===----------------------------------------------------------------------===//
// tf.Shape op
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// tf.ExpandDims op
//===----------------------------------------------------------------------===//

static Expected<int64_t> GetExpandAxisValue(const DenseHostTensor& axis) {
  int64_t axis_value;

  if (axis.NumElements() != 1)
    return MakeStringError("Axis must be a scalar tensor");

  if (axis.dtype().kind() == DType::I32) {
    DHTArrayView<int32_t> view(&axis);
    axis_value = *view.begin();
  } else if (axis.dtype().kind() == DType::I64) {
    DHTArrayView<int64_t> view(&axis);
    axis_value = *view.begin();
  } else {
    return MakeStringError("Unsupported axis data type");
  }

  return axis_value;
}

static AsyncValueRef<DenseHostTensor> TfExpandDimsOp(
    const DenseHostTensor& input, const DenseHostTensor& axis,
    const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  const TensorShape& input_shape = input.shape();
  const int input_rank = input_shape.GetRank();

  // Parse expand axis.
  auto expected_axis = GetExpandAxisValue(axis);
  if (auto err = expected_axis.takeError()) {
    return EmitErrorAsync(exec_ctx, std::move(err));
  }

  // Check that axis value is correct.
  if (*expected_axis < -1 - input_rank || *expected_axis > input_rank) {
    return EmitErrorAsync(exec_ctx,
                          StrCat("Failed to expand axis ", *expected_axis,
                                 " for tensor of rank ", input_rank));
  }

  // We emulate numpy's interpretation of the dim axis when
  // -input_rank >= axis <= input_rank.
  int64_t expand_axis =
      *expected_axis < 0 ? *expected_axis + input_rank + 1 : *expected_axis;

  // Compute the new tensor shape after expansion.
  SmallVector<ssize_t, 4> output_dims(input_rank + 1);
  for (int d = 0; d < expand_axis; ++d) {
    output_dims[d] = input_shape.GetDimensionSize(d);
  }
  output_dims[expand_axis] = 1;
  for (int d = expand_axis + 1; d < input_rank + 1; ++d) {
    output_dims[d] = input_shape.GetDimensionSize(d - 1);
  }

  TensorMetadata output_md(input.metadata().dtype, output_dims);
  return MakeAvailableAsyncValueRef<DenseHostTensor>(host, output_md,
                                                     input.buffer().CopyRef());
}

}  // namespace

//===----------------------------------------------------------------------===//
// Tensorflow shape related ops registration.
//===----------------------------------------------------------------------===//

void RegisterTfShapeCpuOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf.Shape", TFRT_CPU_OP(TfShapeOp),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tf.ExpandDims", TFRT_CPU_OP(TfExpandDimsOp),
                     CpuOpFlags::NoSideEffects);
}

}  // namespace tfrt
