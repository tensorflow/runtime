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

// Tensorflow Concat operations.

#include "concat_op.h"

#include <cstdint>

#include "../../kernels/concat_kernel.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/string_host_tensor.h"

namespace tfrt {
namespace {

static Expected<DenseHostTensor> TfConcatOpDense(
    RepeatedArguments<DenseHostTensor> args, int64_t axis,
    const TensorMetadata& output_md, const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  // Allocate output tensor.
  auto dest = DenseHostTensor::CreateUninitialized(output_md, host);
  if (!dest) {
    return MakeStringError("out of memory allocating result");
  }

  // Call concat kernel.
  switch (args[0].dtype()) {
    default:
      return MakeStringError("Unsupported dtype: ", args[0].dtype());
      break;
    case DType::I1: {
      using T = EigenTypeForDTypeKind<DType::I1>;
      auto error = ::tfrt::cpu::ConcatKernel<T>(args, axis, dest.getPointer());
      if (error) return std::move(error);
      break;
    }
#define DTYPE_NUMERIC(ENUM)                                                   \
  case DType::ENUM: {                                                         \
    using T = EigenTypeForDTypeKind<DType::ENUM>;                             \
    auto error = ::tfrt::cpu::ConcatKernel<T>(args, axis, dest.getPointer()); \
    if (error) return std::move(error);                                       \
    break;                                                                    \
  }
#include "tfrt/dtype/dtype.def"  // NOLINT
  }

  return std::move(*dest);
}

// TODO(tfrt-devs): The implemention below for the string type is synchronous.
// Consider making it multi-threaded if needed.
// TODO(tfrt-devs): Consider merging TfConcatOpString and TfConcatOpDense.
static Expected<StringHostTensor> TfConcatOpString(
    RepeatedArguments<StringHostTensor> args, int64_t axis,
    const TensorMetadata& output_md, const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  auto dest = StringHostTensor::CreateUninitialized(output_md, host);
  if (!dest) {
    return MakeStringError(
        "TfConcatOp: failed to allocate StringHostTensor result");
  }

  auto compute_flat_dim = [](llvm::ArrayRef<Index> dims, int begin, int end) {
    assert(begin >= 0);
    assert(end <= dims.size());
    assert(begin <= end);
    int64_t result = 1;
    for (; begin < end; ++begin) {
      assert(dims[begin] >= 0);
      result *= dims[begin];
    }
    return result;
  };

  llvm::SmallVector<Index, 4> output_dims;
  output_md.shape.GetDimensions(&output_dims);

  // The following logic basically reshape the output and args to rank-2 tensors
  // according to axis, and then copy over all elements in the inner dimension
  // of args to inner dimension of the output.

  int64_t outer_dim = compute_flat_dim(output_dims, 0, axis);
  int64_t inner_dim = compute_flat_dim(output_dims, axis, output_dims.size());

  int64_t inner_offset = 0;
  for (const auto& arg : args) {
    llvm::SmallVector<Index, 4> arg_dims;
    arg.shape().GetDimensions(&arg_dims);
    int64_t arg_inner_dim = compute_flat_dim(arg_dims, axis, arg_dims.size());
    assert(inner_offset + arg_inner_dim <= inner_dim);
    assert(outer_dim == compute_flat_dim(arg_dims, 0, axis));

    int64_t outer_offset = 0;
    int64_t arg_outer_offset = 0;
    for (int i = 0; i < outer_dim; ++i) {
      std::copy(arg.strings().begin() + arg_outer_offset,
                arg.strings().begin() + arg_outer_offset + arg_inner_dim,
                dest->strings().begin() + outer_offset + inner_offset);
      outer_offset += inner_dim;
      arg_outer_offset += arg_inner_dim;
    }

    inner_offset += arg_inner_dim;
  }

  return std::move(*dest);
}

static AsyncValueRef<HostTensor> TfConcatOp(
    RepeatedArguments<HostTensor> inputs, const ExecutionContext& exec_ctx) {
  auto* axis_tensor =
      llvm::dyn_cast<const DenseHostTensor>(&inputs[inputs.size() - 1]);
  if (!axis_tensor)
    return EmitErrorAsync(exec_ctx, "axis must be a dense host tensor");

  auto axis = cpu::ConcatAxis(*axis_tensor);
  if (!axis) return EmitErrorAsync(exec_ctx, axis.takeError());

  // The last tensor in inputs is the axis. Make a range view to cover all but
  // the last tensor input as args.
  auto args = views::Counted(inputs.begin(), inputs.size() - 1);

  auto output_md = cpu::ConcatMetadataKernel(args, *axis);
  if (!output_md) return EmitErrorAsync(exec_ctx, output_md.takeError());

  auto make_async = [&](auto expected_tensor) -> AsyncValueRef<HostTensor> {
    using TensorType = typename decltype(expected_tensor)::value_type;

    if (!expected_tensor) {
      return EmitErrorAsync(exec_ctx, expected_tensor.takeError());
    }

    return MakeAvailableAsyncValueRef<TensorType>(
        exec_ctx.host(), std::move(expected_tensor.get()));
  };

  if (llvm::isa<StringHostTensor>(&args[0])) {
    RepeatedArguments<StringHostTensor> string_args(
        inputs.values().drop_back());
    return make_async(
        TfConcatOpString(string_args, *axis, *output_md, exec_ctx));
  }

  if (llvm::isa<DenseHostTensor>(&args[0])) {
    RepeatedArguments<DenseHostTensor> dense_args(inputs.values().drop_back());
    return make_async(TfConcatOpDense(dense_args, *axis, *output_md, exec_ctx));
  }

  return EmitErrorAsync(
      exec_ctx, StrCat("TfConcatOp: Unsupported dtype: ", args[0].dtype()));
}

}  // namespace

void RegisterTfConcatCpuOp(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf.ConcatV2", TFRT_CPU_OP(TfConcatOp),
                     CpuOpFlags::NoSideEffects | CpuOpFlags::AllowsString);
}

}  // namespace tfrt
