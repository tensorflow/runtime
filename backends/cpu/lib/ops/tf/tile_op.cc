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

//===- tile_op.cc - ---------------------------------------------*- C++ -*-===//
//
// Tensorflow Tile operation.
//
//===----------------------------------------------------------------------===//

#include "tile_op.h"

#include <sys/types.h>

#include <cstdint>

#include "../../kernels/tile_kernel.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/string_host_tensor.h"

namespace tfrt {
namespace {

void TileStringTensor(const StringHostTensor& input, StringHostTensor* output) {
  // Compute strides from the shape.
  auto strides = [](const TensorShape& shape) -> SmallVector<ssize_t, 5> {
    SmallVector<ssize_t, 5> strides(shape.GetRank());
    ssize_t stride = 1;
    for (int i = shape.GetRank() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape.GetDimensionSize(i);
    }
    return strides;
  };

  const int ndims = output->shape().GetRank();
  const ssize_t nelem = output->NumElements();

  auto in_strides = strides(input.shape());
  auto out_strides = strides(output->shape());

  ArrayRef<std::string> inp = input.strings();
  MutableArrayRef<std::string> out = output->strings();

  for (ssize_t o_idx = 0; o_idx < nelem; ++o_idx) {
    ssize_t i_idx = 0;
    ssize_t t = o_idx;
    for (int i = 0; i < ndims; ++i) {
      ssize_t i_dim = input.shape().GetDimensionSize(i);
      i_idx += t / out_strides[i] % i_dim * in_strides[i];
      t %= out_strides[i];
    }
    out[o_idx] = inp[i_idx];
  }
}

static AsyncValueRef<HostTensor> TfTileOp(const HostTensor& input_arg,
                                          const DenseHostTensor& multiples_arg,
                                          const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  // Parse multiples tensor.
  Expected<SmallVector<ssize_t, 5>> expected_multiples =
      cpu::TileMultiples(multiples_arg);
  if (auto err = expected_multiples.takeError())
    return EmitErrorAsync(exec_ctx, std::move(err));

  // Check that multiples are valid.
  const TensorShape& input_shape = input_arg.shape();
  if (expected_multiples->size() != input_shape.GetRank()) {
    return EmitErrorAsync(
        exec_ctx, "Tile multiples must have the same size as input rank");
  }

  // Compute the output shape from the input shape and multiples.
  SmallVector<ssize_t, 5> output_dims;
  for (int d = 0; d < expected_multiples->size(); ++d) {
    output_dims.push_back(input_shape.GetDimensionSize(d) *
                          (*expected_multiples)[d]);
  }

  TensorShape output_shape(output_dims);
  TensorMetadata output_md(input_arg.dtype(), output_shape);

  if (isa<DenseHostTensor>(input_arg)) {
    const DenseHostTensor& input = cast<DenseHostTensor>(input_arg);

    // Allocate output tensor.
    auto dest = DenseHostTensor::CreateUninitialized(output_md, host);
    if (!dest) {
      return EmitErrorAsync(exec_ctx, "out of memory allocating result");
    }

    // Call tile kernel.
    AsyncValueRef<Chain> chain;

    switch (input.dtype().kind()) {
      default:
        chain = EmitErrorAsync(exec_ctx,
                               StrCat("Unsupported dtype: ", input.dtype()));
        break;
#define DTYPE_NUMERIC(ENUM)                                    \
  case DType::ENUM: {                                          \
    using T = EigenTypeForDTypeKind<DType::ENUM>;              \
    chain = ::tfrt::cpu::Tile<T>(input, *expected_multiples,   \
                                 dest.getPointer(), exec_ctx); \
  } break;
#include "tfrt/dtype/dtype.def"  // NOLINT
    }

    return ForwardValue(dest.getValue(), std::move(chain), host);

  } else if (isa<StringHostTensor>(input_arg)) {
    const StringHostTensor& input = cast<StringHostTensor>(input_arg);

    auto dest = StringHostTensor::CreateUninitialized(output_md, host);
    if (!dest) {
      return EmitErrorAsync(exec_ctx, "out of memory allocating result");
    }

    TileStringTensor(input, dest.getPointer());

    return host->MakeAvailableAsyncValueRef<StringHostTensor>(std::move(*dest));

  } else {
    return EmitErrorAsync(exec_ctx, "Unsupported tensor type");
  }
}

}  // namespace

void RegisterTfTileCpuOp(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf.Tile", TFRT_CPU_OP(TfTileOp),
                     CpuOpFlags::NoSideEffects | CpuOpFlags::AllowsString);
}

}  // namespace tfrt
