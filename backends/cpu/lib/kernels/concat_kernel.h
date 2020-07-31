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

//===- concat_kernel.h ------------------------------------------*- C++ -*-===//
//
// Tensorflow Concat kernel implementation.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CONCAT_KERNEL_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CONCAT_KERNEL_H_

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace cpu {

static Expected<int64_t> ConcatAxis(const DenseHostTensor& axis_arg) {
  int64_t axis;

  if (axis_arg.dtype().kind() == DType::I32) {
    DHTArrayView<int32_t> view(&axis_arg);
    axis = *view.begin();
  } else if (axis_arg.dtype().kind() == DType::I64) {
    DHTArrayView<int64_t> view(&axis_arg);
    axis = *view.begin();
  } else {
    return MakeStringError("Unsupported axis data type");
  }

  return axis;
}

static Expected<TensorMetadata> ConcatTensorMetadata(
    ArrayRef<const DenseHostTensor*> args, int64_t axis) {
  const DenseHostTensor* arg0 = args[0];
  const DType& arg0_dtype = arg0->dtype();
  const TensorShape& arg0_shape = arg0->shape();

  const int rank = arg0->shape().GetRank();

  // The size of a result along the concatenation dimension.
  ssize_t concat_axis_dim_size = 0;

  for (int i = 0; i < args.size(); ++i) {
    const DenseHostTensor* arg = args[i];
    const TensorShape& shape = arg->shape();

    // Implicitly convert scalars to vectors of length 1.
    concat_axis_dim_size += rank == 0 ? 1 : shape.GetDimensionSize(axis);

    // Inputs must be of the same rank and data type.
    if (arg->dtype() != arg0_dtype)
      return MakeStringError("Input dtypes do not match");
    if (shape.GetRank() != rank)
      return MakeStringError("Input ranks do not match");

    // Construct a error message for non-matching dimension.
    auto wrong_dim = [&](int dim) -> Error {
      return MakeStringError(
          StrCat("Wrong non-concatenating dimension. Dimension ", dim,
                 " for argument ", i,
                 " doesn't match the dimension of argument 0 (concatenate ",
                 shape, " and ", arg0_shape, " on axis ", axis, ")"));
    };

    // Dimensions before the concatenation axis must match.
    for (int d = 0; d < axis; ++d) {
      if (shape.GetDimensionSize(d) != arg0_shape.GetDimensionSize(d))
        return wrong_dim(d);
    }

    // Dimensions after the concatenation axis must match.
    for (int d = axis + 1; d < rank; ++d) {
      if (shape.GetDimensionSize(d) != arg0_shape.GetDimensionSize(d))
        return wrong_dim(d);
    }
  }

  // Build output tensor dimensions.
  SmallVector<ssize_t, 4> output_dims;
  for (int d = 0; d < axis; ++d)
    output_dims.push_back(arg0_shape.GetDimensionSize(d));
  output_dims.push_back(concat_axis_dim_size);
  for (int d = axis + 1; d < rank; ++d)
    output_dims.push_back(arg0_shape.GetDimensionSize(d));

  return TensorMetadata(arg0->dtype(), TensorShape(output_dims));
}

template <int rank>
struct ConcatRankTag {
  static constexpr int value = rank;
};

template <typename T>
static AsyncValueRef<Chain> ConcatKernel(ArrayRef<const DenseHostTensor*> args,
                                         int64_t axis, DenseHostTensor* output,
                                         const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  // Handle scalars concatenation separately to keep the common path simple.
  if (args[0]->shape().GetRank() == 0) {
    for (int i = 0; i < args.size(); ++i) {
      const T* inp = static_cast<const T*>(args[i]->data());
      T* out = static_cast<T*>(output->data());
      out[i] = *inp;
    }

    return host->MakeAvailableAsyncValueRef<Chain>();
  }

  // TODO(ezhulenev): Make this asynchronous/multithreaded.
  auto rank_dispatch = [&](auto rank_tag) -> void {
    static constexpr int rank = decltype(rank_tag)::value;

    auto output_view = MutableDHTIndexableView<T, rank>(output);
    auto output_t = compat::AsEigenTensor(output_view);

    // Offset for writing to the output along the axis dimension.
    ssize_t offset = 0;

    for (int i = 0; i < args.size(); ++i) {
      auto arg_view = DHTIndexableView<T, rank>(args[i]);
      auto arg_t = compat::AsEigenConstTensor(arg_view);

      // Offsets for the output slice.
      Eigen::DSizes<ssize_t, rank> offsets;
      for (int d = 0; d < rank; d++) offsets[d] = 0;
      offsets[axis] = offset;

      // Size of the output slice.
      Eigen::DSizes<ssize_t, rank> sizes;
      for (int d = 0; d < rank; d++)
        sizes[d] = args[i]->shape().GetDimensionSize(d);

      offset += sizes[axis];

      // Write argument into the output tensor slice.
      output_t.slice(offsets, sizes) = arg_t;
    }
  };

  const int rank = output->shape().GetRank();

  // Dispatch based on the output tensor rank.
  if (rank == 1) {
    rank_dispatch(ConcatRankTag<1>{});
  } else if (rank == 2) {
    rank_dispatch(ConcatRankTag<2>{});
  } else if (rank == 3) {
    rank_dispatch(ConcatRankTag<3>{});
  } else if (rank == 4) {
    rank_dispatch(ConcatRankTag<4>{});
  } else if (rank == 5) {
    rank_dispatch(ConcatRankTag<4>{});
  } else {
    return host->MakeErrorAsyncValueRef("Unsupported output tensor rank");
  }

  return host->MakeAvailableAsyncValueRef<Chain>();
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CONCAT_KERNEL_H_
