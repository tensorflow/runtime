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

// Tensorflow Concat kernel implementation.

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CONCAT_KERNEL_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CONCAT_KERNEL_H_

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/ranges.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace cpu {

inline Expected<int64_t> ConcatAxis(const DenseHostTensor& axis_arg) {
  if (axis_arg.dtype() == DType::I32) {
    DHTArrayView<int32_t> view(&axis_arg);
    return *view.begin();
  } else if (axis_arg.dtype() == DType::I64) {
    DHTArrayView<int64_t> view(&axis_arg);
    return *view.begin();
  } else {
    return MakeStringError("Unsupported axis data type");
  }
}

template <typename TensorRange>
Expected<TensorMetadata> ConcatMetadataKernel(const TensorRange& args,
                                              int axis) {
  const Tensor& arg0 = args[0];
  const DType& arg0_dtype = arg0.dtype();
  const TensorShape& arg0_shape = arg0.shape();

  const int rank = arg0_shape.GetRank();
  // Compute the actual axis from a negative value.
  axis = axis < 0 ? axis + rank : axis;

  // The size of a result along the concatenation dimension.
  ssize_t concat_axis_dim_size = 0;

  for (size_t i = 0; i < args.size(); ++i) {
    const Tensor& arg = args[i];
    const TensorShape& shape = arg.shape();

    // Implicitly convert scalars to vectors of length 1.
    concat_axis_dim_size += rank == 0 ? 1 : shape.GetDimensionSize(axis);

    // Inputs must be of the same rank and data type.
    if (arg.dtype() != arg0_dtype)
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

  return TensorMetadata(arg0_dtype, TensorShape(output_dims));
}

template <int rank>
using ConcatRankTag = std::integral_constant<int, rank>;

template <typename T, typename DHTRange>
Error ConcatKernel(const DHTRange& args, int axis, DenseHostTensor* output) {
  const int input_rank = args[0].shape().GetRank();

  // Handle scalars concatenation separately to keep the common path simple.
  if (input_rank == 0) {
    for (int i = 0; i < args.size(); ++i) {
      const T* inp = static_cast<const T*>(args[i].data());
      T* out = static_cast<T*>(output->data());
      out[i] = *inp;
    }

    return Error::success();
  }

  // Compute the actual axis from a negative value.
  axis = axis < 0 ? axis + input_rank : axis;

  // TODO(ezhulenev): Make this asynchronous/multithreaded.
  auto rank_dispatch = [&](auto rank_tag) -> void {
    static constexpr int rank = decltype(rank_tag)::value;

    auto output_view = MutableDHTIndexableView<T, rank>(output);
    auto output_t = compat::AsEigenTensor(output_view);

    // Offset for writing to the output along the axis dimension.
    ssize_t offset = 0;

    for (auto& dht : args) {
      auto arg_view = DHTIndexableView<T, rank>(&dht);
      auto arg_t = compat::AsEigenConstTensor(arg_view);

      // Offsets for the output slice.
      Eigen::DSizes<ssize_t, rank> offsets;
      for (int d = 0; d < rank; d++) offsets[d] = 0;
      offsets[axis] = offset;

      // Size of the output slice.
      Eigen::DSizes<ssize_t, rank> sizes;
      for (int d = 0; d < rank; d++) sizes[d] = dht.shape().GetDimensionSize(d);

      // It's possible for this input tensor to have zero dimension along the
      // specified axis, in which case no work is needed, b/172595919
      if (sizes[axis] == 0) continue;

      offset += sizes[axis];

      // Write argument into the output tensor slice.
      output_t.slice(offsets, sizes) = arg_t;
    }
  };

  auto rank = output->shape().GetRank();
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
    return MakeStringError("Unsupported output tensor rank");
  }

  return Error::success();
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CONCAT_KERNEL_H_
