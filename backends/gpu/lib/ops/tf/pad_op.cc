// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- pad_op.cc - Implements tf.pad on GPU --------------------*- C++ -*--===//
//
// Implements tf.Pad on GPU.
//
//===----------------------------------------------------------------------===//
#include "pad_op.h"

#define EIGEN_USE_GPU

#include <iostream>  // some eigen header use std::cerr without including it.

#include "../../device/eigen_support.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/common/ops/tf/dnn_ops_util.h"
#include "tfrt/core_runtime/op_attr_type.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/gpu/blas_support.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"
#include "tfrt/gpu/memory/gpu_buffer.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/tensor_serialize_utils.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h"  // from @eigen_archive

namespace tfrt {
namespace gpu {

template <typename T, typename Tpadding, int Rank>
static void Operate(GpuDispatchContext* dctx,
                    compat::EigenConstTensor<T, Rank> input,
                    compat::EigenConstMatrix<Tpadding> paddings, T pad_value,
                    compat::EigenTensor<T, Rank> output) {
  Eigen::array<Eigen::IndexPair<Tpadding>, Rank> paddings_array;
  for (int i = 0; i < Rank; ++i) {
    paddings_array[i] = {paddings(i, 0), paddings(i, 1)};
  }
  functor::Pad<Eigen::GpuDevice, T, Tpadding, Rank> functor;
  functor(*dctx->eigen_gpu_device(), output, input, paddings_array, pad_value);
}

static void CallOperate(GpuDispatchContext* dctx, const DenseGpuTensor& input,
                        const DenseView& paddings, DenseGpuTensor* result) {
#define OPERATE(T, Tpadding, Rank)                                   \
  Operate<T, Tpadding, Rank>(                                        \
      dctx, AsEigenConstTensor<T, Rank>(input),                      \
      compat::AsEigenConstTensor(paddings.GetTensor<Tpadding, 2>()), \
      static_cast<T>(0), AsEigenTensor<T, Rank>(result))

#define SWITCH_ON_RANK(T, Tpadding)                                         \
  switch (input.shape().GetRank()) {                                        \
    case 0:                                                                 \
      return OPERATE(T, Tpadding, 0);                                       \
    case 1:                                                                 \
      return OPERATE(T, Tpadding, 1);                                       \
    case 2:                                                                 \
      return OPERATE(T, Tpadding, 2);                                       \
    case 3:                                                                 \
      return OPERATE(T, Tpadding, 3);                                       \
    case 4:                                                                 \
      return OPERATE(T, Tpadding, 4);                                       \
    case 5:                                                                 \
      return OPERATE(T, Tpadding, 5);                                       \
    case 6:                                                                 \
      return OPERATE(T, Tpadding, 6);                                       \
    case 7:                                                                 \
      return OPERATE(T, Tpadding, 7);                                       \
    case 8:                                                                 \
      return OPERATE(T, Tpadding, 8);                                       \
    default:                                                                \
      llvm_unreachable(                                                     \
          "Tensors with more than 8 dimentions should have been caught by " \
          "metadata function");                                             \
  }

#define SWITCH_ON_TPADDING(T)                                                \
  switch (paddings.dtype().kind()) {                                         \
    case DType::I32:                                                         \
      SWITCH_ON_RANK(T, int32_t);                                            \
    case DType::I64:                                                         \
      SWITCH_ON_RANK(T, int64_t);                                            \
    default:                                                                 \
      llvm_unreachable(                                                      \
          "Padding dtype other than int32/int64 should have been caught by " \
          "the metadata function");                                          \
  }

  // TODO(iga): Support I16, and *I8
#define DTYPE_TRIVIAL(ENUM) SWITCH_ON_TPADDING(TypeForDTypeKind<DType::ENUM>)

  switch (input.dtype().kind()) {
    case DType::I32:
      DTYPE_TRIVIAL(I32);
    case DType::I64:
      DTYPE_TRIVIAL(I64);
    case DType::F16:
      DTYPE_TRIVIAL(F16);
    case DType::F32:
      DTYPE_TRIVIAL(F32);
    case DType::F64:
      DTYPE_TRIVIAL(F64);
    default:
      llvm_unreachable("Unsupported type in tf.Pad");
  }

#undef DTYPE_TRIVIAL
#undef SWITCH_ON_TPADDING
#undef SWITCH_ON_RANK
#undef OPERATE
}

llvm::Expected<DenseGpuTensor> EnqueueGpuPadOp(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const DenseView& paddings, const TensorMetadata& result_md) {
  size_t size_in_bytes = result_md.GetHostSizeInBytes();
  TFRT_ASSIGN_OR_RETURN(RCReference<GpuBuffer> buffer,
                        dctx->allocator()->Allocate(
                            /*size=*/size_in_bytes, dctx->stream()));

  if (size_in_bytes == 0) {
    return DenseGpuTensor(result_md.shape, result_md.dtype, std::move(buffer));
  }

  DenseGpuTensor result(result_md.shape, result_md.dtype, std::move(buffer));

  CallOperate(dctx, input, paddings, &result);
  return result;
}

static llvm::Expected<DenseGpuTensor> GpuPadOp(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const DenseGpuTensor& /* paddings input is ignored */,
    const TensorMetadata& result_md) {
  // TODO(tfrt-devs): Read paddings from dense host tensor.
  llvm::SmallVector<int32_t, 8> default_paddings_data(8, 0);
  auto channel_order = GuessChannelOrder(input.shape());
  if (!channel_order) return MakeStringError("Could not guess channel order.");
  auto spatial_offset = *channel_order == ChannelOrder::ChannelLast ? 2 : 4;
  std::fill_n(default_paddings_data.begin() + spatial_offset, 4, 3);
  DenseView default_paddings(GetDType<int32_t>(), {4, 2},
                             default_paddings_data.data());

  return EnqueueGpuPadOp(dctx, input, default_paddings, result_md);
}

static llvm::Expected<DenseGpuTensor> GpuPadFoldedOp(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const OpAttrsRef& attrs, const TensorMetadata& result_md) {
  DenseAttr dense_attr;
  if (!attrs.Get("paddings", &dense_attr)) {
    return MakeStringError("_tf.Pad needs a `paddings` dense attribute");
  }

  DenseView paddings = CreateDenseView(dense_attr);

  return EnqueueGpuPadOp(dctx, input, paddings, result_md);
}

}  // namespace gpu

void RegisterPadGpuTfOps(GpuOpRegistry* registry) {
  registry->AddOp("tf.Pad", TFRT_GPU_OP(gpu::GpuPadOp));

  // "_tf.Pad" is a compiler-optimized version of "tf.Pad", where the paddings
  // argument is folded to a dense attribute.
  registry->AddOp("_tf.Pad", TFRT_GPU_OP(gpu::GpuPadFoldedOp), {"paddings"});
}

}  // namespace tfrt
