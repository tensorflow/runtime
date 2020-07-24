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

//===- transpose_op.cu.cc ---------------------------------------*- C++ -*-===//
//
// Implementations of TF Transpose op.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Error.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"
#include "tfrt/gpu/stream/cudart_wrapper.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/tensor_serialize_utils.h"
#include "transpose_op.h"

namespace tfrt {
namespace gpu {

namespace {

static constexpr int kTileSize = 16;

// Transpose 2D matrix coalescing memory access patterns.
template <typename T>
__global__ void Transpose2DCoalesced(T* out, T* in, int width, int height) {
  // Add 1 to inner dimension to avoid bank conflicts.
  __shared__ T tile[kTileSize][kTileSize + 1];

  // Load one element per thread into the shared tile memory.
  int x_index = blockIdx.x * kTileSize + threadIdx.x;
  int y_index = blockIdx.y * kTileSize + threadIdx.y;

  if ((x_index < width) && (y_index < height)) {
    int input_idx = y_index * width + x_index;
    tile[threadIdx.y][threadIdx.x] = in[input_idx];
  }

  // Wait for data in tile to be ready.
  __syncthreads();

  // Write data from shared memory tile to the output memory.
  x_index = blockIdx.y * kTileSize + threadIdx.x;
  y_index = blockIdx.x * kTileSize + threadIdx.y;

  if ((x_index < height) && (y_index < width)) {
    int output_idx = y_index * height + x_index;
    out[output_idx] = tile[threadIdx.x][threadIdx.y];
  }
}

// Swaps dimensions 1 and 2 in tensor of rank 3.
template <typename T>
__global__ void SwapDims1And2In3(T* out, T* in, int planes, int width,
                                 int height) {
  // Add 1 to inner dimension to avoid bank conflicts.
  __shared__ T tile[kTileSize][kTileSize + 1];

  const int stride = width * height;

  for (int plane = 0; plane < planes; ++plane) {
    int offset = plane * stride;

    // Wait for the completion of all writes frome the previous loop iteration.
    if (plane > 0) __syncthreads();

    // Load one element per thread into the shared tile memory.
    int x_index = blockIdx.x * kTileSize + threadIdx.x;
    int y_index = blockIdx.y * kTileSize + threadIdx.y;

    if ((x_index < width) && (y_index < height)) {
      int input_idx = offset + y_index * width + x_index;
      tile[threadIdx.y][threadIdx.x] = in[input_idx];
    }

    // Wait for data in tile to be ready.
    __syncthreads();

    // Write data from shared memory tile to the output memory.
    x_index = blockIdx.y * kTileSize + threadIdx.x;
    y_index = blockIdx.x * kTileSize + threadIdx.y;

    if ((x_index < height) && (y_index < width)) {
      int output_idx = offset + y_index * height + x_index;
      out[output_idx] = tile[threadIdx.x][threadIdx.y];
    }
  }
}

unsigned NumBlocks(ssize_t size, unsigned threads_per_block) {
  return static_cast<unsigned>((size + threads_per_block - 1) /
                               threads_per_block);
}

struct CoalescedDimsAndPerm {
  SmallVector<ssize_t, 8> dims;
  SmallVector<ssize_t, 8> perm;
};

// Helper function that takes a tensor shape, a permutation, combines the
// neighboring shapes if their indices in the permutation are consecutive.
// The function outputs the combined shape and new permutation.
// Example: Tensor shape {2, 3, 4, 5, 120} and permutation {0, 4, 1, 2, 3} will
// produce new shape {2, 60, 120} and new permutation {0, 2, 1}.
CoalescedDimsAndPerm CoalesceTranspose(const TensorShape& shape,
                                       ArrayRef<ssize_t> perm) {
  assert(shape.GetRank() == perm.size());

  if (shape.GetRank() == 1) return {{shape.GetDimensionSize(0)}, {perm[0]}};

  SmallVector<ssize_t, 8> new_dim_position(shape.GetRank(), -1);
  SmallVector<ssize_t, 8> combined_dims(shape.GetRank(), 0);

  int cur_head = perm[0];
  new_dim_position[cur_head] = 0;
  combined_dims[0] = shape.GetDimensionSize(cur_head);

  int dim_idx = 0;
  for (int perm_idx = 1; perm_idx < shape.GetRank(); ++perm_idx) {
    // If two indices in permutation are consecutive numbers, combine their
    // dimensions.
    if (cur_head + 1 == perm[perm_idx]) {
      cur_head = perm[perm_idx];
      combined_dims[dim_idx] *= shape.GetDimensionSize(cur_head);
    } else {
      // Else start a new dimension.
      cur_head = perm[perm_idx];
      dim_idx++;
      new_dim_position[cur_head] = dim_idx;
      combined_dims[dim_idx] = shape.GetDimensionSize(cur_head);
    }
  }

  // Compact the new permutations and dimension sizes.
  CoalescedDimsAndPerm merged;
  merged.dims.resize(dim_idx + 1);
  merged.perm.resize(dim_idx + 1);
  dim_idx = 0;
  for (int i = 0; i < new_dim_position.size(); ++i) {
    if (new_dim_position[i] >= 0) {
      int new_perm_idx = new_dim_position[i];
      merged.perm[dim_idx] = new_perm_idx;
      merged.dims[dim_idx] = combined_dims[new_perm_idx];
      dim_idx++;
    }
  }

  return merged;
}

llvm::Error DispatchTrivialTranspose(GpuDispatchContext* dctx, DType dtype,
                                     const GpuBuffer& input,
                                     const GpuBuffer& output,
                                     const CoalescedDimsAndPerm& transpose) {
  assert(transpose.dims.size() == 2 && transpose.perm.size() == 2);

  ssize_t height = transpose.dims[0];
  ssize_t width = transpose.dims[1];

  dim3 grid(NumBlocks(width, kTileSize), NumBlocks(height, kTileSize), 1);
  dim3 threads(kTileSize, kTileSize, 1);
  size_t shared_memory_size_bytes = 0;

  auto launch = [&](auto type_tag) {
    using T = decltype(type_tag);
    return stream::CudaLaunchKernel(
        dctx->current_context(), &Transpose2DCoalesced<T>, grid, threads,
        shared_memory_size_bytes, dctx->stream(), GetRawPointer<T>(output),
        GetRawPointer<T>(input), width, height);
  };

  switch (dtype.kind()) {
    default:
      return MakeStringError("Unsupported data type: ", dtype);

    case DType::F16:
      if (auto error = launch(Eigen::half{})) return error;
      break;

    case DType::F32:
      if (auto error = launch(float{})) return error;
      break;

    case DType::I64:
      if (auto error = launch(int64_t{})) return error;
      break;
  }

  return llvm::Error::success();
}

llvm::Error DispatchSwapDims1And2In3(GpuDispatchContext* dctx, DType dtype,
                                     const GpuBuffer& input,
                                     const GpuBuffer& output,

                                     const CoalescedDimsAndPerm& transpose) {
  assert(transpose.dims.size() == 3 && transpose.perm.size() == 3);

  ssize_t planes = transpose.dims[0];
  ssize_t height = transpose.dims[1];
  ssize_t width = transpose.dims[2];

  dim3 grid(NumBlocks(width, kTileSize), NumBlocks(height, kTileSize), 1);
  dim3 threads(kTileSize, kTileSize, 1);
  size_t shared_memory_size_bytes = 0;

  auto launch = [&](auto type_tag) {
    using T = decltype(type_tag);
    return stream::CudaLaunchKernel(
        dctx->current_context(), &SwapDims1And2In3<T>, grid, threads,
        shared_memory_size_bytes, dctx->stream(), GetRawPointer<T>(output),
        GetRawPointer<T>(input), planes, width, height);
  };

  switch (dtype.kind()) {
    default:
      return MakeStringError("Unsupported data type: ", dtype);

    case DType::F16:
      if (auto error = launch(Eigen::half{})) return error;
      break;

    case DType::F32:
      if (auto error = launch(float{})) return error;
      break;

    case DType::I64:
      if (auto error = launch(int64_t{})) return error;
      break;
  }

  return llvm::Error::success();
}

}  // namespace

static llvm::Expected<DenseGpuTensor> ComputeTransposeGpuOpImpl(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    ArrayRef<ssize_t> perm, const TensorMetadata& result_md) {
  size_t num_result_elements = result_md.shape.GetNumElements();
  size_t size_in_bytes = result_md.dtype.GetHostSize() * num_result_elements;

  using Perm = SmallVector<ssize_t, 8>;
  auto transpose = CoalesceTranspose(input.shape(), perm);

  // Transpose: [x, y] -> [y, x]
  bool is_trivial_transpose = (transpose.perm == Perm({1, 0}));
  // Transpose: [z, x, y] -> [z, y, x]
  bool is_swap_1_and_2_in_3 = (transpose.perm == Perm({0, 2, 1}));

  // TODO(ezhulenev): Support other types of permutation.
  if (!is_trivial_transpose && !is_swap_1_and_2_in_3) {
    return MakeStringError("Unsupported tf.Transpose permutation");
  }

  TFRT_ASSIGN_OR_RETURN(auto output_buffer,
                        dctx->allocator()->Allocate(
                            /*size=*/size_in_bytes, dctx->stream()));

  if (is_trivial_transpose) {
    if (auto err = DispatchTrivialTranspose(dctx, input.dtype(), input.buffer(),
                                            *output_buffer, transpose)) {
      return std::move(err);
    }

  } else if (is_swap_1_and_2_in_3) {
    if (auto err = DispatchSwapDims1And2In3(dctx, input.dtype(), input.buffer(),
                                            *output_buffer, transpose)) {
      return std::move(err);
    }
  }

  // TODO(ezhulenev): Don't use CUDA specific API here.
  if (cudaGetLastError() != cudaSuccess) {
    return MakeStringError("CUDA launch error");
  }

  return DenseGpuTensor(result_md.shape, result_md.dtype,
                        std::move(output_buffer));
}

static llvm::Expected<DenseGpuTensor> ComputeTransposeGpuOp(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const DenseGpuTensor& /*perm*/, const TensorMetadata& result_md) {
  // TODO(tfrt-devs): Read perm from the dense host tensor.
  static constexpr ssize_t default_perm[] = {0, 3, 1, 2};

  return ComputeTransposeGpuOpImpl(dctx, input, default_perm, result_md);
}

static llvm::Expected<DenseGpuTensor> ComputeTransposeGpuOpFolded(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const OpAttrsRef& attrs, const TensorMetadata& result_md) {
  DenseAttr perm_attr;
  if (!attrs.Get("perm", &perm_attr)) {
    return MakeStringError("tf.Transpose needs a `perm` dense attribute");
  }

  DenseView perm_view = CreateDenseView(perm_attr);
  assert(perm_view.shape().GetRank() == 1);

  SmallVector<ssize_t, 4> perm;

  switch (perm_view.dtype().kind()) {
    case DType::I32: {
      auto value = perm_view.GetFlat<int32_t>();
      perm.assign(value.begin(), value.end());
      break;
    }
    case DType::I64: {
      auto value = perm_view.GetFlat<int64_t>();
      perm.assign(value.begin(), value.end());
      break;
    }
    default:
      llvm_unreachable("unsupported dtype for perm in tf.Transpose.");
  }

  return ComputeTransposeGpuOpImpl(dctx, input, perm, result_md);
}

}  // namespace gpu

void RegisterTransposeGpuTfOps(GpuOpRegistry* registry) {
  registry->AddOp("tf.Transpose", TFRT_GPU_OP(gpu::ComputeTransposeGpuOp));

  // "_tf.Transpose" is a compiler-optimized version of "tf.Transpose" where
  // the perm argument is folded to an attribute.
  registry->AddOp("_tf.Transpose",
                  TFRT_GPU_OP(gpu::ComputeTransposeGpuOpFolded), {"perm"});
}

}  // namespace tfrt
