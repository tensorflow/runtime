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

// Implementations of TF reduction ops.

#include <cstddef>
#include <iterator>

#include "cub/device/device_reduce.cuh"               // from @cub_archive
#include "cub/device/device_segmented_reduce.cuh"     // from @cub_archive
#include "cub/iterator/counting_input_iterator.cuh"   // from @cub_archive
#include "cub/iterator/transform_input_iterator.cuh"  // from @cub_archive
#include "llvm/Support/Error.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/common/ops/tf/dnn_ops_util.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/gpu/wrapper/cudart_wrapper.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {
namespace gpu {
namespace {

//===----------------------------------------------------------------------===//
// Reduction CUDA kernels.
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): For ROCm it must be 64. See TF_RED_WARPSIZE in Tensorflow.
static constexpr int kWarpSize = 32;

template <typename T, typename ReduceOp, typename TransformOp>
__global__ void OuterReductionKernel(const T* input, T* output,
                                     int outer_dim_size, int inner_dim_size,
                                     T init, ReduceOp reduce,
                                     TransformOp transform) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= inner_dim_size) return;

  T sum = init;

  // Iterate over outer dimension.
  for (int outer_idx = 0; outer_idx < outer_dim_size; ++outer_idx) {
    sum = reduce(sum, input[outer_idx * inner_dim_size + gid]);
  }

  output[gid] = transform(sum);
}

// Reduces the columns of batch of matrices.
//
// Run with a [inner_dim_size/32, outer_dim_size] blocks of [1024] threads.
template <typename T, typename ReduceOp, typename TransformOp>
__global__ void __launch_bounds__(1024)
    MiddleReductionKernel(const T* input, T* output, int reduction_dim_size,
                          int inner_dim_size, T init, ReduceOp reduce,
                          TransformOp transform) {
  int thread_idx = threadIdx.x;
  int lane_idx = thread_idx & 31;
  int warp_idx = thread_idx / 32;
  int inner_idx = (blockIdx.x * 32) + lane_idx;
  int reduction_idx = reduction_dim_size * blockIdx.y + warp_idx;

  input += inner_idx + inner_dim_size * reduction_idx;
  T sum = init;
  for (int i = warp_idx; inner_idx < inner_dim_size && i < reduction_dim_size;
       i += 32) {
    sum = reduce(sum, *reinterpret_cast<const T* __restrict__>(input));
    input += inner_dim_size * 32;
  }
  __shared__ T buffer[32 * 33];
  // write_idx = lane_idx * 33 + warp_idx
  int write_idx = (lane_idx * 32) + lane_idx + warp_idx;
  buffer[write_idx] = sum;
  __syncthreads();
  // read_idx = warp_idx * 33 + lane_idx
  int read_idx = thread_idx + warp_idx;
  T value = buffer[read_idx];

  for (int offset = 1; offset < 32; offset += offset) {
    value = reduce(value, static_cast<T>(__shfl_down_sync(~0u, value, offset)));
  }
  if (lane_idx == 0) {
    int inner_idx = (blockIdx.x * 32) + warp_idx;
    if (inner_idx < inner_dim_size) {
      output += inner_dim_size * blockIdx.y;
      output[inner_idx] = transform(value);
    }
  }
}

// Inner reduction kernel that maps a warp to each row.
template <typename T, typename ReduceOp, typename TransformOp>
__global__ void InnerReductionKernel(const T* input, T* output,
                                     int outer_dim_size, int inner_dim_size,
                                     T init, ReduceOp reduce,
                                     TransformOp transform) {
  assert(blockDim.x % kWarpSize == 0);

  // If inner dimension size is 1 there is nothing to reduce.
  if (inner_dim_size == 1) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < outer_dim_size) output[gid] = input[gid];
    return;
  }

  // Each warp will do an inner dimension reduction.
  const int warps_per_block = blockDim.x / kWarpSize;
  const int warp_index = threadIdx.x / kWarpSize;

  // Offset within a warp.
  const int lane = threadIdx.x % kWarpSize;

  // Index along the outer dimension (aka row).
  const int row = blockIdx.x * warps_per_block + warp_index;
  if (row >= outer_dim_size) return;

  // Index along the inner dimension (aka col).
  int col = lane;

  auto sum = init;

  // Compute partial reduction over inner dimension.
  for (; col < inner_dim_size; col += kWarpSize) {
    sum = reduce(sum, input[row * inner_dim_size + col]);
  }

  // Reduce partial results using warp reduce.
  using WarpReduce = cub::WarpReduce<T>;
  __shared__ typename WarpReduce::TempStorage tmp_storage;

  sum = WarpReduce(tmp_storage)
            .Reduce(sum, reduce, min(inner_dim_size, kWarpSize));
  if (lane == 0) output[row] = transform(sum);
}

//===----------------------------------------------------------------------===//
// Host code for launching reduction kernels.
//===----------------------------------------------------------------------===//

template <typename T>
struct Multiplies {
  __host__ __device__ explicit Multiplies(T multiplier)
      : multiplier(multiplier) {}
  __host__ __device__ T operator()(const T& x) const { return x * multiplier; }
  T multiplier;
};

// Minimalistic output iterator adaptor for CUB reductions. Transforms values
// before assigning to pointee.
template <typename Iterator, typename TransformOp>
class TransformOutputIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = typename std::iterator_traits<Iterator>::value_type;

  // These types should be defined for portability.
  using difference_type = void;
  using pointer = void;
  using reference = void;

  __host__ __device__ TransformOutputIterator(Iterator iterator,
                                              TransformOp transform)
      : iterator_(iterator), transform_(transform) {}

  __device__ const TransformOutputIterator& operator*() const { return *this; }

  __device__ TransformOutputIterator operator[](ptrdiff_t diff) const {
    return {iterator_ + diff, transform_};
  }

  template <typename T>
  __device__ const TransformOutputIterator& operator=(const T& value) const {
    *iterator_ = static_cast<value_type>(transform_(value));
    return *this;
  }

 private:
  Iterator iterator_;
  TransformOp transform_;
};
}  // namespace

template <typename T, typename ReduceOp, typename TransformOp>
static llvm::Error FullReduction(GpuDispatchContext* dctx, const T* input,
                                 T* output, int in_size, T init,
                                 ReduceOp reduce, TransformOp transform) {
  size_t temp_storage_bytes = 0;
  TransformOutputIterator<T*, TransformOp> output_iter(output, transform);

  auto launch = [&](void* temp_storage_ptr) -> llvm::Error {
    auto result = cub::DeviceReduce::Reduce(
        temp_storage_ptr, temp_storage_bytes, input, output_iter, in_size,
        reduce, init, dctx->stream());
    if (result != cudaSuccess)
      return wrapper::MakeError(result, "cub::DeviceReduce::Reduce");
    return llvm::Error::success();
  };

  // Get required amount of temp storage.
  if (auto err = launch(nullptr)) return std::move(err);

  TFRT_ASSIGN_OR_RETURN(
      GpuBuffer tmp_buffer,
      GpuBuffer::Allocate(dctx->allocator(),
                          /*size=*/temp_storage_bytes, dctx->stream()));

  // Do reduction.
  return launch(GetRawPointer<void>(tmp_buffer));
}

static Index NumBlocks(Index num_elements, Index elements_per_block) {
  return (num_elements + elements_per_block - 1) / elements_per_block;
}

template <typename T, typename ReduceOp, typename TransformOp>
static llvm::Error OuterReduction(GpuDispatchContext* dctx, const T* input,
                                  T* output, Index outer_dim_size,
                                  Index inner_dim_size, T init, ReduceOp reduce,
                                  TransformOp transform) {
  Index threads_per_block = 128;
  Index num_blocks = NumBlocks(inner_dim_size, threads_per_block);

  return wrapper::CudaLaunchKernel(
      dctx->current_context(), &OuterReductionKernel<T, ReduceOp, TransformOp>,
      num_blocks, threads_per_block, 0, dctx->stream(), input, output,
      outer_dim_size, inner_dim_size, init, reduce, transform);
}

template <typename T, typename ReduceOp, typename TransformOp>
static llvm::Error InnerReduction(GpuDispatchContext* dctx, const T* input,
                                  T* output, Index outer_dim_size,
                                  Index inner_dim_size, T init, ReduceOp reduce,
                                  TransformOp transform) {
  // For small inner dimension it's more efficient to compute reduction with a
  // custom CUDA kernel that does row-per-warp reduction.
  // TODO(ezhulenev): Benchmark if it's still true with latest CUB and GPUs.
  if (inner_dim_size < 1024) {
    const int threads_per_block = 128;
    const int warps_per_block = threads_per_block / kWarpSize;
    const int num_blocks = NumBlocks(outer_dim_size, warps_per_block);

    return wrapper::CudaLaunchKernel(
        dctx->current_context(),
        &InnerReductionKernel<T, ReduceOp, TransformOp>, num_blocks,
        threads_per_block, 0, dctx->stream(), input, output, outer_dim_size,
        inner_dim_size, init, reduce, transform);
  }

  size_t temp_storage_bytes = 0;

  // Setup segment offsets with counting and transform iterator.
  Multiplies<int> striding(static_cast<int>(inner_dim_size));

  cub::CountingInputIterator<int> counting_iter(0);
  cub::TransformInputIterator<int, Multiplies<int>,
                              cub::CountingInputIterator<int>>
      offset_iter(counting_iter, striding);
  TransformOutputIterator<T*, TransformOp> output_iter(output, transform);

  auto launch = [&](void* temp_storage_ptr) -> llvm::Error {
    auto result = cub::DeviceSegmentedReduce::Reduce(
        temp_storage_ptr, temp_storage_bytes, input, output_iter,
        /*num_segments=*/outer_dim_size, offset_iter, offset_iter + 1, reduce,
        init, dctx->stream());
    if (result != cudaSuccess)
      return wrapper::MakeError(result, "cub::DeviceSegmentedReduce::Reduce");
    return llvm::Error::success();
  };

  // Get required amount of temp storage.
  if (auto err = launch(nullptr)) return std::move(err);

  TFRT_ASSIGN_OR_RETURN(
      GpuBuffer tmp_buffer,
      GpuBuffer::Allocate(dctx->allocator(),
                          /*size=*/sizeof(uint8_t), dctx->stream()));

  // Do reduction.
  return launch(GetRawPointer<void>(tmp_buffer));
}

template <typename T, typename ReduceOp, typename TransformOp>
static llvm::Error MiddleDimReduction(GpuDispatchContext* dctx, const T* input,
                                      T* output, Index outer_dim_size,
                                      Index middle_dim_size,
                                      Index inner_dim_size, T init,
                                      ReduceOp reduce, TransformOp transform) {
  auto grid_width = static_cast<unsigned>(NumBlocks(inner_dim_size, 32));
  dim3 grid_dim = {grid_width, static_cast<unsigned>(outer_dim_size), 1};
  dim3 block_dim = {1024, 1, 1};
  size_t shared_memory_size_bytes = 0;
  auto stream = static_cast<CUstream>(dctx->stream());
  return wrapper::CudaLaunchKernel(
      dctx->current_context(), MiddleReductionKernel<T, ReduceOp, TransformOp>,
      grid_dim, block_dim, shared_memory_size_bytes, stream, input, output,
      middle_dim_size, inner_dim_size, init, reduce, transform);
}

//===----------------------------------------------------------------------===//
// Dispatch to optimal reduction primitive based on the input shape and
// reduction indices.
//===----------------------------------------------------------------------===//

using ReductionDims2 = struct {
  Index outer_dim_size;
  Index inner_dim_size;
};
using ReductionDims3 = struct {
  Index outer_dim_size;
  Index middle_dim_size;
  Index inner_dim_size;
};

static llvm::Optional<ReductionDims2> IsOuterReduction(
    const TensorShape& shape, ArrayRef<int32_t> reduction_indices) {
  // View input tensor as a 2d tensor: [outer_dims_size, inner_dims_size].
  Index outer_dims_size = 1;
  Index inner_dims_size = 1;

  // Check that reduction indices are indeed outer reduction.
  for (int i = 0; i < reduction_indices.size(); ++i) {
    if (reduction_indices[i] != i) return llvm::None;
  }

  for (int i = 0; i < shape.GetRank(); ++i) {
    if (i < reduction_indices.size()) {
      outer_dims_size *= shape.GetDimensionSize(i);
    } else {
      inner_dims_size *= shape.GetDimensionSize(i);
    }
  }

  return {{outer_dims_size, inner_dims_size}};
}

static llvm::Optional<ReductionDims2> IsInnerReduction(
    const TensorShape& shape, ArrayRef<int32_t> reduction_indices) {
  // View input tensor as a 2d tensor: [outer_dims_size, inner_dims_size].
  Index outer_dims_size = 1;
  Index inner_dims_size = 1;

  // Check that reduction indices are indeed inner reduction.
  const int offset = shape.GetRank() - reduction_indices.size();
  for (int i = 0; i < reduction_indices.size(); ++i) {
    if (reduction_indices[i] != offset + i) return llvm::None;
  }

  for (int i = 0; i < shape.GetRank(); ++i) {
    if (i < offset) {
      outer_dims_size *= shape.GetDimensionSize(i);
    } else {
      inner_dims_size *= shape.GetDimensionSize(i);
    }
  }

  return {{outer_dims_size, inner_dims_size}};
}

static llvm::Optional<ReductionDims3> IsMiddleReduction(
    const TensorShape& shape, ArrayRef<int32_t> reduction_indices) {
  Index outer_dim_size = 1;
  for (int i = 0; i < reduction_indices.front(); ++i)
    outer_dim_size *= shape.GetDimensionSize(i);
  Index middle_dim_size = shape.GetDimensionSize(reduction_indices.front());
  for (int i = 1; i < reduction_indices.size(); ++i) {
    if (reduction_indices[i] != reduction_indices[i - 1] + 1) return llvm::None;
    middle_dim_size *= shape.GetDimensionSize(reduction_indices[i]);
  }
  Index inner_dim_size = 1;
  for (int i = reduction_indices.back() + 1; i < shape.GetRank(); ++i)
    inner_dim_size *= shape.GetDimensionSize(i);
  return {{outer_dim_size, middle_dim_size, inner_dim_size}};
}

template <typename T, typename ReduceOp, typename TransformOp>
static llvm::Error Reduce(GpuDispatchContext* dctx, const DenseGpuTensor& input,
                          const GpuBuffer& output, const TensorMetadata& out_md,
                          ArrayRef<int32_t> reduction_indices, T init,
                          ReduceOp reduce, TransformOp transform) {
  auto input_ptr = GetRawPointer<const T>(input);
  auto output_ptr = GetRawPointer<T>(output);

  if (out_md.shape.GetRank() == 0) {
    return FullReduction(dctx, input_ptr, output_ptr, input.NumElements(), init,
                         reduce, transform);
  } else if (auto dims = IsOuterReduction(input.shape(), reduction_indices)) {
    // View input tensor as a 2d tensor: [outer_dim_size, inner_dim_size].
    return OuterReduction(dctx, input_ptr, output_ptr, dims->outer_dim_size,
                          dims->inner_dim_size, init, reduce, transform);
  } else if (auto dims = IsInnerReduction(input.shape(), reduction_indices)) {
    // View input tensor as a 2d tensor: [outer_dim_size, inner_dims_size].
    return InnerReduction(dctx, input_ptr, output_ptr, dims->outer_dim_size,
                          dims->inner_dim_size, init, reduce, transform);
  } else if (auto dims = IsMiddleReduction(input.shape(), reduction_indices)) {
    // View input tensor as a 3d tensor:
    //     [outer_dim_size, middle_dim_size, inner_dims_size].
    return MiddleDimReduction(dctx, input_ptr, output_ptr, dims->outer_dim_size,
                              dims->middle_dim_size, dims->inner_dim_size, init,
                              reduce, transform);
  } else {
    return MakeStringError(
        "Invalid reduction requested: in_rank=", input.shape().GetRank(),
        " out_rank=", out_md.shape.GetRank());
  }
}

//===----------------------------------------------------------------------===//
// TFRT reduction op implementations.
//===----------------------------------------------------------------------===//

template <typename T>
static llvm::Error ReduceMean(GpuDispatchContext* dctx,
                              const DenseGpuTensor& input,
                              const GpuBuffer& output,
                              const TensorMetadata& out_md,
                              ArrayRef<int32_t> reduction_indices) {
  // Number of input elements per output element.
  Index num_reduced = input.NumElements() / out_md.shape.GetNumElements();
  return Reduce(dctx, input, output, out_md, reduction_indices, T{0},
                cub::Sum(), Multiplies<T>(T{1} / num_reduced));
}

static llvm::Expected<DenseGpuTensor> ComputeMeanGpuOpImpl(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    ArrayRef<int32_t> reduction_indices, const TensorMetadata& result_md) {
  size_t num_result_elements = result_md.shape.GetNumElements();
  size_t size_in_bytes = GetHostSize(result_md.dtype) * num_result_elements;

  TFRT_ASSIGN_OR_RETURN(
      GpuBuffer output_buffer,
      GpuBuffer::Allocate(dctx->allocator(),
                          /*size=*/size_in_bytes, dctx->stream()));

  switch (input.dtype()) {
    default:
      return MakeStringError("Unsupported data type: ", input.dtype());

    case DType::F16:
      if (auto err = ReduceMean<Eigen::half>(dctx, input, output_buffer,
                                             result_md, reduction_indices))
        return std::move(err);
      break;

    case DType::F32:
      if (auto err = ReduceMean<float>(dctx, input, output_buffer, result_md,
                                       reduction_indices))
        return std::move(err);
      break;
  }

  return DenseGpuTensor(
      result_md.shape, result_md.dtype,
      MakeAvailableAsyncValueRef<GpuBuffer>(std::move(output_buffer)));
}

static llvm::Expected<DenseGpuTensor> ComputeMeanGpuOpFolded(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const OpAttrsRef& attrs, const TensorMetadata& result_md) {
  DenseAttr dense_attr;
  if (!attrs.Get("reduction_indices", &dense_attr)) {
    return MakeStringError(
        "tf.Mean needs a `reduction_indices` dense attribute");
  }

  DenseView dense_view = CreateDenseView(dense_attr);
  switch (dense_view.dtype()) {
    case DType::I32:
      return ComputeMeanGpuOpImpl(dctx, input, dense_view.GetFlat<int32_t>(),
                                  result_md);
    case DType::I64: {
      llvm::SmallVector<int32_t, 4> reduction_indices;
      llvm::copy(dense_view.GetFlat<int64_t>(),
                 std::back_inserter(reduction_indices));
      return ComputeMeanGpuOpImpl(dctx, input, reduction_indices, result_md);
    }
    default:
      llvm_unreachable("unsupported dtype for reduction indices input tf.Mean");
  }
}

static llvm::Expected<DenseGpuTensor> ComputeMeanGpuOp(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const DenseGpuTensor& /*reduction_indices*/) {
  // TODO(tfrt-devs): Read reduction_indices from a dense host tensor.
  auto channel_order = GuessChannelOrder(input.shape());
  if (!channel_order) return MakeStringError("Could not guess channel order.");
  auto spatial_offset = *channel_order == ChannelOrder::ChannelLast ? 1 : 2;
  llvm::SmallVector<int32_t, 2> reduction_indices = {spatial_offset,
                                                     spatial_offset + 1};

  llvm::SmallVector<Index, 2> result_dims;
  if (*channel_order == ChannelOrder::ChannelLast) {
    result_dims = {input.shape().GetDimensionSize(0),
                   input.shape().GetDimensionSize(3)};
  } else {
    result_dims = {input.shape().GetDimensionSize(0),
                   input.shape().GetDimensionSize(1)};
  }
  TensorMetadata result_md(input.dtype(), result_dims);

  return ComputeMeanGpuOpImpl(dctx, input, reduction_indices, result_md);
}

void RegisterReductionGpuTfOps(GpuOpRegistry* registry) {
  registry->AddOp("tf.Mean", TFRT_GPU_OP(gpu::ComputeMeanGpuOp));

  // "_tf.Mean" is a compiler-optimized version of "tf.Mean", where the argument
  // "reduction_indices" is folded to an attribute.
  registry->AddOp("_tf.Mean", TFRT_GPU_OP(gpu::ComputeMeanGpuOpFolded),
                  {"reduction_indices"});
}

}  // namespace gpu
}  // namespace tfrt
