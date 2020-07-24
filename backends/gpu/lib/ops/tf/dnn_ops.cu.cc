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

//===- dnn_ops.cu.cc --- Implements DNN related CUDA kernels ---*- C++ -*--===//
//
// Implements hand-written CUDA kernels useful for DNN ops.
//
//===----------------------------------------------------------------------===//
#include "dnn_ops_cu.h"
//
#include "tfrt/common/ops/tf/dnn_ops_util.h"
// TODO(fishx): use gpu native type instead of eigen for fp16.
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/gpu/memory/gpu_buffer.h"
#include "tfrt/gpu/stream/cudart_wrapper.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"

namespace tfrt {
namespace gpu {
namespace {

// Wrapper for CUDA __ldg. Reads the data and caches it in read-only data cache.
template <typename T>
__host__ __device__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
  return __ldg(ptr);
#else
  return *ptr;
#endif
}

// Specialize ldg for Eigen::half.
template <>
__host__ __device__ Eigen::half ldg(const Eigen::half* ptr) {
  return Eigen::half(Eigen::half_impl::raw_uint16_to_half(ldg(&ptr->x)));
}

int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T, int IndexCount, T DefaultValue>
struct Array {
  Array(const std::array<T, IndexCount>& array) {
    for (int i = 0; i < IndexCount; i++) {
      data[i] = array[i];
    }
  }
  __host__ __device__ const T& operator[](int index) const {
    return data[index];
  }
  __host__ __device__ T& operator[](int index) { return data[index]; }
  __host__ __device__ Array() {
    for (int i = 0; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  __host__ __device__ Array(T a0) {
    data[0] = a0;
    for (int i = 1; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  __host__ __device__ Array(T a0, T a1) {
    data[0] = a0;
    data[1] = a1;
    for (int i = 2; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  __host__ __device__ Array(T a0, T a1, T a2) {
    data[0] = a0;
    data[1] = a1;
    data[2] = a2;
    for (int i = 3; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  T data[IndexCount];
};

// Helper for range-based for loop using 'delta' increments.
// Usage: see GpuGridRangeX() functions below.
template <typename T>
class GpuGridRange {
  struct Iterator {
    __device__ Iterator(T index, T delta) : index_(index), delta_(delta) {}
    __device__ T operator*() const { return index_; }
    __device__ Iterator& operator++() {
      index_ += delta_;
      return *this;
    }
    __device__ bool operator!=(const Iterator& other) const {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      // Anything past an end iterator (delta_ == 0) is equal.
      // In range-based for loops, this optimizes to 'return less'.
      if (!other.delta_) {
        return less;
      }
      if (!delta_) {
        return greater;
      }
      return less || greater;
    }

   private:
    T index_;
    const T delta_;
  };

 public:
  __device__ GpuGridRange(T begin, T delta, T end)
      : begin_(begin), delta_(delta), end_(end) {}

  __device__ Iterator begin() const { return Iterator{begin_, delta_}; }
  __device__ Iterator end() const { return Iterator{end_, 0}; }

 private:
  T begin_;
  T delta_;
  T end_;
};

template <typename T>
__device__ GpuGridRange<T> GpuGridRangeX(T count) {
  return GpuGridRange<T>(blockIdx.x * blockDim.x + threadIdx.x,
                         gridDim.x * blockDim.x, count);
}

// A dimension type with compile-time known size.
template <int IndexCount>
struct Dimension : Array<int, IndexCount, 1> {
  typedef Array<int, IndexCount, 1> Base;
  __host__ __device__ Dimension() : Base() {}
  __host__ __device__ Dimension(int a0) : Base(a0) {}
  __host__ __device__ Dimension(int a0, int a1) : Base(a0, a1) {}
  __host__ __device__ Dimension(int a0, int a1, int a2) : Base(a0, a1, a2) {}
  Dimension(const std::array<int, IndexCount>& array) : Base(array) {}
};

// An index type with compile-time known size.
template <int IndexCount>
struct Index : Array<int, IndexCount, 0> {
  typedef Array<int, IndexCount, 0> Base;
  __host__ __device__ Index() : Base() {}
  __host__ __device__ Index(int a0) : Base(a0) {}
  __host__ __device__ Index(int a0, int a1) : Base(a0, a1) {}
  __host__ __device__ Index(int a0, int a1, int a2) : Base(a0, a1, a2) {}
};

// A helper function that converts a tensor index into an array index.
template <int IndexCount>
__host__ __device__ int TensorIndexToFlat(const Index<IndexCount>& index,
                                          const Dimension<IndexCount>& dims) {
  int flat_index = index[0];
  for (int i = 1; i < IndexCount; i++) {
    flat_index = flat_index * dims[i] + index[i];
  }
  return flat_index;
}

// A helper function that converts an array index into a tensor index.
template <int IndexCount>
__host__ __device__ Index<IndexCount> FlatToTensorIndex(
    int index, const Dimension<IndexCount>& dims) {
  Index<IndexCount> tensor_index;
  for (int i = IndexCount - 1; i >= 0; i--) {
    int new_index = index / dims[i];
    tensor_index[i] = index - dims[i] * new_index;
    index = new_index;
  }
  return tensor_index;
}

// A simple CUDA custom kernel to shuffle dimensions of a 3D tensor according to
// the given shuffle permutation in template parameters. Shuffle permutation
// <sp0, sp1, sp2> shuffles dimensions such that input dimension 0 goes to sp0,
// 1 goes to sp1 and 2 goes to sp2. For example, shuffle permutation <2, 0, 1>
// will populate output so that input[x][y][z] is equal to (*output)[y][z][x].
//
// Requires that nthreads is equal to the total number of elements in the input
// tensor.
template <typename T, int sp0, int sp1, int sp2>
__global__ void ShuffleInTensor3Simple(int nthreads,
                                       const T* __restrict__ input,
                                       Dimension<3> input_dims,
                                       T* __restrict__ output) {
  Dimension<3> output_dims;
  output_dims[sp0] = input_dims[0];
  output_dims[sp1] = input_dims[1];
  output_dims[sp2] = input_dims[2];

  // Iterate over output as opposed to iterating over input for better
  // performance. Iterating over output will generate sequential writes and
  // random reads that performs better compared to sequential reads and random
  // writes.
  for (int output_index : GpuGridRangeX<int>(nthreads)) {
    Index<3> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<3> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[sp0];
    input_tensor_index[1] = output_tensor_index[sp1];
    input_tensor_index[2] = output_tensor_index[sp2];

    int input_index = TensorIndexToFlat(input_tensor_index, input_dims);

    output[output_index] = ldg(input + input_index);
  }
}

struct GpuLaunchConfig {
  // Logical number of thread that works on the elements. If each logical
  // thread works on exactly a single element, this is the same as the working
  // element count.
  int virtual_thread_count = -1;
  // Number of threads per block.
  int thread_per_block = -1;
  // Number of blocks for GPU kernel launch.
  int block_count = -1;
};

// Calculate the GPU launch config we should use for a kernel launch.
// This is assuming the kernel is quite simple and will largely be
// memory-limited.
// REQUIRES: work_element_count > 0.
llvm::Expected<GpuLaunchConfig> GetGpuLaunchConfig(
    stream::CurrentContext current, int work_element_count) {
  using PropPair = std::pair<CUcontext, std::unique_ptr<cudaDeviceProp>>;
  // TODO(iga): If user keep creating contexts, this vector will grow forever.
  // Ideally, we should have some way to storing/retrieving per context state
  // from CurrentContext, whose life-time is properly managed.
  static llvm::SmallVector<PropPair, 8>* configs =
      new llvm::SmallVector<PropPair, 8>();

  assert(work_element_count > 0);

  const cudaDeviceProp* props = nullptr;
  CUcontext cu_ctx = static_cast<CUcontext>(current.context());
  for (const PropPair& prop_pair : *configs) {
    if (prop_pair.first == cu_ctx) {
      props = prop_pair.second.get();
      break;
    }
  }
  if (props == nullptr) {
    // TODO(iga): Expose cuDeviceGetProperties instead. It has the
    // equivalent hipDeviceGetProperties.
    TFRT_ASSIGN_OR_RETURN(cudaDeviceProp tmp_props,
                          stream::CudaGetDeviceProperties(current));
    configs->emplace_back(
        std::make_pair(cu_ctx, std::make_unique<cudaDeviceProp>(tmp_props)));
    props = configs->back().second.get();
  }

  GpuLaunchConfig config;
  const int virtual_thread_count = work_element_count;
  const int physical_thread_count =
      std::min(props->multiProcessorCount * props->maxThreadsPerMultiProcessor,
               virtual_thread_count);
  const int thread_per_block = std::min(1024, props->maxThreadsPerBlock);
  const int block_count =
      std::min(DivUp(physical_thread_count, thread_per_block),
               props->maxThreadsPerMultiProcessor);

  config.virtual_thread_count = virtual_thread_count;
  config.thread_per_block = thread_per_block;
  config.block_count = block_count;
  return config;
}

template <typename T>
struct TransformFilter {
  llvm::Error operator()(stream::CurrentContext current,
                         const stream::Stream& stream,
                         ChannelOrder channel_order, const DenseGpuTensor& in,
                         GpuBuffer* out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.shape().GetDimensionSize(0);  // spatial dimensions
    for (int i = 1; i < 2; i++) {
      combined_dims[0] *= in.shape().GetDimensionSize(i);
    }
    combined_dims[1] = in.shape().GetDimensionSize(2);  // input filters
    combined_dims[2] = in.shape().GetDimensionSize(3);  // output filters
    TFRT_ASSIGN_OR_RETURN(GpuLaunchConfig config,
                          GetGpuLaunchConfig(current, in.NumElements()));

    if (channel_order == ChannelOrder::ChannelFirst) {
      ShuffleInTensor3Simple<T, 2, 1, 0>
          <<<config.block_count, config.thread_per_block, 0,
             static_cast<cudaStream_t>(stream)>>>(
              config.virtual_thread_count, GetRawPointer<T>(in), combined_dims,
              GetRawPointer<T>(*out));
    } else if (channel_order == ChannelOrder::ChannelLast) {
      ShuffleInTensor3Simple<T, 1, 2, 0>
          <<<config.block_count, config.thread_per_block, 0,
             static_cast<cudaStream_t>(stream)>>>(
              config.virtual_thread_count, GetRawPointer<T>(in), combined_dims,
              GetRawPointer<T>(*out));
    } else {
      return MakeStringError("Unsupported channel order: ", channel_order);
    }
    return llvm::Error::success();
  }
};

// -------------------------------------------------------------------------- //
// FusedBatchNormInferenceFunctor implementation.                             //
// -------------------------------------------------------------------------- //

// Generic kernel, that does all computations by converting input to U data
// type. We use it when CUDA architecture doesn't have fast arithmetic for the
// T data type (e.g. no fp16 arithmetic before sm_53).
// TODO(tfrt-devs): Evaluate possible performance regression from removing
// all template parameters except T and U.
template <typename T, typename U, ChannelOrder channel_order,
          bool add_side_input, FusedBatchNormActivationMode activation_mode,
          bool is_generic_kernel>
struct FusedBatchNormInferenceKernel {
  static_assert(channel_order == ChannelOrder::ChannelFirst ||
                    channel_order == ChannelOrder::ChannelLast,
                "Unsupported channel order");

  __device__ static void run(int32_t count, int32_t channels_size,
                             int32_t inner_dim_size, const T* __restrict__ in,
                             const U* __restrict__ scale,
                             const U* __restrict__ offset,
                             const U* __restrict__ mean,
                             const U* __restrict__ var,
                             const T* __restrict__ side_input, float epsilon,
                             T* __restrict__ out) {
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t total_device_threads = gridDim.x * blockDim.x;

    while (index < count) {
      const int channel = (channel_order == ChannelOrder::ChannelLast)
                              ? index % channels_size
                              : (index / inner_dim_size) % channels_size;

      U in_v = U(in[index]);
      U scale_v = scale[channel];
      U offset_v = offset[channel];
      U mean_v = mean[channel];
      U var_v = var[channel];

      U scaling_factor_v = rsqrt(var_v + epsilon) * scale_v;
      static_assert(std::is_same<U, float>::value, "U data type must be float");
      U shifted_v = fmaf(in_v - mean_v, scaling_factor_v, offset_v);

      if (add_side_input) {
        shifted_v += U(side_input[index]);
      }

      if (activation_mode == FusedBatchNormActivationMode::kIdentity) {
        out[index] = T(shifted_v);
      } else if (activation_mode == FusedBatchNormActivationMode::kRelu) {
        out[index] = T(shifted_v < U(0) ? U(0) : shifted_v);
      }

      index += total_device_threads;
    }
  }
};

// Specialization for T=Eigen::half and U=float. Not used currently.
// TODO(b/135435976): Temporary disable non-generic kernel implementation.
template <ChannelOrder channel_order, bool add_side_input,
          FusedBatchNormActivationMode activation_mode>
struct FusedBatchNormInferenceKernel<Eigen::half, float, channel_order,
                                     add_side_input, activation_mode,
                                     /*is_generic_kernel=*/false> {
  using T = Eigen::half;
  using U = float;

  // If CUDA architecture doesn't support fast fp16 computation, we will
  // fallback on generic kernel defined above.
  using GenericKernel =
      FusedBatchNormInferenceKernel<T, U, channel_order, add_side_input,
                                    activation_mode,
                                    /*is_generic_kernel=*/true>;

  __device__ static void run(int32_t count, int32_t channels_size,
                             int32_t inner_dim_size, const T* __restrict__ in,
                             const U* __restrict__ scale,
                             const U* __restrict__ offset,
                             const U* __restrict__ mean,
                             const U* __restrict__ var,
                             const T* __restrict__ side_input, float epsilon,
                             T* __restrict__ out) {
    // Old GPUs do not have (or have very slow) fp16 arithmetic.
    // TODO(b/150632628): Update this a higher version since fp16 arithmetic
    // throughput is low for 610.
#if __CUDA_ARCH__ >= 610
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t total_device_threads = gridDim.x * blockDim.x;

    int32_t half2_count = count >> 1;

    half epsilon_h = __float2half(epsilon);
    half2 epsilon_h2 = __float2half2_rn(epsilon);

    const int32_t max_channel_size = channels_size - 1;

    while (index < half2_count) {
      int32_t channel[2];
      if (channel_order == ChannelOrder::ChannelLast) {
        channel[0] = (2 * index) % channels_size;
        channel[1] = channel[0] == max_channel_size ? 0 : channel[0] + 1;
      } else {
        channel[0] = ((2 * index) / inner_dim_size) % channels_size;
        channel[1] = ((2 * index + 1) / inner_dim_size) % channels_size;
      }

      half2 in_v = reinterpret_cast<const half2*>(in)[index];
      half2 scale_v = __floats2half2_rn(scale[channel[0]], scale[channel[1]]);
      half2 offset_v =
          __floats2half2_rn(offset[channel[0]], offset[channel[1]]);
      half2 mean_v = __floats2half2_rn(mean[channel[0]], mean[channel[1]]);
      half2 var_v = __floats2half2_rn(var[channel[0]], var[channel[1]]);

      half2 scaling_factor_v =
          __hmul2(h2rsqrt(__hadd2(var_v, epsilon_h2)), scale_v);
      half2 shifted_v =
          __hfma2(__hsub2(in_v, mean_v), scaling_factor_v, offset_v);

      if (add_side_input) {
        shifted_v = __hadd2(shifted_v,
                            reinterpret_cast<const half2*>(side_input)[index]);
      }

      if (activation_mode == FusedBatchNormActivationMode::kIdentity) {
        reinterpret_cast<half2*>(out)[index] = shifted_v;

      } else if (activation_mode == FusedBatchNormActivationMode::kRelu) {
        const half2 kZeroH = __float2half2_rn(0.f);
        const half2 mask_h = __hgt2(shifted_v, kZeroH);
        reinterpret_cast<half2*>(out)[index] = __hmul2(mask_h, shifted_v);
      }

      index += total_device_threads;
    }

    if ((count & 0x1) == 1 && index == half2_count) {
      index = count - 1;

      const int32_t channel = (channel_order == ChannelOrder::ChannelLast)
                                  ? index % channels_size
                                  : (index / inner_dim_size) % channels_size;

      half in_v = in[index];
      half scale_v = __float2half(scale[channel]);
      half offset_v = __float2half(offset[channel]);
      half mean_v = __float2half(mean[channel]);
      half var_v = __float2half(var[channel]);

      half scaling_factor_v = __hmul(hrsqrt(__hadd(var_v, epsilon_h)), scale_v);
      half shifted_v = __hfma(__hsub(in_v, mean_v), scaling_factor_v, offset_v);

      if (add_side_input) {
        shifted_v = __hadd(shifted_v, side_input[index]);
      }

      if (activation_mode == FusedBatchNormActivationMode::kIdentity) {
        out[index] = shifted_v;

      } else if (activation_mode == FusedBatchNormActivationMode::kRelu) {
        // TODO(b/150631743): Update with a faster implementation.
        const half kZeroH = __float2half(0.f);
        const half mask_h = __hgt(shifted_v, kZeroH);
        out[index] = __hmul(mask_h, shifted_v);
      }
    }

#else
    GenericKernel::run(count, channels_size, inner_dim_size, in, scale, offset,
                       mean, var, side_input, epsilon, out);
#endif  // __CUDA_ARCH__ >= 610
  }
};

template <typename T, typename U, ChannelOrder channel_order,
          bool add_side_input, FusedBatchNormActivationMode activation_mode>
__global__ void FusedBatchNormInferenceMetaKernel(
    int32_t count, int32_t channels_size, int32_t inner_dim_size, const T* in,
    const U* scale, const U* offset, const U* mean, const U* var,
    const T* side_input, float epsilon, T* out) {
  // We prefer to run non-generic specialization, for the given types T and U.
  // TODO(b/135435976): Temporary disable non-generic kernel implementation.
  FusedBatchNormInferenceKernel<
      T, U, channel_order, add_side_input, activation_mode,
      /*is_generic_kernel=*/true>::run(count, channels_size, inner_dim_size, in,
                                       scale, offset, mean, var, side_input,
                                       epsilon, out);
}

// This is a functor to launch custom CUDA kernel for FusedBatchNorm inference
// with side input and activation.
template <typename T, typename U>
struct FusedBatchNormInferenceFunctor {
  llvm::Error operator()(
      stream::CurrentContext current, const stream::Stream& stream,
      ChannelOrder channel_order, const DenseGpuTensor& input,
      const DenseGpuTensor& scale, const DenseGpuTensor& bias,
      const DenseGpuTensor& mean, const DenseGpuTensor& variance,
      const DenseGpuTensor* side_input, float epsilon,
      FusedBatchNormActivationMode activation_mode, GpuBuffer* output_buffer) {
    int32_t count = input.NumElements();
    if (count == 0) return llvm::Error::success();

    TFRT_ASSIGN_OR_RETURN(GpuLaunchConfig config,
                          GetGpuLaunchConfig(current, count));

    const bool no_side_input = side_input == nullptr;
    const bool add_side_input = !no_side_input;
    const T* side_input_ptr =
        no_side_input ? nullptr : GetRawPointer<T>(*side_input);

    const bool no_activation =
        activation_mode == FusedBatchNormActivationMode::kIdentity;
    const bool relu_activation =
        activation_mode == FusedBatchNormActivationMode::kRelu;

    auto launch = [&](auto* kernel, int channel_size, int inner_dim_size) {
      return stream::CudaLaunchKernel(
          current, kernel, config.block_count, config.thread_per_block, 0,
          stream, count, channel_size, inner_dim_size, GetRawPointer<T>(input),
          GetRawPointer<U>(scale), GetRawPointer<U>(bias),
          GetRawPointer<U>(mean), GetRawPointer<U>(variance), side_input_ptr,
          epsilon, GetRawPointer<T>(*output_buffer));
    };

    auto input_shape = GetDimensions(input.shape());
    if (channel_order == ChannelOrder::ChannelFirst) {
      const int channel_size = input_shape[1];
      const int inner_dim_size = input_shape[2] * input_shape[3];
      if (no_activation && no_side_input) {
        return launch(&FusedBatchNormInferenceMetaKernel<
                          T, U, ChannelOrder::ChannelFirst,
                          /*add_side_input=*/false,
                          FusedBatchNormActivationMode::kIdentity>,
                      channel_size, inner_dim_size);
      } else if (relu_activation && no_side_input) {
        return launch(
            &FusedBatchNormInferenceMetaKernel<
                T, U, ChannelOrder::ChannelFirst,
                /*add_side_input=*/false, FusedBatchNormActivationMode::kRelu>,
            channel_size, inner_dim_size);
      } else if (no_activation && add_side_input) {
        return launch(&FusedBatchNormInferenceMetaKernel<
                          T, U, ChannelOrder::ChannelFirst,
                          /*add_side_input=*/true,
                          FusedBatchNormActivationMode::kIdentity>,
                      channel_size, inner_dim_size);
      } else if (relu_activation && add_side_input) {
        return launch(
            &FusedBatchNormInferenceMetaKernel<
                T, U, ChannelOrder::ChannelFirst,
                /*add_side_input=*/true, FusedBatchNormActivationMode::kRelu>,
            channel_size, inner_dim_size);
      }
    } else if (channel_order == ChannelOrder::ChannelLast) {
      const int channel_size = input_shape[3];
      const int inner_dim_size = 1;
      if (no_activation && no_side_input) {
        return launch(&FusedBatchNormInferenceMetaKernel<
                          T, U, ChannelOrder::ChannelLast,
                          /*add_side_input=*/false,
                          FusedBatchNormActivationMode::kIdentity>,
                      channel_size, inner_dim_size);
      } else if (relu_activation && no_side_input) {
        return launch(
            &FusedBatchNormInferenceMetaKernel<
                T, U, ChannelOrder::ChannelLast,
                /*add_side_input=*/false, FusedBatchNormActivationMode::kRelu>,
            channel_size, inner_dim_size);
      } else if (no_activation && add_side_input) {
        return launch(&FusedBatchNormInferenceMetaKernel<
                          T, U, ChannelOrder::ChannelLast,
                          /*add_side_input=*/true,
                          FusedBatchNormActivationMode::kIdentity>,
                      channel_size, inner_dim_size);
      } else if (relu_activation && add_side_input) {
        return launch(
            &FusedBatchNormInferenceMetaKernel<
                T, U, ChannelOrder::ChannelLast,
                /*add_side_input=*/true, FusedBatchNormActivationMode::kRelu>,
            channel_size, inner_dim_size);
      }
    }
    return MakeStringError("no fused batch norm kernel was launched");
  }
};

}  // namespace

llvm::Error TransformFilterTensor(stream::CurrentContext current,
                                  const stream::Stream& stream,
                                  ChannelOrder channel_order,
                                  const DenseGpuTensor& input_filter,
                                  GpuBuffer* output_filter) {
  switch (input_filter.dtype().kind()) {
#define DTYPE_NUMERIC(ENUM)                                                 \
  case DType::ENUM: {                                                       \
    TransformFilter<tfrt::EigenTypeForDTypeKind<DType::ENUM>> tx;           \
    return tx(current, stream, channel_order, input_filter, output_filter); \
  }
#include "tfrt/dtype/dtype.def"
    default:
      return MakeStringError("TransformFilterTensor does not support dtype: ",
                             input_filter.dtype());
  }
}

llvm::Error FusedBatchNormEx(
    stream::CurrentContext current, const stream::Stream& stream,
    ChannelOrder channel_order, const DenseGpuTensor& input,
    const DenseGpuTensor& scale, const DenseGpuTensor& bias,
    const DenseGpuTensor& mean, const DenseGpuTensor& variance,
    const DenseGpuTensor* side_input, float epsilon,
    FusedBatchNormActivationMode activation_mode, GpuBuffer* output_buffer) {
  switch (input.dtype().kind()) {
    case DType::F16: {
      auto functor = FusedBatchNormInferenceFunctor<Eigen::half, float>();
      if (auto error = functor(current, stream, channel_order, input, scale,
                               bias, mean, variance, side_input, epsilon,
                               activation_mode, output_buffer))
        return error;
      break;
    }
    case DType::F32: {
      auto functor = FusedBatchNormInferenceFunctor<float, float>();
      if (auto error = functor(current, stream, channel_order, input, scale,
                               bias, mean, variance, side_input, epsilon,
                               activation_mode, output_buffer))
        return error;
      break;
    }
    default:
      llvm_unreachable("unsupported dtype for input in FusedBatchNormEx");
  }

  return llvm::Error::success();
}

}  // namespace gpu
}  // namespace tfrt
