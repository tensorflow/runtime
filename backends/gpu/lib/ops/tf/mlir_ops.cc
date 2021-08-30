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

// Collates list of all TF operations with pre-generated GPU code.

#include <numeric>
#include <unordered_map>

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/ops/tf/bias_add_f16_kernel.h"
#include "tfrt/gpu/ops/tf/bias_add_f32_kernel.h"
#include "tfrt/gpu/ops/tf/bias_add_f64_kernel.h"
#include "tfrt/gpu/ops/tf/relu_f16_kernel.h"
#include "tfrt/gpu/ops/tf/relu_f32_kernel.h"
#include "tfrt/gpu/ops/tf/relu_f64_kernel.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/logging.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {
namespace gpu {
static auto AllocateBuffer(GpuDispatchContext* dctx, const DType& dtype,
                           const TensorShape& shape) {
  return GpuBuffer::Allocate(dctx->allocator(),
                             shape.GetNumElements() * GetHostSize(dtype),
                             dctx->stream());
}

// Loads a CUDA module from an cubin or fatbin image.
class ModuleLoader {
  struct ContextHash {
    size_t operator()(const wrapper::Context& context) const {
      switch (context.platform()) {
        case wrapper::Platform::CUDA:
          return std::hash<CUcontext>()(static_cast<CUcontext>(context));
        case wrapper::Platform::ROCm:
          return std::hash<hipCtx_t>()(static_cast<hipCtx_t>(context));
        case wrapper::Platform::NONE:
          return 0;
      }
    }
  };

 public:
  explicit ModuleLoader(const void* image) : image_(image) {}

  llvm::Expected<wrapper::Function> GetFunction(GpuDispatchContext* dctx,
                                                const char* func_name) {
    auto ctx = dctx->current_context().context();
    auto it = functions_.find(ctx);
    if (it == functions_.end()) {
      auto module = LoadModule(dctx, image_);
      wrapper::Function function = GetFunction(module.get(), func_name);
      if (function) modules_.push_back(std::move(module));
      // Add function even if it's null because the error won't change.
      it = functions_.emplace_hint(it, ctx, function);
    }
    if (auto function = it->second) return function;
    return MakeStringError("Failed to load kernel");
  }

 private:
  // Reports errors directly and always returns a (potentially null) module.
  static wrapper::OwningModule LoadModule(GpuDispatchContext* dctx,
                                          const void* image) {
    static const size_t kLogBufferSize = 64 * 1024;
    std::array<char, kLogBufferSize> log_buffer = {};
    std::vector<CUjit_option> jit_options = {CU_JIT_FALLBACK_STRATEGY,
                                             CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                                             CU_JIT_ERROR_LOG_BUFFER};
    std::vector<void*> jit_values = {
        static_cast<char*>(nullptr) + /*CU_PREFER_BINARY=*/1,
        static_cast<char*>(nullptr) + log_buffer.size(), log_buffer.data()};
    auto module_or_error = CuModuleLoadDataEx(dctx->current_context(), image,
                                              jit_options, jit_values);
    if (log_buffer.front() != '\0') {
      TFRT_LOG(ERROR) << MakeStringError("JIT compilation error: ",
                                         log_buffer.data());
    }
    if (module_or_error) return std::move(*module_or_error);
    TFRT_LOG(ERROR) << module_or_error.takeError();
    return nullptr;
  }

  static wrapper::Function GetFunction(wrapper::Module module,
                                       const char* func_name) {
    if (!module) return {};
    auto function_or_error = wrapper::CuModuleGetFunction(module, func_name);
    if (function_or_error) return *function_or_error;
    TFRT_LOG(ERROR) << function_or_error.takeError();
    return {};
  }

  const void* image_;  // Not owning.
  std::vector<wrapper::OwningModule> modules_;
  std::unordered_map<wrapper::Context, wrapper::Function, ContextHash>
      functions_;
};

template <size_t N>
struct MemRefArgument {
  const void* base;  // Always null.
  const void* data;
  int64_t offset;  // Always 0.
  std::array<int64_t, N> dimensions;
  std::array<int64_t, N> strides;
};

template <size_t N>
MemRefArgument<N> MakeMemRefArgument(wrapper::Pointer<const void> ptr,
                                     const std::array<Index, N>& dimensions) {
  assert(dimensions.size() == N);
  MemRefArgument<N> result = {nullptr, ptr.raw(), 0};
  int64_t stride = 1;
  for (int i = N - 1; i >= 0; --i) {
    auto dim = static_cast<int64_t>(dimensions[i]);
    result.dimensions[i] = dim;
    result.strides[i] = stride;
    stride *= dim;
  }
  return result;
}

template <size_t N, typename It>
void CopyMemRefArgumentPtrs(const MemRefArgument<N>& memref_arg, It out) {
  *out++ = &memref_arg.base;
  *out++ = &memref_arg.data;
  *out++ = &memref_arg.offset;
  for (const auto& dimension : memref_arg.dimensions) *out++ = &dimension;
  for (const auto& stride : memref_arg.strides) *out++ = &stride;
}

auto GetGridDim(llvm::ArrayRef<Index> shape,
                llvm::ArrayRef<unsigned> block_dim) {
  std::array<unsigned, 3> grid_dim;
  for (int i = 0; i < shape.size(); ++i)
    grid_dim[i] = (shape[i] + block_dim[i] - 1) / block_dim[i];
  for (int i = shape.size(); i < 3; ++i) grid_dim[i] = 1;
  return grid_dim;
}

static llvm::Expected<DenseGpuTensor> ComputeBiasAddGpuOp(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const DenseGpuTensor& bias, const TensorMetadata& result_md) {
  TFRT_ASSIGN_OR_RETURN(auto output_buffer,
                        AllocateBuffer(dctx, result_md.dtype, result_md.shape));

  TFRT_ASSIGN_OR_RETURN(
      auto function, [&]() -> llvm::Expected<wrapper::Function> {
        switch (input.dtype()) {
          case DType::F16: {
            static ModuleLoader module_loader(kBiasAddF16Kernel);
            return module_loader.GetFunction(dctx, "bias_add_kernel");
          }
          case DType::F32: {
            static ModuleLoader module_loader(kBiasAddF32Kernel);
            return module_loader.GetFunction(dctx, "bias_add_kernel");
          }
          case DType::F64: {
            static ModuleLoader module_loader(kBiasAddF32Kernel);
            return module_loader.GetFunction(dctx, "bias_add_kernel");
          }
          default:
            return MakeStringError("Unsupported type: ", input.dtype());
        }
      }());

  llvm::SmallVector<Index, 4> dimensions;
  input.shape().GetDimensions(&dimensions);
  std::array<Index, 2> shape = {
      {std::accumulate(dimensions.begin(), dimensions.end() - 1, 1,
                       std::multiplies<Index>()),
       dimensions.back()}};
  std::array<Index, 1> bias_shape = {{shape[1]}};

  auto output_arg = MakeMemRefArgument(output_buffer.pointer(), shape);
  auto input_arg = MakeMemRefArgument(input.buffer().pointer(), shape);
  auto bias_arg = MakeMemRefArgument(bias.buffer().pointer(), bias_shape);

  llvm::SmallVector<const void*, 32> arg_ptrs;
  auto arg_ptrs_inserter = std::back_inserter(arg_ptrs);
  CopyMemRefArgumentPtrs(input_arg, arg_ptrs_inserter);
  CopyMemRefArgumentPtrs(bias_arg, arg_ptrs_inserter);
  CopyMemRefArgumentPtrs(output_arg, arg_ptrs_inserter);

  // Note: this corresponds to the tile_size in gen_kernel_image_hdr().
  std::array<unsigned, 3> block_dim = {{16, 16, 1}};
  auto grid_dim = GetGridDim(shape, block_dim);
  unsigned shared_memory_size_bytes = 0;

  if (auto error = CuLaunchKernel(
          dctx->current_context(), function, grid_dim[0], grid_dim[1],
          grid_dim[2], block_dim[0], block_dim[1], block_dim[2],
          shared_memory_size_bytes, dctx->stream(), arg_ptrs, nullptr))
    return std::move(error);

  return DenseGpuTensor(
      result_md.shape, result_md.dtype,
      MakeAvailableAsyncValueRef<GpuBuffer>(std::move(output_buffer)));
}

static llvm::Expected<DenseGpuTensor> ComputeReluGpuOp(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const TensorMetadata& result_md) {
  TFRT_ASSIGN_OR_RETURN(auto output_buffer,
                        AllocateBuffer(dctx, result_md.dtype, result_md.shape));

  TFRT_ASSIGN_OR_RETURN(
      auto function, [&]() -> llvm::Expected<wrapper::Function> {
        switch (input.dtype()) {
          case DType::F16: {
            static ModuleLoader module_loader(kReluF16Kernel);
            return module_loader.GetFunction(dctx, "relu_kernel");
          }
          case DType::F32: {
            static ModuleLoader module_loader(kReluF32Kernel);
            return module_loader.GetFunction(dctx, "relu_kernel");
          }
          case DType::F64: {
            static ModuleLoader module_loader(kReluF64Kernel);
            return module_loader.GetFunction(dctx, "relu_kernel");
          }
          default:
            return MakeStringError("Unsupported type: ", input.dtype());
        }
      }());

  std::array<Index, 1> shape = {{input.shape().GetNumElements()}};

  auto input_arg = MakeMemRefArgument(input.buffer().pointer(), shape);
  auto output_arg = MakeMemRefArgument(output_buffer.pointer(), shape);

  llvm::SmallVector<const void*, 32> arg_ptrs;
  auto arg_ptrs_inserter = std::back_inserter(arg_ptrs);
  CopyMemRefArgumentPtrs(input_arg, arg_ptrs_inserter);
  CopyMemRefArgumentPtrs(output_arg, arg_ptrs_inserter);

  // Note: this corresponds to the tile_size in gen_kernel_image_hdr().
  std::array<unsigned, 3> block_dim = {{256, 1, 1}};
  auto grid_dim = GetGridDim(shape, block_dim);
  unsigned shared_memory_size_bytes = 0;

  if (auto error = CuLaunchKernel(
          dctx->current_context(), function, grid_dim[0], grid_dim[1],
          grid_dim[2], block_dim[0], block_dim[1], block_dim[2],
          shared_memory_size_bytes, dctx->stream(), arg_ptrs, nullptr))
    return std::move(error);

  return DenseGpuTensor(
      result_md.shape, result_md.dtype,
      MakeAvailableAsyncValueRef<GpuBuffer>(std::move(output_buffer)));
}

void RegisterMlirGpuTfOps(GpuOpRegistry* registry) {
  registry->AddOp("tf.BiasAdd", TFRT_GPU_OP(ComputeBiasAddGpuOp));
  registry->AddOp("tf.Relu", TFRT_GPU_OP(ComputeReluGpuOp));
}
}  // namespace gpu
}  // namespace tfrt
