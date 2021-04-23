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

// Thin wrapper around the CUDA runtime API adding llvm::Error and explicit
// context.
#ifndef TFRT_GPU_WRAPPER_CUDART_WRAPPER_H_
#define TFRT_GPU_WRAPPER_CUDART_WRAPPER_H_

#include <cstddef>
#include <memory>

#include "cuda_runtime.h"  // from @cuda_headers
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

extern template void internal::LogResult(llvm::raw_ostream&, cudaError_t);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cudaError_t error);

// The following functions map directly to CUDA runtime calls.
//
// Note: prefer the driver API (see cuda_wrapper.h) over this runtime API.
// The API here is merely for interacting with auto-registered CUDA kernels.
//
// Please consult NVIDIA's documentation for more detail:
// http://docs.nvidia.com/cuda/cuda-runtime-api

// The CUDA runtime is initialized implicitly. cudaFree(nullptr) is the standard
// way to force initialization. For details, see
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#initialization
llvm::Error CudaFree(std::nullptr_t);

llvm::Expected<cudaDeviceProp> CudaGetDeviceProperties(CurrentContext current);
llvm::Expected<int> CudaRuntimeGetVersion();
llvm::Error CudaGetLastError(CurrentContext current);
llvm::Error CudaPeekAtLastError(CurrentContext current);
llvm::Error CudaLaunchKernel(CurrentContext current, const void* function,
                             dim3 grid_dim, dim3 block_dim, void** arguments,
                             size_t shared_memory_size_bytes, CUstream stream);
llvm::Error CudaLaunchCooperativeKernel(CurrentContext current,
                                        const void* function, dim3 grid_dim,
                                        dim3 block_dim, void** arguments,
                                        size_t shared_memory_size_bytes,
                                        CUstream stream);
llvm::Error CudaLaunchCooperativeKernelMultiDevice(
    CurrentContext current, struct cudaLaunchParams* arguments,
    unsigned int numDevices, unsigned int flags);
llvm::Error CudaFuncSetCacheConfig(CurrentContext current, const void* function,
                                   cudaFuncCache cacheConfig);
llvm::Error CudaFuncSetSharedMemConfig(CurrentContext current,
                                       const void* function,
                                       cudaSharedMemConfig config);
llvm::Expected<cudaFuncAttributes> CudaFuncGetAttributes(CurrentContext current,
                                                         const void* function);
llvm::Error CudaFuncSetAttribute(CurrentContext current, const void* function,
                                 cudaFuncAttribute attribute, int value);

namespace internal {
template <typename... Ts, size_t... Is>
std::array<void*, sizeof...(Ts)> GetArrayOfElementPointersImpl(
    std::tuple<Ts...>* tuple, std::index_sequence<Is...>) {
  return {{&std::get<Is>(*tuple)...}};
}
template <typename... Ts>
std::array<void*, sizeof...(Ts)> GetArrayOfElementPointers(
    std::tuple<Ts...>* tuple) {
  return GetArrayOfElementPointersImpl(tuple, std::index_sequence_for<Ts...>{});
}

template <bool...>
struct BoolPack;
template <bool... Bs>
using AllFalse = std::is_same<BoolPack<Bs..., false>, BoolPack<false, Bs...>>;
}  // namespace internal

// Helper function to launch kernels.
template <typename... Ts, typename... Args>
llvm::Error CudaLaunchKernel(CurrentContext current, void (*function)(Ts...),
                             dim3 grid_dim, dim3 block_dim,
                             size_t shared_memory_size_bytes, CUstream stream,
                             Args... arguments) {
  static_assert(internal::AllFalse<(std::is_reference<Ts>::value)...>::value,
                "Kernels with reference arguments have undefined behaviour.");
  // Cast arguments and forward them as an array of pointers.
  auto args_tuple = std::tuple<Ts...>(arguments...);
  auto arg_ptrs = internal::GetArrayOfElementPointers(&args_tuple);
  auto func_ptr = reinterpret_cast<const void*>(function);
  return CudaLaunchKernel(current, func_ptr, grid_dim, block_dim,
                          arg_ptrs.data(), shared_memory_size_bytes, stream);
}
template <typename... Ts, typename... Args>
llvm::Error CudaLaunchCooperativeKernel(CurrentContext current,
                                        void (*function)(Ts...), dim3 grid_dim,
                                        dim3 block_dim,
                                        size_t shared_memory_size_bytes,
                                        CUstream stream, Args... arguments) {
  static_assert(internal::AllFalse<(std::is_reference<Ts>::value)...>::value,
                "Kernels with reference arguments have undefined behaviour.");
  // Cast arguments and forward them as an array of pointers.
  auto args_tuple = std::tuple<Ts...>(arguments...);
  auto arg_ptrs = internal::GetArrayOfElementPointers(&args_tuple);
  auto func_ptr = reinterpret_cast<const void*>(function);
  return CudaLaunchCooperativeKernel(current, func_ptr, grid_dim, block_dim,
                                     arg_ptrs.data(), shared_memory_size_bytes,
                                     stream);
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CUDART_WRAPPER_H_
