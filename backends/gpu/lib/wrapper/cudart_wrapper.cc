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

// Thin wrapper around the CUDA runtime API adding llvm::Error and explicit
// context.
#include "tfrt/gpu/wrapper/cudart_wrapper.h"

#include "llvm/Support/FormatVariadic.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

template llvm::raw_ostream& internal::operator<<(llvm::raw_ostream&,
                                                 const ErrorData<cudaError_t>&);

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cudaError_t error) {
  const char* name = cudaGetErrorName(error);
  if (name != nullptr) {
    os << name;
  } else {
    os << llvm::formatv("cudaError_t({0})", static_cast<int>(error));
  }
  const char* msg = cudaGetErrorString(error);
  if (msg != nullptr) os << " (" << msg << ")";
  return os;
}

llvm::Error CudaFree(std::nullptr_t) { return TO_ERROR(cudaFree(nullptr)); }

llvm::Expected<cudaDeviceProp> CudaGetDeviceProperties(CurrentContext current) {
  CheckCudaContext(current);
  int device;
  // Get device of the current context. We don't want to expose the device id of
  // the CUDA runtime, which is different from the device ordinal of the driver
  // API.
  RETURN_IF_ERROR(cudaGetDevice(&device));
  cudaDeviceProp properties;
  RETURN_IF_ERROR(cudaGetDeviceProperties(&properties, device));
  return properties;
}

llvm::Expected<int> CudaRuntimeGetVersion() {
  int version;
  RETURN_IF_ERROR(cudaRuntimeGetVersion(&version));
  return version;
}

llvm::Error CudaGetLastError(CurrentContext current) {
  CheckCudaContext(current);
  return TO_ERROR(cudaGetLastError());
}

llvm::Error CudaPeekAtLastError(CurrentContext current) {
  CheckCudaContext(current);
  return TO_ERROR(cudaPeekAtLastError());
}

llvm::Error CudaLaunchKernel(CurrentContext current, const void* function,
                             dim3 grid_dim, dim3 block_dim, void** arguments,
                             size_t shared_memory_size_bytes, CUstream stream) {
  CheckCudaContext(current);
  return TO_ERROR(cudaLaunchKernel(function, grid_dim, block_dim, arguments,
                                   shared_memory_size_bytes, stream));
}

llvm::Error CudaLaunchCooperativeKernel(CurrentContext current,
                                        const void* function, dim3 grid_dim,
                                        dim3 block_dim, void** arguments,
                                        size_t shared_memory_size_bytes,
                                        CUstream stream) {
  CheckCudaContext(current);
  return TO_ERROR(
      cudaLaunchCooperativeKernel(function, grid_dim, block_dim, arguments,
                                  shared_memory_size_bytes, stream));
}

llvm::Error CudaLaunchCooperativeKernelMultiDevice(
    CurrentContext current, struct cudaLaunchParams* arguments,
    unsigned int numDevices, unsigned int flags) {
  CheckCudaContext(current);
  return TO_ERROR(
      cudaLaunchCooperativeKernelMultiDevice(arguments, numDevices, flags));
}

llvm::Error CudaFuncSetCacheConfig(CurrentContext current, const void* function,
                                   cudaFuncCache cacheConfig) {
  CheckCudaContext(current);
  return TO_ERROR(cudaFuncSetCacheConfig(function, cacheConfig));
}

llvm::Error CudaFuncSetSharedMemConfig(CurrentContext current,
                                       const void* function,
                                       cudaSharedMemConfig config) {
  CheckCudaContext(current);
  return TO_ERROR(cudaFuncSetSharedMemConfig(function, config));
}

llvm::Expected<cudaFuncAttributes> CudaFuncGetAttributes(CurrentContext current,
                                                         const void* function) {
  CheckCudaContext(current);
  cudaFuncAttributes attributes;
  RETURN_IF_ERROR(cudaFuncGetAttributes(&attributes, function));
  return attributes;
}

llvm::Error CudaFuncSetAttribute(CurrentContext current, const void* function,
                                 cudaFuncAttribute attribute, int value) {
  CheckCudaContext(current);
  return TO_ERROR(cudaFuncSetAttribute(function, attribute, value));
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
