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

// Thin wrapper around the HIP API adding llvm::Error and explicit context.
#ifndef TFRT_GPU_WRAPPER_HIP_WRAPPER_H_
#define TFRT_GPU_WRAPPER_HIP_WRAPPER_H_

#include <cstddef>
#include <memory>

#include "tfrt/gpu/wrapper/driver_wrapper.h"
#include "tfrt/gpu/wrapper/hip_stub.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, hipError_t error);

// The following functions map directly to HIP calls.
//
// Please consult AMD's documentation for more detail:
// http://github.com/ROCm-Developer-Tools/HIP

llvm::Error HipInit();
llvm::Error HipFree(std::nullptr_t);
llvm::Expected<int> HipDriverGetVersion();
llvm::Expected<int> HipRuntimeGetVersion();

llvm::Error HipGetLastError(CurrentContext current);
llvm::Error HipPeekAtLastError(CurrentContext current);
llvm::Expected<hipDeviceProp_t> HipGetDeviceProperties(CurrentContext current);

llvm::Expected<int> HipDeviceGetCount();
llvm::Expected<Device> HipDeviceGet(int ordinal);
llvm::Expected<std::string> HipDeviceGetName(Device device);
llvm::Expected<size_t> HipDeviceTotalMem(Device device);
llvm::Expected<int> HipDeviceGetAttribute(hipDeviceAttribute_t attribute,
                                          Device device);
llvm::Expected<std::string> HipDeviceGetPCIBusId(Device device);
llvm::Expected<size_t> HipDeviceGetLimit(CurrentContext current,
                                         hipLimit_t limit);
llvm::Expected<StreamPriorityRange> HipDeviceGetStreamPriorityRange(
    CurrentContext current);
llvm::Error HipDeviceSynchronize(CurrentContext current);
llvm::Expected<int> HipDeviceCanAccessPeer(Device src_dev, Device dst_dev);
llvm::Error HipDeviceEnablePeerAccess(CurrentContext current,
                                      Device peer_device);
llvm::Error HipDeviceDisablePeerAccess(CurrentContext current,
                                       Device peer_device);

llvm::Expected<OwningContext> HipDevicePrimaryCtxRetain(Device device);
llvm::Error HipDevicePrimaryCtxRelease(Device device);
llvm::Error HipDevicePrimaryCtxReset(Device device);
llvm::Expected<ContextState> HipDevicePrimaryCtxGetState(Device device);
llvm::Error HipDevicePrimaryCtxSetFlags(Device device, hipDeviceFlags_t flags);

llvm::Expected<OwningContext> HipCtxCreate(hipDeviceFlags_t flags,
                                           Device device);
llvm::Error HipCtxDestroy(hipCtx_t context);
// Discouraged, have the caller provide a Context or CurrentContext instead.
// See documentation of CurrentContext for details.
llvm::Expected<CurrentContext> HipCtxGetCurrent();
llvm::Expected<CurrentContext> HipCtxSetCurrent(hipCtx_t context);
llvm::Expected<unsigned> HipCtxGetApiVersion(hipCtx_t context);
llvm::Expected<Device> HipCtxGetDevice(CurrentContext current);
llvm::Expected<hipDeviceFlags_t> HipCtxGetFlags(CurrentContext current);

llvm::Expected<OwningStream> HipStreamCreate(CurrentContext current,
                                             hipStreamFlags_t flags);
llvm::Expected<OwningStream> HipStreamCreate(CurrentContext current,
                                             hipStreamFlags_t flags,
                                             int priority);
llvm::Error HipStreamDestroy(hipStream_t stream);
llvm::Expected<int> HipStreamGetPriority(hipStream_t stream);
llvm::Expected<hipStreamFlags_t> HipStreamGetFlags(hipStream_t stream);
llvm::Error HipStreamSynchronize(hipStream_t stream);
llvm::Expected<bool> HipStreamQuery(hipStream_t stream);
llvm::Error HipStreamWaitEvent(hipStream_t stream, hipEvent_t event);

llvm::Expected<OwningEvent> HipEventCreate(CurrentContext current,
                                           hipEventFlags_t flags);
llvm::Error HipEventDestroy(hipEvent_t event);
llvm::Error HipEventRecord(hipEvent_t event, hipStream_t stream);
llvm::Error HipEventSynchronize(hipEvent_t event);
llvm::Expected<bool> HipEventQuery(hipEvent_t event);
llvm::Expected<float> HipEventElapsedTime(hipEvent_t start, hipEvent_t end);

llvm::Expected<DeviceMemory<void>> HipMemAlloc(CurrentContext current,
                                               size_t size_bytes);
llvm::Error HipMemFree(Pointer<void> pointer);
llvm::Expected<HostMemory<void>> HipMemHostAlloc(CurrentContext current,
                                                 size_t size_bytes,
                                                 hipHostMallocFlags_t flags);
llvm::Error HipMemHostFree(Pointer<void> pointer);
llvm::Expected<RegisteredMemory<void>> HipMemHostRegister(
    CurrentContext current, void* ptr, size_t size_bytes,
    hipHostRegisterFlags_t flags);
llvm::Error HipMemHostUnregister(Pointer<void> pointer);
llvm::Expected<DeviceMemory<void>> HipMemAllocManaged(
    CurrentContext current, size_t size_bytes, hipMemAttachFlags_t flags);

llvm::Expected<Pointer<void>> HipMemHostGetDevicePointer(
    Pointer<void> host_ptr);
llvm::Expected<MemoryRange<void>> HipMemGetAddressRange(CurrentContext current,
                                                        Pointer<void> ptr);
llvm::Expected<MemoryInfo> HipMemGetInfo(CurrentContext current);
llvm::Expected<hipPointerAttribute_t> HipPointerGetAttributes(
    Pointer<const void> ptr);

llvm::Error HipMemcpy(CurrentContext current, Pointer<void> dst,
                      Pointer<const void> src, size_t count_bytes);
llvm::Error HipMemcpyAsync(CurrentContext current, Pointer<void> dst,
                           Pointer<const void> src, size_t count_bytes,
                           hipStream_t stream);
llvm::Error HipMemcpyPeer(Pointer<void> dst_ptr, Device dst_dev,
                          Pointer<const void> src_ptr, Device src_dev,
                          size_t count_bytes);
llvm::Error HipMemcpyPeerAsync(Pointer<void> dst_ptr, Device dst_dev,
                               Pointer<const void> src_ptr, Device src_dev,
                               size_t count_bytes, hipStream_t stream);
llvm::Error HipMemsetD8(CurrentContext current, Pointer<void> dst,
                        std::uint8_t value, size_t count);
llvm::Error HipMemsetD32(CurrentContext current, Pointer<void> dst,
                         std::uint32_t value, size_t count);
llvm::Error HipMemsetD8Async(CurrentContext current, Pointer<void> dst,
                             std::uint8_t value, size_t count,
                             hipStream_t stream);
llvm::Error HipMemsetD32Async(CurrentContext current, Pointer<void> dst,
                              std::uint32_t value, size_t count,
                              hipStream_t stream);

llvm::Expected<OwningModule> HipModuleLoadData(CurrentContext current,
                                               const void* image);
llvm::Expected<OwningModule> HipModuleLoadDataEx(
    CurrentContext current, const void* image,
    llvm::ArrayRef<hipJitOption> options, llvm::ArrayRef<void*> option_values);

llvm::Error HipModuleUnload(hipModule_t module);
llvm::Expected<Function> HipModuleGetFunction(hipModule_t module,
                                              const char* name);
llvm::Expected<MemoryRange<void>> HipModuleGetGlobal(hipModule_t module,
                                                     const char* name);

llvm::Expected<hipFuncAttributes> HipFuncGetAttributes(CurrentContext current,
                                                       hipFunction_t function);

llvm::Error HipLaunchKernel(CurrentContext current, hipFunction_t function,
                            unsigned grid_dim_x, unsigned grid_dim_y,
                            unsigned grid_dim_z, unsigned block_dim_x,
                            unsigned block_dim_y, unsigned block_dim_z,
                            unsigned shared_memory_size_bytes,
                            hipStream_t stream, llvm::ArrayRef<void*> arguments,
                            llvm::ArrayRef<void*> extras);
llvm::Error HipLaunchCooperativeKernel(
    CurrentContext current, hipFunction_t function, unsigned grid_dim_x,
    unsigned grid_dim_y, unsigned grid_dim_z, unsigned block_dim_x,
    unsigned block_dim_y, unsigned block_dim_z,
    unsigned shared_memory_size_bytes, hipStream_t stream,
    llvm::ArrayRef<void*> arguments);

llvm::Expected<int> HipOccupancyMaxActiveBlocksPerMultiprocessor(
    CurrentContext current, hipFunction_t function, int block_size,
    size_t dynamic_shared_memory_size);
llvm::Expected<MaxPotentialBlockSize> HipOccupancyMaxPotentialBlockSize(
    CurrentContext current, hipFunction_t function,
    size_t dynamic_shared_memory_size, int block_size_limit);

// Helper functions to allocate memory of type T. Does not perform any manual
// alignment.
template <typename T>
llvm::Expected<DeviceMemory<T>> HipMemAlloc(CurrentContext current,
                                            size_t count) {
  auto memory = HipMemAlloc(current, sizeof(T) * count);
  if (!memory) return memory.takeError();
  return DeviceMemory<T>(static_cast<Pointer<T>>(memory->release()));
}
template <typename T>
llvm::Expected<HostMemory<T>> HipMemHostAlloc(CurrentContext current,
                                              size_t count,
                                              hipHostMallocFlags_t flags) {
  auto memory = HipMemHostAlloc(current, sizeof(T) * count, flags);
  if (!memory) return memory.takeError();
  return HostMemory<T>(static_cast<Pointer<T>>(memory->release()));
}
template <typename T>
llvm::Expected<DeviceMemory<T>> HipMemAllocManaged(CurrentContext current,
                                                   size_t count,
                                                   hipMemAttachFlags_t flags) {
  auto memory = HipMemAllocManaged(current, sizeof(T) * count, flags);
  if (!memory) return memory.takeError();
  return DeviceMemory<T>(static_cast<Pointer<T>>(memory->release()));
}

// Helper functions to get a device pointer or address range of type T.
template <typename T>
llvm::Expected<Pointer<T>> HipMemHostGetDevicePointer(Pointer<T> host_ptr) {
  void* void_ptr = const_cast<void*>(host_ptr.raw(Platform::ROCm));
  auto result = HipMemHostGetDevicePointer({void_ptr, Platform::ROCm});
  if (!result) return result.takeError();
  return static_cast<Pointer<T>>(result);
}
template <typename T>
llvm::Expected<MemoryRange<T>> HipMemGetAddressRange(CurrentContext current,
                                                     Pointer<T> ptr) {
  void* void_ptr = const_cast<void*>(ptr.raw(Platform::ROCm));
  auto result = HipMemGetAddressRange(current, {void_ptr, Platform::ROCm});
  if (!result) return result.takeError();
  return {static_cast<Pointer<T>>(result->base), result->size_bytes};
}

// Helper function to launch kernels. Attention, no argument type checking is
// performed!
template <typename... Args>
llvm::Error HipLaunchKernel(CurrentContext current, hipFunction_t function,
                            std::array<unsigned, 3> grid_dim,
                            std::array<unsigned, 3> block_dim,
                            unsigned shared_memory_size_bytes,
                            hipStream_t stream, Args... arguments) {
  // Forward arguments as an array of pointers.
  std::array<void*, sizeof...(Args)> arg_ptrs = {{&arguments...}};
  return HipLaunchKernel(current, function, grid_dim[0], grid_dim[1],
                         grid_dim[2], block_dim[0], block_dim[1], block_dim[2],
                         shared_memory_size_bytes, stream, arg_ptrs, nullptr);
}
template <typename... Args>
llvm::Error HipLaunchCooperativeKernel(CurrentContext current,
                                       hipFunction_t function,
                                       std::array<unsigned, 3> grid_dim,
                                       std::array<unsigned, 3> block_dim,
                                       unsigned shared_memory_size_bytes,
                                       hipStream_t stream, Args... arguments) {
  // Forward arguments as an array of pointers.
  std::array<void*, sizeof...(Args)> arg_ptrs = {{&arguments...}};
  return HipLaunchCooperativeKernel(
      current, function, grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0],
      block_dim[1], block_dim[2], shared_memory_size_bytes, stream, arg_ptrs);
}
}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_HIP_WRAPPER_H_
