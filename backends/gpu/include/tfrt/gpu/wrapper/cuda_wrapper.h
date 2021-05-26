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

// Thin wrapper around the CUDA API adding llvm::Error and explicit context.
#ifndef TFRT_GPU_WRAPPER_CUDA_WRAPPER_H_
#define TFRT_GPU_WRAPPER_CUDA_WRAPPER_H_

#include <cstddef>
#include <memory>

#include "cuda.h"  // from @cuda_headers
#include "tfrt/gpu/wrapper/driver_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, CUresult result);

// The following functions map directly to CUDA calls.
//
// Please consult NVIDIA's documentation for more detail:
// http://docs.nvidia.com/cuda/cuda-driver-api

llvm::Error CuInit();
llvm::Expected<int> CuDriverGetVersion();

llvm::Expected<int> CuDeviceGetCount();
llvm::Expected<Device> CuDeviceGet(int ordinal);
llvm::Expected<std::string> CuDeviceGetName(Device device);
llvm::Expected<size_t> CuDeviceTotalMem(Device device);
llvm::Expected<int> CuDeviceGetAttribute(CUdevice_attribute attribute,
                                         Device device);
llvm::Expected<int> CuDeviceCanAccessPeer(Device src_dev, Device dst_dev);
llvm::Expected<int> CuDeviceGetP2PAttribute(CUdevice_P2PAttribute attribute,
                                            Device src_dev, Device dst_dev);

llvm::Expected<OwningContext> CuDevicePrimaryCtxRetain(Device device);
llvm::Error CuDevicePrimaryCtxRelease(Device device);
llvm::Error CuDevicePrimaryCtxReset(Device device);
llvm::Expected<ContextState> CuDevicePrimaryCtxGetState(Device device);
llvm::Error CuDevicePrimaryCtxSetFlags(Device device, CUctx_flags flags);

llvm::Expected<OwningContext> CuCtxCreate(CUctx_flags flags, Device device);
llvm::Error CuCtxDestroy(CUcontext context);
// Discouraged, have the caller provide a Context or CurrentContext instead.
// See documentation of CurrentContext for details.
llvm::Expected<CurrentContext> CuCtxGetCurrent();
llvm::Expected<CurrentContext> CuCtxSetCurrent(CUcontext context);
llvm::Error CuCtxSynchronize(CurrentContext current);
llvm::Expected<unsigned> CuCtxGetApiVersion(CUcontext context);
llvm::Expected<Device> CuCtxGetDevice(CurrentContext current);
llvm::Expected<CUctx_flags> CuCtxGetFlags(CurrentContext current);
llvm::Expected<StreamPriorityRange> CuCtxGetStreamPriorityRange(
    CurrentContext current);
llvm::Expected<size_t> CuCtxGetLimit(CurrentContext current, CUlimit limit);
llvm::Error CuCtxSetLimit(CurrentContext current, CUlimit limit, size_t value);
llvm::Expected<CUfunc_cache> CuCtxGetCacheConfig(CurrentContext current);
llvm::Error CuCtxSetCacheConfig(CurrentContext current, CUfunc_cache config);
llvm::Expected<CUsharedconfig> CuCtxGetSharedMemConfig(CurrentContext current);
llvm::Error CuCtxSetSharedMemConfig(CurrentContext current,
                                    CUsharedconfig config);
llvm::Error CuCtxEnablePeerAccess(CurrentContext current,
                                  CUcontext peer_context);
llvm::Error CuCtxDisablePeerAccess(CurrentContext current,
                                   CUcontext peer_context);

llvm::Expected<OwningStream> CuStreamCreate(CurrentContext current,
                                            CUstream_flags flags);
llvm::Expected<OwningStream> CuStreamCreate(CurrentContext current,
                                            CUstream_flags flags, int priority);
llvm::Error CuStreamDestroy(CUstream stream);
llvm::Expected<Context> CuStreamGetCtx(CUstream stream);
llvm::Expected<CUstream_flags> CuStreamGetFlags(CUstream stream);
llvm::Expected<int> CuStreamGetPriority(CUstream stream);
llvm::Error CuStreamSynchronize(CUstream stream);
llvm::Expected<bool> CuStreamQuery(CUstream stream);
llvm::Error CuStreamWaitEvent(CUstream stream, CUevent event);

llvm::Expected<OwningEvent> CuEventCreate(CurrentContext current,
                                          CUevent_flags flags);
llvm::Error CuEventDestroy(CUevent event);
llvm::Error CuEventRecord(CUevent event, CUstream stream);
llvm::Error CuEventSynchronize(CUevent event);
llvm::Expected<bool> CuEventQuery(CUevent event);
llvm::Expected<float> CuEventElapsedTime(CUevent start, CUevent end);

llvm::Expected<DeviceMemory<void>> CuMemAlloc(CurrentContext current,
                                              size_t size_bytes);
llvm::Error CuMemFree(Pointer<void> pointer);
llvm::Expected<HostMemory<void>> CuMemHostAlloc(CurrentContext current,
                                                size_t size_bytes,
                                                CUmemhostalloc_flags flags);
llvm::Error CuMemHostFree(Pointer<void> pointer);
llvm::Expected<RegisteredMemory<void>> CuMemHostRegister(
    CurrentContext current, void* ptr, size_t size_bytes,
    CUmemhostregister_flags flags);
llvm::Error CuMemHostUnregister(Pointer<void> pointer);
llvm::Expected<DeviceMemory<void>> CuMemAllocManaged(CurrentContext current,
                                                     size_t size_bytes,
                                                     CUmemAttach_flags flags);

llvm::Expected<Pointer<void>> CuMemHostGetDevicePointer(Pointer<void> host_ptr);
llvm::Expected<MemoryRange<void>> CuMemGetAddressRange(CurrentContext current,
                                                       Pointer<void> ptr);
llvm::Expected<MemoryInfo> CuMemGetInfo(CurrentContext current);
llvm::Expected<CUmemhostalloc_flags> CuMemHostGetFlags(CurrentContext current,
                                                       Pointer<void> ptr);
llvm::Error CuMemRangeGetAttribute(void* data, size_t data_size,
                                   CUmem_range_attribute attribute,
                                   Pointer<const void> ptr, size_t size_bytes);
llvm::Error CuMemRangeGetAttributes(
    llvm::ArrayRef<void*> data, llvm::ArrayRef<size_t> data_sizes,
    llvm::ArrayRef<CUmem_range_attribute> attributes, Pointer<const void> ptr,
    size_t size_bytes);
llvm::Error CuPointerGetAttribute(void* data, CUpointer_attribute attribute,
                                  Pointer<const void> ptr);
llvm::Error CuPointerGetAttributes(
    llvm::ArrayRef<void*> data, llvm::ArrayRef<CUpointer_attribute> attributes,
    Pointer<const void> ptr);

llvm::Error CuMemcpy(CurrentContext current, Pointer<void> dst,
                     Pointer<const void> src, size_t count_bytes);
llvm::Error CuMemcpyAsync(CurrentContext current, Pointer<void> dst,
                          Pointer<const void> src, size_t count_bytes,
                          CUstream stream);
llvm::Error CuMemcpyPeer(Pointer<void> dst_ptr, CUcontext dst_ctx,
                         Pointer<const void> src_ptr, CUcontext src_ctx,
                         size_t count_bytes);
llvm::Error CuMemcpyPeerAsync(Pointer<void> dst_ptr, CUcontext dst_ctx,
                              Pointer<const void> src_ptr, CUcontext src_ctx,
                              size_t count_bytes, CUstream stream);
llvm::Error CuMemsetD8(CurrentContext current, Pointer<void> dst,
                       std::uint8_t value, size_t count);
llvm::Error CuMemsetD16(CurrentContext current, Pointer<void> dst,
                        std::uint16_t value, size_t count);
llvm::Error CuMemsetD32(CurrentContext current, Pointer<void> dst,
                        std::uint32_t value, size_t count);
llvm::Error CuMemsetD8Async(CurrentContext current, Pointer<void> dst,
                            std::uint8_t value, size_t count, CUstream stream);
llvm::Error CuMemsetD16Async(CurrentContext current, Pointer<void> dst,
                             std::uint16_t value, size_t count,
                             CUstream stream);
llvm::Error CuMemsetD32Async(CurrentContext current, Pointer<void> dst,
                             std::uint32_t value, size_t count,
                             CUstream stream);

llvm::Expected<OwningModule> CuModuleLoadData(CurrentContext current,
                                              const void* image);
llvm::Expected<OwningModule> CuModuleLoadDataEx(
    CurrentContext current, const void* image,
    llvm::ArrayRef<CUjit_option> jit_options,
    llvm::ArrayRef<void*> jit_option_values);
llvm::Error CuModuleUnload(CUmodule module);
llvm::Expected<Function> CuModuleGetFunction(CUmodule module, const char* name);

llvm::Expected<int> CuFuncGetAttribute(CurrentContext current,
                                       CUfunction_attribute attribute,
                                       CUfunction function);
llvm::Error CuFuncSetAttribute(CurrentContext current, CUfunction function,
                               CUfunction_attribute attribute, int value);
llvm::Error CuFuncSetCacheConfig(CurrentContext current, CUfunction function,
                                 CUfunc_cache config);
llvm::Error CuFuncSetSharedMemConfig(CurrentContext current,
                                     CUfunction function,
                                     CUsharedconfig config);

llvm::Error CuLaunchKernel(CurrentContext current, CUfunction function,
                           unsigned grid_dim_x, unsigned grid_dim_y,
                           unsigned grid_dim_z, unsigned block_dim_x,
                           unsigned block_dim_y, unsigned block_dim_z,
                           unsigned shared_memory_size_bytes, CUstream stream,
                           llvm::ArrayRef<const void*> arguments,
                           llvm::ArrayRef<const void*> extras);
llvm::Error CuLaunchCooperativeKernel(
    CurrentContext current, CUfunction function, unsigned grid_dim_x,
    unsigned grid_dim_y, unsigned grid_dim_z, unsigned block_dim_x,
    unsigned block_dim_y, unsigned block_dim_z,
    unsigned shared_memory_size_bytes, CUstream stream,
    llvm::ArrayRef<const void*> arguments);

llvm::Expected<int> CuOccupancyMaxActiveBlocksPerMultiprocessor(
    CurrentContext current, CUfunction function, int block_size,
    size_t dynamic_shared_memory_size);
llvm::Expected<int> CuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    CurrentContext current, CUfunction function, int block_size,
    size_t dynamic_shared_memory_size, CUoccupancy_flags flags);
llvm::Expected<MaxPotentialBlockSize> CuOccupancyMaxPotentialBlockSize(
    CurrentContext current, CUfunction function,
    const std::function<size_t(int)>& block_size_to_dynamic_shared_memory_size,
    int block_size_limit);
llvm::Expected<MaxPotentialBlockSize> CuOccupancyMaxPotentialBlockSizeWithFlags(
    CurrentContext current, CUfunction function,
    const std::function<size_t(int)>& block_size_to_dynamic_shared_memory_size,
    int block_size_limit, CUoccupancy_flags flags);

// Helper functions to allocate memory of type T. Does not perform any manual
// alignment.
template <typename T>
llvm::Expected<DeviceMemory<T>> CuMemAlloc(CurrentContext current,
                                           size_t count) {
  auto memory = CuMemAlloc(current, sizeof(T) * count);
  if (!memory) return memory.takeError();
  return DeviceMemory<T>(static_cast<Pointer<T>>(memory->release()));
}
template <typename T>
llvm::Expected<HostMemory<T>> CuMemHostAlloc(CurrentContext current,
                                             size_t count,
                                             CUmemhostalloc_flags flags) {
  auto memory = CuMemHostAlloc(current, sizeof(T) * count, flags);
  if (!memory) return memory.takeError();
  return HostMemory<T>(static_cast<Pointer<T>>(memory->release()));
}
template <typename T>
llvm::Expected<DeviceMemory<T>> CuMemAllocManaged(CurrentContext current,
                                                  size_t count,
                                                  CUmemAttach_flags flags) {
  auto memory = CuMemAllocManaged(current, sizeof(T) * count, flags);
  if (!memory) return memory.takeError();
  return DeviceMemory<T>(static_cast<Pointer<T>>(memory->release()));
}

// Helper functions to get a device pointer or address range of type T.
template <typename T>
llvm::Expected<Pointer<T>> CuMemHostGetDevicePointer(Pointer<T> host_ptr) {
  void* void_ptr = const_cast<void*>(host_ptr.raw(Platform::CUDA));
  auto result = CuMemHostGetDevicePointer({void_ptr, Platform::CUDA});
  if (!result) return result.takeError();
  return static_cast<Pointer<T>>(result);
}
template <typename T>
llvm::Expected<MemoryRange<T>> CuMemGetAddressRange(CurrentContext current,
                                                    Pointer<T> ptr) {
  void* void_ptr = const_cast<void*>(ptr.raw(Platform::CUDA));
  auto result = CuMemGetAddressRange(current, {void_ptr, Platform::CUDA});
  if (!result) return result.takeError();
  return {static_cast<Pointer<T>>(result->base), result->size_bytes};
}

// Helper function to launch kernels. Attention, no argument type checking is
// performed!
template <typename... Args>
llvm::Error CuLaunchKernel(CurrentContext current, CUfunction function,
                           std::array<unsigned, 3> grid_dim,
                           std::array<unsigned, 3> block_dim,
                           unsigned shared_memory_size_bytes, CUstream stream,
                           Args... arguments) {
  // Forward arguments as an array of pointers.
  std::array<const void*, sizeof...(Args)> arg_ptrs = {{&arguments...}};
  return CuLaunchKernel(current, function, grid_dim[0], grid_dim[1],
                        grid_dim[2], block_dim[0], block_dim[1], block_dim[2],
                        shared_memory_size_bytes, stream, arg_ptrs, nullptr);
}
template <typename... Args>
llvm::Error CuLaunchCooperativeKernel(CurrentContext current,
                                      CUfunction function,
                                      std::array<unsigned, 3> grid_dim,
                                      std::array<unsigned, 3> block_dim,
                                      unsigned shared_memory_size_bytes,
                                      CUstream stream, Args... arguments) {
  // Forward arguments as an array of pointers.
  std::array<const void*, sizeof...(Args)> arg_ptrs = {{&arguments...}};
  return CuLaunchCooperativeKernel(
      current, function, grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0],
      block_dim[1], block_dim[2], shared_memory_size_bytes, stream, arg_ptrs);
}
}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CUDA_WRAPPER_H_
