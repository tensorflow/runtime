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

// Thin abstraction layer for CUDA and HIP driver API.
#ifndef TFRT_GPU_WRAPPER_DRIVER_WRAPPER_H_
#define TFRT_GPU_WRAPPER_DRIVER_WRAPPER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>

#include "tfrt/gpu/wrapper/wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

// Platform-discriminated enums.
using ContextFlags = Enum<struct ContextFlagsTag>;
using StreamFlags = Enum<struct StreamFlagsTag>;
using EventFlags = Enum<struct EventFlagsTag>;
using MemHostAllocFlags = Enum<struct MemHostAllocFlagsTag>;
using MemHostRegisterFlags = Enum<struct MemHostRegisterFlagsTag>;
using MemAttachFlags = Enum<struct MemAttachFlagsTag>;

// Handle to CUDA or ROCm device.
class Device {
 public:
  Device() = default;
  Device(int device_id, Platform platform)
      : device_id_(device_id), platform_(platform) {}
  int id(Platform platform) const {
    assert(platform_ == platform);
    return device_id_;
  }
  Platform platform() const { return platform_; }
  bool operator==(Device other) const {
    return device_id_ == other.device_id_ && platform_ == other.platform_;
  }
  bool operator!=(Device other) const { return !(*this == other); }

 private:
  static_assert(std::is_same<CUdevice, int>::value, "");
  static_assert(std::is_same<hipDevice_t, int>::value, "");

  int device_id_ = 0;
  Platform platform_ = Platform::NONE;

  friend raw_ostream& operator<<(raw_ostream& os, Device device) {
    return os << device.device_id_ << " (" << device.platform() << ")";
  }
};

// Non-owning handles of GPU resources.
using Module = Resource<CUmodule, hipModule_t>;
// Note: does not support CU_STREAM_LEGACY (0x1) or CU_STREAM_PER_THREAD (0x2).
// Those special handle values use the lower two bits and therefore interfere
// with PointerIntPair. Will fail runtime asserts if used.
using Event = Resource<CUevent, hipEvent_t>;
using Function = Resource<CUfunction, hipFunction_t>;

namespace internal {
struct ModuleDeleter {
  using pointer = Module;
  void operator()(Module module) const;
};
struct EventDeleter {
  using pointer = Event;
  void operator()(Event event) const;
};

struct DeviceMemoryDeallocator {
  void operator()(Pointer<void> pointer) const;
};
struct HostMemoryDeallocator {
  void operator()(Pointer<void> pointer) const;
};
struct HostMemoryUnregisterer {
  void operator()(Pointer<void> pointer) const;
};
template <typename T, typename Deleter>
struct MemoryDeleter : public Deleter {
  using pointer = Pointer<T>;
};
template <typename T, typename Deleter>
using OwningMemory = std::unique_ptr<T, MemoryDeleter<T, Deleter>>;
}  // namespace internal

// RAII wrappers for resources. Instances own the underlying resource.
//
// They are implemented as std::unique_ptrs with custom deleters.
//
// Use get() and release() to access the non-owning handle, please use with
// appropriate care.
using OwningModule = internal::OwningResource<internal::ModuleDeleter>;
using OwningEvent = internal::OwningResource<internal::EventDeleter>;

// RAII wrappers for GPU memory. Instances own the underlying memory.
template <typename T>
using DeviceMemory =
    internal::OwningMemory<T, internal::DeviceMemoryDeallocator>;
template <typename T>
using HostMemory = internal::OwningMemory<T, internal::HostMemoryDeallocator>;
template <typename T>
using RegisteredMemory =
    internal::OwningMemory<T, internal::HostMemoryUnregisterer>;

// Return types for functions returning multiple values.
struct ContextState {
  ContextFlags flags;
  int active;
};
template <typename T>
struct MemoryRange {
  Pointer<T> base;
  size_t size_bytes;
};
struct MemoryInfo {
  size_t free_bytes;
  size_t total_bytes;
};
struct StreamPriorityRange {
  int least;
  int greatest;
};
struct MaxPotentialBlockSize {
  int min_num_blocks;
  int block_size;
};

// The following functions forward to the CUDA or HIP wrapper.
//
// This API corresponds to the intersection of the CUDA and HIP wrappers.

llvm::Error Init(Platform platform);
llvm::Expected<int> DriverGetVersion(Platform platform);

llvm::Expected<int> DeviceGetCount(Platform platform);
llvm::Expected<Device> DeviceGet(Platform platform, int ordinal);
llvm::Expected<std::string> DeviceGetName(Device device);
llvm::Expected<size_t> DeviceTotalMem(Device device);
llvm::Expected<int> DeviceCanAccessPeer(Device device, Device peer_dev);

llvm::Expected<OwningContext> DevicePrimaryCtxRetain(Device device);
llvm::Error DevicePrimaryCtxRelease(Device device);
llvm::Error DevicePrimaryCtxReset(Device device);
llvm::Expected<ContextState> DevicePrimaryCtxGetState(Device device);
llvm::Error DevicePrimaryCtxSetFlags(Device device, ContextFlags flags);

llvm::Expected<OwningContext> CtxCreate(Device device);
llvm::Expected<OwningContext> CtxCreate(ContextFlags flags, Device device);
llvm::Error CtxDestroy(Context context);
// Avoid if possible. See documentation of CurrentContext.
llvm::Expected<CurrentContext> CtxGetCurrent();
llvm::Expected<CurrentContext> CtxSetCurrent(Context context);
llvm::Error CtxSynchronize(CurrentContext current);
llvm::Expected<unsigned> CtxGetApiVersion(Context context);
llvm::Expected<Device> CtxGetDevice(CurrentContext current);
llvm::Expected<ContextFlags> CtxGetFlags(CurrentContext current);
llvm::Expected<StreamPriorityRange> CtxGetStreamPriorityRange(
    CurrentContext current);

llvm::Expected<OwningStream> StreamCreate(CurrentContext current,
                                          StreamFlags flags);
llvm::Expected<OwningStream> StreamCreate(CurrentContext current,
                                          StreamFlags flags, int priority);
llvm::Expected<OwningStream> StreamCreateNonBlocking(CurrentContext current);
llvm::Error StreamDestroy(Stream stream);
llvm::Expected<int> StreamGetPriority(Stream stream);
llvm::Expected<StreamFlags> StreamGetFlags(Stream stream);
llvm::Error StreamSynchronize(Stream stream);
llvm::Expected<bool> StreamQuery(Stream stream);
llvm::Error StreamWaitEvent(Stream stream, Event event);

llvm::Expected<OwningEvent> EventCreate(CurrentContext current,
                                        EventFlags flags);
llvm::Expected<OwningEvent> EventCreateNoTiming(CurrentContext current);
llvm::Error EventDestroy(Event event);
llvm::Error EventRecord(Event event, Stream stream);
llvm::Error EventSynchronize(Event event);
llvm::Expected<bool> EventQuery(Event event);
llvm::Expected<float> EventElapsedTime(Event start, Event end);

llvm::Expected<DeviceMemory<void>> MemAlloc(CurrentContext current,
                                            size_t size_bytes);
llvm::Error MemFree(Pointer<void> pointer);
llvm::Expected<HostMemory<void>> MemHostAlloc(CurrentContext current,
                                              size_t size_bytes,
                                              MemHostAllocFlags flags);
llvm::Expected<HostMemory<void>> MemHostAllocWriteCombined(
    CurrentContext current, size_t size_bytes);
llvm::Error MemHostFree(Pointer<void> pointer);
llvm::Expected<RegisteredMemory<void>> MemHostRegister(
    CurrentContext current, void* ptr, size_t size_bytes,
    MemHostRegisterFlags flags);
// Forwards to above with CU_MEMHOSTREGISTER_DEVICEMAP/hipHostRegisterMapped.
llvm::Expected<RegisteredMemory<void>> MemHostRegister(CurrentContext current,
                                                       void* ptr,
                                                       size_t size_bytes);
llvm::Error MemHostUnregister(Pointer<void> pointer);
llvm::Expected<DeviceMemory<void>> MemAllocManaged(CurrentContext current,
                                                   size_t size_bytes,
                                                   MemAttachFlags flags);

llvm::Expected<Pointer<void>> MemHostGetDevicePointer(Pointer<void> host_ptr);
llvm::Expected<MemoryRange<void>> MemGetAddressRange(CurrentContext current,
                                                     Pointer<void> ptr);
llvm::Expected<MemoryInfo> MemGetInfo(CurrentContext current);

llvm::Error Memcpy(CurrentContext current, Pointer<void> dst,
                   Pointer<const void> src, size_t count_bytes);
llvm::Error MemcpyAsync(CurrentContext current, Pointer<void> dst,
                        Pointer<const void> src, size_t count_bytes,
                        Stream stream);
llvm::Error MemcpyPeer(Pointer<void> dst_ptr, Context dst_ctx,
                       Pointer<const void> src_ptr, Context src_ctx,
                       size_t count_bytes);
llvm::Error MemcpyPeerAsync(Pointer<void> dst_ptr, Context dst_ctx,
                            Pointer<const void> src_ptr, Context src_ctx,
                            size_t count_bytes, Stream stream);
llvm::Error MemsetD8(CurrentContext current, Pointer<void> dst, std::uint8_t uc,
                     size_t count);
llvm::Error MemsetD32(CurrentContext current, Pointer<void> dst,
                      std::uint32_t value, size_t count);
llvm::Error MemsetD8Async(CurrentContext current, Pointer<void> dst,
                          std::uint8_t value, size_t count, Stream stream);
llvm::Error MemsetD32Async(CurrentContext current, Pointer<void> dst,
                           std::uint32_t value, size_t count, Stream stream);

llvm::Expected<OwningModule> ModuleLoadData(CurrentContext current,
                                            const void* image);

struct ModuleLoadOptions {
  // If set, the string will be resized automatically.
  std::string* info_log_buffer;
  // If set, the string will be resized automatically.
  std::string* error_log_buffer;

  std::optional<int> log_verbose;

  enum class FallbackStrategy { kPreferPtx, kPreferBinary };
  std::optional<FallbackStrategy> fallback_strategy;
};
llvm::Expected<OwningModule> ModuleLoadDataEx(CurrentContext current,
                                              const void* image,
                                              const ModuleLoadOptions& options);

llvm::Error ModuleUnload(Module module);
llvm::Expected<Function> ModuleGetFunction(Module module, const char* name);
llvm::Expected<MemoryRange<void>> ModuleGetGlobal(Module module,
                                                  const char* name);

llvm::Error LaunchKernel(CurrentContext current, Function function,
                         unsigned grid_dim_x, unsigned grid_dim_y,
                         unsigned grid_dim_z, unsigned block_dim_x,
                         unsigned block_dim_y, unsigned block_dim_z,
                         unsigned shared_memory_size_bytes, Stream stream,
                         llvm::ArrayRef<void*> arguments,
                         llvm::ArrayRef<void*> extras);
llvm::Error LaunchCooperativeKernel(CurrentContext current, Function function,
                                    unsigned grid_dim_x, unsigned grid_dim_y,
                                    unsigned grid_dim_z, unsigned block_dim_x,
                                    unsigned block_dim_y, unsigned block_dim_z,
                                    unsigned shared_memory_size_bytes,
                                    Stream stream,
                                    llvm::ArrayRef<void*> arguments);

llvm::Expected<int> OccupancyMaxActiveBlocksPerMultiprocessor(
    CurrentContext current, Function function, int block_size,
    size_t dynamic_shared_memory_size);
llvm::Expected<MaxPotentialBlockSize> OccupancyMaxPotentialBlockSize(
    CurrentContext current, Function function,
    size_t dynamic_shared_memory_size, int block_size_limit);

// Helper functions to allocate memory of type T. Does not perform any manual
// alignment.
template <typename T>
llvm::Expected<DeviceMemory<T>> MemAlloc(CurrentContext current, size_t count) {
  auto memory = MemAlloc(current, sizeof(T) * count);
  if (!memory) return memory.takeError();
  return DeviceMemory<T>(static_cast<Pointer<T>>(memory->release()));
}
template <typename T>
llvm::Expected<HostMemory<T>> MemHostAlloc(CurrentContext current, size_t count,
                                           MemHostAllocFlags flags) {
  auto memory = MemHostAlloc(current, sizeof(T) * count, flags);
  if (!memory) return memory.takeError();
  return HostMemory<T>(static_cast<Pointer<T>>(memory->release()));
}
template <typename T>
llvm::Expected<DeviceMemory<T>> MemAllocManaged(CurrentContext current,
                                                size_t count,
                                                MemAttachFlags flags) {
  auto memory = MemAllocManaged(current, sizeof(T) * count, flags);
  if (!memory) return memory.takeError();
  return DeviceMemory<T>(static_cast<Pointer<T>>(memory->release()));
}

// Helper functions to get a device pointer or address range of type T. The
// implementations forward to the corresponding overloads for Pointer<void>.
template <typename T>
llvm::Expected<Pointer<T>> MemHostGetDevicePointer(Pointer<T> host_ptr) {
  auto platform = host_ptr.platform();
  auto raw = const_cast<std::remove_cv_t<T*>>(host_ptr.raw(platform));
  auto result = MemHostGetDevicePointer(Pointer<void>(raw, platform));
  if (!result) return result.takeError();
  return static_cast<Pointer<T>>(*result);
}
template <typename T>
llvm::Expected<MemoryRange<T>> MemGetAddressRange(CurrentContext current,
                                                  Pointer<T> ptr) {
  auto platform = ptr.platform();
  auto raw = const_cast<std::remove_cv_t<T*>>(ptr.raw(platform));
  auto result = MemGetAddressRange(current, Pointer<void>(raw, platform));
  if (!result) return result.takeError();
  return MemoryRange<T>{static_cast<Pointer<T>>(result->base),
                        result->size_bytes};
}

// Helper function to launch kernels. Attention, no argument type checking is
// performed!
template <typename... Args>
llvm::Error LaunchKernel(CurrentContext current, Function function,
                         std::array<unsigned, 3> grid_dim,
                         std::array<unsigned, 3> block_dim,
                         unsigned shared_memory_size_bytes, Stream stream,
                         Args... arguments) {
  // Forward arguments as an array of pointers.
  std::array<void*, sizeof...(Args)> arg_ptrs = {{&arguments...}};
  return LaunchKernel(current, function, grid_dim[0], grid_dim[1], grid_dim[2],
                      block_dim[0], block_dim[1], block_dim[2],
                      shared_memory_size_bytes, stream, arg_ptrs,
                      llvm::ArrayRef<void*>{});
}

template <typename... Args>
llvm::Error LaunchCooperativeKernel(CurrentContext current, Function function,
                                    std::array<unsigned, 3> grid_dim,
                                    std::array<unsigned, 3> block_dim,
                                    unsigned shared_memory_size_bytes,
                                    Stream stream, Args... arguments) {
  // Forward arguments as an array of pointers.
  std::array<void*, sizeof...(Args)> arg_ptrs = {{&arguments...}};
  return LaunchCooperativeKernel(
      current, function, grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0],
      block_dim[1], block_dim[2], shared_memory_size_bytes, stream, arg_ptrs);
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_DRIVER_WRAPPER_H_
