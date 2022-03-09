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

// Thin abstraction layer for CUDA and HIP driver API.
#include "tfrt/gpu/wrapper/driver_wrapper.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/gpu/wrapper/hip_wrapper.h"
#include "tfrt/support/logging.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, Platform platform) {
  switch (platform) {
    case Platform::NONE:
      return os << "NONE";
    case Platform::CUDA:
      return os << "CUDA";
    case Platform::ROCm:
      return os << "ROCm";
    default:
      return os << llvm::formatv("Platform({0})", static_cast<int>(platform));
  }
}

Expected<wrapper::Platform> ParsePlatform(llvm::StringRef platform) {
  if (platform == "NONE") return wrapper::Platform::NONE;
  if (platform == "CUDA") return wrapper::Platform::CUDA;
  if (platform == "ROCm") return wrapper::Platform::ROCm;
  return MakeStringError("Invalid platform: ", platform);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, CurrentContext current) {
  return os << current.context();
}

#ifndef NDEBUG
CurrentContext::CurrentContext() { ++kContextTls.ref_count; }

CurrentContext::CurrentContext(const CurrentContext&) {
  ++kContextTls.ref_count;
}

CurrentContext::~CurrentContext() {
  assert(kContextTls.ref_count >= 0);
  --kContextTls.ref_count;
}
#endif

Context CurrentContext::context() const {
  auto platform = kContextTls.platform;
  switch (platform) {
    case Platform::CUDA:
      return {kContextTls.cuda_ctx};
    case Platform::ROCm:
      return {kContextTls.hip_ctx};
    default:
      return {nullptr, platform};
  }
}

Platform CurrentContext::platform() const { return kContextTls.platform; }

void internal::ContextDeleter::operator()(Context context) const {
  auto platform = context.platform();
  switch (platform) {
    case Platform::CUDA:
      // Try to destroy as ordinary context first. If that reports an invalid
      // context, release as device primary context. This extra bit of logic
      // doesn't hurt here because this function does not need to be fast. The
      // alternative of tracking whether a context is primary is more complex
      // and would need to be done when updating kContextTls and for the context
      // returned from CuStreamGetCtx(), but those should be lightweight.
      return LogIfError(llvm::handleErrors(
          CuCtxDestroy(context),
          [&](std::unique_ptr<ErrorInfo<CUresult>> info) {
            if (GetResult(*info) != CUDA_ERROR_INVALID_CONTEXT)
              return llvm::Error(std::move(info));
            auto device = CuCtxGetDevice(context);
            if (!device) return device.takeError();
            return CuDevicePrimaryCtxRelease({*device, Platform::CUDA});
          }));
    case Platform::ROCm:
      return LogIfError(llvm::handleErrors(
          HipCtxDestroy(context),
          [&](std::unique_ptr<ErrorInfo<hipError_t>> info) {
            if (GetResult(*info) != hipErrorInvalidContext)
              return llvm::Error(std::move(info));
            auto device = HipCtxGetDevice(context);
            if (!device) return device.takeError();
            return HipDevicePrimaryCtxRelease({*device, Platform::ROCm});
          }));
    default:
      return;
  }
}
void internal::ModuleDeleter::operator()(Module module) const {
  LogIfError(ModuleUnload(module));
}
void internal::StreamDeleter::operator()(Stream stream) const {
  LogIfError(StreamDestroy(stream));
}
void internal::EventDeleter::operator()(Event event) const {
  LogIfError(EventDestroy(event));
}
void internal::DeviceMemoryDeallocator::operator()(
    Pointer<void> pointer) const {
  LogIfError(MemFree(pointer));
}
void internal::HostMemoryDeallocator::operator()(Pointer<void> pointer) const {
  LogIfError(MemHostFree(pointer));
}
void internal::HostMemoryUnregisterer::operator()(Pointer<void> pointer) const {
  LogIfError(MemHostUnregister(pointer));
}

llvm::Error Init(Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CuInit();
    case Platform::ROCm:
      return HipInit();
    default:
      return llvm::Error::success();
  }
}

llvm::Expected<int> DriverGetVersion(Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CuDriverGetVersion();
    case Platform::ROCm:
      return HipDriverGetVersion();
    default:
      return 0;
  }
}

llvm::Expected<int> DeviceGetCount(Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CuDeviceGetCount();
    case Platform::ROCm:
      return HipDeviceGetCount();
    default:
      return 0;
  }
}

llvm::Expected<Device> DeviceGet(Platform platform, int ordinal) {
  switch (platform) {
    case Platform::CUDA:
      return CuDeviceGet(ordinal);
    case Platform::ROCm:
      return HipDeviceGet(ordinal);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<std::string> DeviceGetName(Device device) {
  auto platform = device.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuDeviceGetName(device);
    case Platform::ROCm:
      return HipDeviceGetName(device);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<size_t> DeviceTotalMem(Device device) {
  auto platform = device.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuDeviceTotalMem(device);
    case Platform::ROCm:
      return HipDeviceTotalMem(device);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<int> DeviceCanAccessPeer(Device device, Device peer_dev) {
  auto platform = device.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuDeviceCanAccessPeer(device, peer_dev);
    case Platform::ROCm:
      return HipDeviceCanAccessPeer(device, peer_dev);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningContext> DevicePrimaryCtxRetain(Device device) {
  auto platform = device.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuDevicePrimaryCtxRetain(device);
    case Platform::ROCm:
      return HipDevicePrimaryCtxRetain(device);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DevicePrimaryCtxRelease(Device device) {
  auto platform = device.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuDevicePrimaryCtxRelease(device);
    case Platform::ROCm:
      return HipDevicePrimaryCtxRelease(device);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DevicePrimaryCtxReset(Device device) {
  auto platform = device.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuDevicePrimaryCtxReset(device);
    case Platform::ROCm:
      return HipDevicePrimaryCtxReset(device);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<ContextState> DevicePrimaryCtxGetState(Device device) {
  auto platform = device.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuDevicePrimaryCtxGetState(device);
    case Platform::ROCm:
      return HipDevicePrimaryCtxGetState(device);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DevicePrimaryCtxSetFlags(Device device, ContextFlags flags) {
  auto platform = device.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuDevicePrimaryCtxSetFlags(device, flags);
    case Platform::ROCm:
      return HipDevicePrimaryCtxSetFlags(device, flags);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningContext> CtxCreate(Device device) {
  constexpr int flag = CU_CTX_SCHED_AUTO;
  static_assert(flag == hipDeviceScheduleAuto, "different value");
  return CtxCreate(ContextFlags(flag, device.platform()), device);
}

llvm::Expected<OwningContext> CtxCreate(ContextFlags flags, Device device) {
  auto platform = device.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuCtxCreate(flags, device);
    case Platform::ROCm:
      return HipCtxCreate(flags, device);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CtxDestroy(Context context) {
  auto platform = context.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuCtxDestroy(context);
    case Platform::ROCm:
      return HipCtxDestroy(context);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<CurrentContext> CtxGetCurrent() {
  // No need to call platform specific functions.
  return CreateCurrentContext();
}

llvm::Expected<CurrentContext> CtxSetCurrent(Context context) {
  auto platform = context.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuCtxSetCurrent(context);
    case Platform::ROCm:
      return HipCtxSetCurrent(context);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CtxSynchronize(CurrentContext current) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuCtxSynchronize(current);
    case Platform::ROCm:
      return HipDeviceSynchronize(current);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<unsigned> CtxGetApiVersion(Context context) {
  auto platform = context.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuCtxGetApiVersion(context);
    case Platform::ROCm:
      return HipCtxGetApiVersion(context);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<Device> CtxGetDevice(CurrentContext current) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuCtxGetDevice(current);
    case Platform::ROCm:
      return HipCtxGetDevice(current);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<ContextFlags> CtxGetFlags(CurrentContext current) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA: {
      auto result = CuCtxGetFlags(current);
      if (!result) return result.takeError();
      return static_cast<ContextFlags>(*result);
    }
    case Platform::ROCm: {
      auto result = HipCtxGetFlags(current);
      if (!result) return result.takeError();
      return static_cast<ContextFlags>(*result);
    }
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<StreamPriorityRange> CtxGetStreamPriorityRange(
    CurrentContext current) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuCtxGetStreamPriorityRange(current);
    case Platform::ROCm:
      return HipDeviceGetStreamPriorityRange(current);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningStream> StreamCreate(CurrentContext current,
                                          StreamFlags flags) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuStreamCreate(current, flags);
    case Platform::ROCm:
      return HipStreamCreate(current, flags);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningStream> StreamCreate(CurrentContext current,
                                          StreamFlags flags, int priority) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuStreamCreate(current, flags, priority);
    case Platform::ROCm:
      return HipStreamCreate(current, flags, priority);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningStream> StreamCreateNonBlocking(CurrentContext current) {
  constexpr int flag = CU_STREAM_NON_BLOCKING;
  static_assert(flag == hipStreamNonBlocking, "different value");
  return StreamCreate(current, StreamFlags(flag, current.platform()));
}

llvm::Error StreamDestroy(Stream stream) {
  auto platform = stream.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuStreamDestroy(stream);
    case Platform::ROCm:
      return HipStreamDestroy(stream);
    default:
      return llvm::Error::success();
  }
}

llvm::Expected<int> StreamGetPriority(Stream stream) {
  auto platform = stream.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuStreamGetPriority(stream);
    case Platform::ROCm:
      return HipStreamGetPriority(stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<StreamFlags> StreamGetFlags(Stream stream) {
  auto platform = stream.platform();
  switch (platform) {
    case Platform::CUDA: {
      auto result = CuStreamGetFlags(stream);
      if (!result) return result.takeError();
      return static_cast<StreamFlags>(*result);
    }
    case Platform::ROCm: {
      auto result = HipStreamGetFlags(stream);
      if (!result) return result.takeError();
      return static_cast<StreamFlags>(*result);
    }
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error StreamSynchronize(Stream stream) {
  auto platform = stream.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuStreamSynchronize(stream);
    case Platform::ROCm:
      return HipStreamSynchronize(stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<bool> StreamQuery(Stream stream) {
  auto platform = stream.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuStreamQuery(stream);
    case Platform::ROCm:
      return HipStreamQuery(stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error StreamWaitEvent(Stream stream, Event event) {
  auto platform = stream.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuStreamWaitEvent(stream, event);
    case Platform::ROCm:
      return HipStreamWaitEvent(stream, event);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningEvent> EventCreate(CurrentContext current,
                                        EventFlags flags) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuEventCreate(current, flags);
    case Platform::ROCm:
      return HipEventCreate(current, flags);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningEvent> EventCreateNoTiming(CurrentContext current) {
  constexpr int flag = CU_EVENT_DISABLE_TIMING;
  static_assert(flag == hipEventDisableTiming, "different value");
  return EventCreate(current, EventFlags(flag, current.platform()));
}

llvm::Error EventDestroy(Event event) {
  auto platform = event.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuEventDestroy(event);
    case Platform::ROCm:
      return HipEventDestroy(event);
    default:
      return llvm::Error::success();
  }
}

llvm::Error EventRecord(Event event, Stream stream) {
  auto platform = event.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuEventRecord(event, stream);
    case Platform::ROCm:
      return HipEventRecord(event, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error EventSynchronize(Event event) {
  auto platform = event.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuEventSynchronize(event);
    case Platform::ROCm:
      return HipEventSynchronize(event);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<bool> EventQuery(Event event) {
  auto platform = event.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuEventQuery(event);
    case Platform::ROCm:
      return HipEventQuery(event);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<float> EventElapsedTime(Event start, Event end) {
  auto platform = start.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuEventElapsedTime(start, end);
    case Platform::ROCm:
      return HipEventElapsedTime(start, end);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<DeviceMemory<void>> MemAlloc(CurrentContext current,
                                            size_t size_bytes) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemAlloc(current, size_bytes);
    case Platform::ROCm:
      return HipMemAlloc(current, size_bytes);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error MemFree(Pointer<void> pointer) {
  auto platform = pointer.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemFree(pointer);
    case Platform::ROCm:
      return HipMemFree(pointer);
    default:
      return llvm::Error::success();
  }
}

llvm::Expected<HostMemory<void>> MemHostAlloc(CurrentContext current,
                                              size_t size_bytes,
                                              MemHostAllocFlags flags) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemHostAlloc(current, size_bytes, flags);
    case Platform::ROCm:
      return HipMemHostAlloc(current, size_bytes, flags);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<HostMemory<void>> MemHostAllocWriteCombined(
    CurrentContext current, size_t size_bytes) {
  // Write-combined makes only sense if memory is also GPU accessible.
  // Besides, GPUs all use UVA these days and this makes DEVICEMAP a no-op.
  constexpr int dm_flag = CU_MEMHOSTALLOC_DEVICEMAP;
  constexpr int wc_flag = CU_MEMHOSTALLOC_WRITECOMBINED;
  static_assert(dm_flag == hipHostMallocMapped, "different value");
  static_assert(wc_flag == hipHostMallocWriteCombined, "different value");
  return wrapper::MemHostAlloc(
      current, size_bytes,
      MemHostAllocFlags(dm_flag | wc_flag, current.platform()));
}

llvm::Error MemHostFree(Pointer<void> pointer) {
  auto platform = pointer.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemHostFree(pointer);
    case Platform::ROCm:
      return HipMemHostFree(pointer);
    default:
      return llvm::Error::success();
  }
}

llvm::Expected<RegisteredMemory<void>> MemHostRegister(
    CurrentContext current, void* ptr, size_t size_bytes,
    MemHostRegisterFlags flags) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemHostRegister(current, ptr, size_bytes, flags);
    case Platform::ROCm:
      return HipMemHostRegister(current, ptr, size_bytes, flags);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<RegisteredMemory<void>> MemHostRegister(CurrentContext current,
                                                       void* ptr,
                                                       size_t size_bytes) {
  constexpr int flags = CU_MEMHOSTREGISTER_DEVICEMAP;
  static_assert(flags == hipHostRegisterMapped, "different value");
  return MemHostRegister(
      current, ptr, size_bytes,
      wrapper::MemHostRegisterFlags(flags, current.platform()));
}

llvm::Error MemHostUnregister(Pointer<void> pointer) {
  auto platform = pointer.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemHostUnregister(pointer);
    case Platform::ROCm:
      return HipMemHostUnregister(pointer);
    default:
      return llvm::Error::success();
  }
}

llvm::Expected<DeviceMemory<void>> MemAllocManaged(CurrentContext current,
                                                   size_t size_bytes,
                                                   MemAttachFlags flags) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemAllocManaged(current, size_bytes, flags);
    case Platform::ROCm:
      return HipMemAllocManaged(current, size_bytes, flags);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<Pointer<void>> MemHostGetDevicePointer(Pointer<void> host_ptr) {
  auto platform = host_ptr.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemHostGetDevicePointer(host_ptr);
    case Platform::ROCm:
      return HipMemHostGetDevicePointer(host_ptr);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<MemoryRange<void>> MemGetAddressRange(CurrentContext current,
                                                     Pointer<void> ptr) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemGetAddressRange(current, ptr);
    case Platform::ROCm:
      return HipMemGetAddressRange(current, ptr);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<MemoryInfo> MemGetInfo(CurrentContext current) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemGetInfo(current);
    case Platform::ROCm:
      return HipMemGetInfo(current);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error Memcpy(CurrentContext current, Pointer<void> dst,
                   Pointer<const void> src, size_t count_bytes) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemcpy(current, dst, src, count_bytes);
    case Platform::ROCm:
      return HipMemcpy(current, dst, src, count_bytes);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error MemcpyAsync(CurrentContext current, Pointer<void> dst,
                        Pointer<const void> src, size_t count_bytes,
                        Stream stream) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemcpyAsync(current, dst, src, count_bytes, stream);
    case Platform::ROCm:
      return HipMemcpyAsync(current, dst, src, count_bytes, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error MemcpyPeer(Pointer<void> dst_ptr, Context dst_ctx,
                       Pointer<const void> src_ptr, Context src_ctx,
                       size_t count_bytes) {
  auto platform = dst_ptr.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemcpyPeer(dst_ptr, dst_ctx, src_ptr, src_ctx, count_bytes);
    case Platform::ROCm: {
      auto dst_dev = HipCtxGetDevice(static_cast<hipCtx_t>(dst_ctx));
      if (!dst_dev) return dst_dev.takeError();
      auto src_dev = HipCtxGetDevice(static_cast<hipCtx_t>(src_ctx));
      if (!src_dev) return src_dev.takeError();
      return HipMemcpyPeer(dst_ptr, {*dst_dev, Platform::ROCm}, src_ptr,
                           {*src_dev, Platform::ROCm}, count_bytes);
    }
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error MemcpyPeerAsync(Pointer<void> dst_ptr, Context dst_ctx,
                            Pointer<const void> src_ptr, Context src_ctx,
                            size_t count_bytes, Stream stream) {
  auto platform = dst_ptr.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemcpyPeerAsync(dst_ptr, dst_ctx, src_ptr, src_ctx, count_bytes,
                               stream);
    case Platform::ROCm: {
      auto dst_dev = HipCtxGetDevice(static_cast<hipCtx_t>(dst_ctx));
      if (!dst_dev) return dst_dev.takeError();
      auto src_dev = HipCtxGetDevice(static_cast<hipCtx_t>(src_ctx));
      if (!src_dev) return src_dev.takeError();
      return HipMemcpyPeerAsync(dst_ptr, {*dst_dev, Platform::ROCm}, src_ptr,
                                {*src_dev, Platform::ROCm}, count_bytes,
                                stream);
    }
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error MemsetD8(CurrentContext current, Pointer<void> dst, std::uint8_t uc,
                     size_t count) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemsetD8(current, dst, uc, count);
    case Platform::ROCm:
      return HipMemsetD8(current, dst, uc, count);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error MemsetD32(CurrentContext current, Pointer<void> dst,
                      std::uint32_t value, size_t count) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemsetD32(current, dst, value, count);
    case Platform::ROCm:
      return HipMemsetD32(current, dst, value, count);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error MemsetD8Async(CurrentContext current, Pointer<void> dst,
                          std::uint8_t value, size_t count, Stream stream) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemsetD8Async(current, dst, value, count, stream);
    case Platform::ROCm:
      return HipMemsetD8Async(current, dst, value, count, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error MemsetD32Async(CurrentContext current, Pointer<void> dst,
                           std::uint32_t value, size_t count, Stream stream) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuMemsetD32Async(current, dst, value, count, stream);
    case Platform::ROCm:
      return HipMemsetD32Async(current, dst, value, count, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningModule> ModuleLoadData(CurrentContext current,
                                            const void* image) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuModuleLoadData(current, image);
    case Platform::ROCm:
      return HipRTCModuleLoadData(current, image);
    default:
      return InvalidPlatform(platform);
  }
}

namespace {

template <Platform kPlatform>
constexpr std::conditional_t<kPlatform == Platform::CUDA, CUjit_option,
                             hipJitOption>
EnumSwitch(CUjit_option cuda_opt, hipJitOption hip_opt);

template <>
constexpr CUjit_option EnumSwitch<Platform::CUDA>(CUjit_option cuda_opt,
                                                  hipJitOption hip_opt) {
  return cuda_opt;
}

template <>
constexpr hipJitOption EnumSwitch<Platform::ROCm>(CUjit_option cuda_opt,
                                                  hipJitOption hip_opt) {
  return hip_opt;
}

template <Platform kPlatform>
class JitOptions {
 private:
  static_assert(kPlatform != Platform::NONE,
                "NONE platform cannot have JIT options.");

 public:
  using OptionType = std::conditional_t<kPlatform == Platform::CUDA,
                                        CUjit_option, hipJitOption>;

  static constexpr OptionType kInfoLogBuffer =
      EnumSwitch<kPlatform>(CU_JIT_INFO_LOG_BUFFER, hipJitOptionInfoLogBuffer);
  static constexpr OptionType kInfoLogBufferSize = EnumSwitch<kPlatform>(
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, hipJitOptionInfoLogBufferSizeBytes);
  static constexpr OptionType kErrorLogBuffer = EnumSwitch<kPlatform>(
      CU_JIT_ERROR_LOG_BUFFER, hipJitOptionErrorLogBuffer);
  static constexpr OptionType kErrorLogBufferSize = EnumSwitch<kPlatform>(
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, hipJitOptionErrorLogBufferSizeBytes);
  static constexpr OptionType kLogVerbose =
      EnumSwitch<kPlatform>(CU_JIT_LOG_VERBOSE, hipJitOptionLogVerbose);
  static constexpr OptionType kFallbackStrategy = EnumSwitch<kPlatform>(
      CU_JIT_FALLBACK_STRATEGY, hipJitOptionFallbackStrategy);
};

// Avoid C++14 linkage issues by explicitly defining the values outside the
// class.
// TODO(imintz): Replace with inline constants when c++17.
template <Platform P>
constexpr typename JitOptions<P>::OptionType JitOptions<P>::kInfoLogBuffer;
template <Platform P>
constexpr typename JitOptions<P>::OptionType JitOptions<P>::kInfoLogBufferSize;
template <Platform P>
constexpr typename JitOptions<P>::OptionType JitOptions<P>::kErrorLogBuffer;
template <Platform P>
constexpr typename JitOptions<P>::OptionType JitOptions<P>::kErrorLogBufferSize;
template <Platform P>
constexpr typename JitOptions<P>::OptionType JitOptions<P>::kLogVerbose;
template <Platform P>
constexpr typename JitOptions<P>::OptionType JitOptions<P>::kFallbackStrategy;

template <Platform kPlatform>
class RawModuleLoadOptions {
 public:
  explicit RawModuleLoadOptions(const ModuleLoadOptions& in_options) {
    // Sufficiently large  buffer size used for recording info/error logs.
    constexpr size_t kBufferSize = 4096;

    if (in_options.info_log_buffer != nullptr) {
      options_.push_back(JitOptions<kPlatform>::kInfoLogBufferSize);
      // CUDA writes the number of bytes written back into the options array, so
      // record the position of the size value for write-back on cleanup.
      info_log_buffer_size_opt_index_ = option_values_.size();
      option_values_.push_back(reinterpret_cast<void*>(kBufferSize));

      info_log_ = in_options.info_log_buffer;
      info_log_->resize(kBufferSize);
      options_.push_back(JitOptions<kPlatform>::kInfoLogBuffer);
      option_values_.push_back(const_cast<char*>(info_log_->data()));
    }
    if (in_options.error_log_buffer != nullptr) {
      options_.push_back(JitOptions<kPlatform>::kErrorLogBufferSize);
      error_log_buffer_size_opt_index_ = option_values_.size();
      option_values_.push_back(reinterpret_cast<void*>(kBufferSize));

      error_log_ = in_options.error_log_buffer;
      error_log_->resize(kBufferSize);
      options_.push_back(JitOptions<kPlatform>::kErrorLogBuffer);
      option_values_.push_back(const_cast<char*>(error_log_->data()));
    }
    if (in_options.log_verbose) {
      options_.push_back(JitOptions<kPlatform>::kLogVerbose);
      option_values_.push_back(
          const_cast<int*>(in_options.log_verbose.getPointer()));
    }

    if (in_options.fallback_strategy) {
      options_.push_back(JitOptions<kPlatform>::kFallbackStrategy);
      option_values_.push_back(
          reinterpret_cast<void*>(*in_options.fallback_strategy));
    }
  }

  ~RawModuleLoadOptions() { MaybeUpdateLogBufferLengths(); }

  using OptionT = std::conditional_t<kPlatform == Platform::CUDA, CUjit_option,
                                     hipJitOption>;

  llvm::ArrayRef<OptionT> options() const { return options_; }
  llvm::ArrayRef<void*> option_values() const { return option_values_; }

 private:
  // The CUDA interface writes the log output size at the address of the input
  // length. This method is used to mask this interface and make it appear more
  // sane.
  void MaybeUpdateLogBufferLengths() {
    if (info_log_) {
      info_log_->resize(reinterpret_cast<size_t>(
          option_values_[info_log_buffer_size_opt_index_]));
    }
    if (error_log_) {
      error_log_->resize(reinterpret_cast<size_t>(
          option_values_[error_log_buffer_size_opt_index_]));
    }
  }

  llvm::SmallVector<OptionT, 4> options_;
  llvm::SmallVector<void*, 4> option_values_;

  std::string* info_log_;
  int info_log_buffer_size_opt_index_ = -1;
  std::string* error_log_;
  int error_log_buffer_size_opt_index_ = -1;
};

}  // namespace

llvm::Expected<OwningModule> ModuleLoadDataEx(
    CurrentContext current, const void* image,
    const ModuleLoadOptions& options) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA: {
      RawModuleLoadOptions<Platform::CUDA> parsed_opts(options);
      return CuModuleLoadDataEx(current, image, parsed_opts.options(),
                                parsed_opts.option_values());
    }
    case Platform::ROCm: {
      RawModuleLoadOptions<Platform::ROCm> parsed_opts(options);
      return HipModuleLoadDataEx(current, image, parsed_opts.options(),
                                 parsed_opts.option_values());
    }
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error ModuleUnload(Module module) {
  auto platform = module.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuModuleUnload(module);
    case Platform::ROCm:
      return HipModuleUnload(module);
    default:
      return llvm::Error::success();
  }
}

llvm::Expected<Function> ModuleGetFunction(Module module, const char* name) {
  auto platform = module.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuModuleGetFunction(module, name);
    case Platform::ROCm:
      return HipModuleGetFunction(module, name);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<MemoryRange<void>> ModuleGetGlobal(Module module,
                                                  const char* name) {
  auto platform = module.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuModuleGetGlobal(module, name);
    case Platform::ROCm:
      return HipModuleGetGlobal(module, name);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error LaunchKernel(CurrentContext current, Function function,
                         unsigned grid_dim_x, unsigned grid_dim_y,
                         unsigned grid_dim_z, unsigned block_dim_x,
                         unsigned block_dim_y, unsigned block_dim_z,
                         unsigned shared_memory_size_bytes, Stream stream,
                         llvm::ArrayRef<void*> arguments,
                         llvm::ArrayRef<void*> extras) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuLaunchKernel(current, function, grid_dim_x, grid_dim_y,
                            grid_dim_z, block_dim_x, block_dim_y, block_dim_z,
                            shared_memory_size_bytes, stream, arguments,
                            extras);
    case Platform::ROCm:
      return HipLaunchKernel(current, function, grid_dim_x, grid_dim_y,
                             grid_dim_z, block_dim_x, block_dim_y, block_dim_z,
                             shared_memory_size_bytes, stream, arguments,
                             extras);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error LaunchCooperativeKernel(CurrentContext current, Function function,
                                    unsigned grid_dim_x, unsigned grid_dim_y,
                                    unsigned grid_dim_z, unsigned block_dim_x,
                                    unsigned block_dim_y, unsigned block_dim_z,
                                    unsigned shared_memory_size_bytes,
                                    Stream stream,
                                    llvm::ArrayRef<void*> arguments) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuLaunchCooperativeKernel(
          current, function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
          block_dim_y, block_dim_z, shared_memory_size_bytes, stream,
          arguments);
    case Platform::ROCm:
      return HipLaunchCooperativeKernel(
          current, function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
          block_dim_y, block_dim_z, shared_memory_size_bytes, stream,
          arguments);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<int> OccupancyMaxActiveBlocksPerMultiprocessor(
    CurrentContext current, Function function, int block_size,
    size_t dynamic_shared_memory_size) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuOccupancyMaxActiveBlocksPerMultiprocessor(
          current, function, block_size, dynamic_shared_memory_size);
    case Platform::ROCm:
      return HipOccupancyMaxActiveBlocksPerMultiprocessor(
          current, function, block_size, dynamic_shared_memory_size);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<MaxPotentialBlockSize> OccupancyMaxPotentialBlockSize(
    CurrentContext current, Function function,
    size_t dynamic_shared_memory_size, int block_size_limit) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CuOccupancyMaxPotentialBlockSize(
          current, function, [&](int) { return dynamic_shared_memory_size; },
          block_size_limit);
    case Platform::ROCm:
      return HipOccupancyMaxPotentialBlockSize(
          current, function, dynamic_shared_memory_size, block_size_limit);
    default:
      return InvalidPlatform(platform);
  }
}

// Definitions from wrapper_detail.h.

thread_local ContextTls kContextTls;

llvm::Error CheckNoCurrentContext() {
#ifndef NDEBUG
  if (kContextTls.ref_count != 0) {
    return MakeStringError(
        "Existing CurrentContext instance(s) in same thread.");
  }
#endif
  return llvm::Error::success();
}

struct CurrentContext::Factory {
  static CurrentContext Create() { return CurrentContext(); }
};
CurrentContext CreateCurrentContext() {
  return CurrentContext::Factory::Create();
}

static void LogIfErrorImpl(llvm::Error error, Severity severity) {
  llvm::handleAllErrors(std::move(error), [&](const llvm::ErrorInfoBase& info) {
    tfrt::internal::LogStream(__FILE__, __LINE__, severity) << info.message();
  });
}

void LogIfError(llvm::Error&& error) {
  LogIfErrorImpl(std::move(error), Severity::ERROR);
}

void DieIfError(llvm::Error&& error) {
  LogIfErrorImpl(std::move(error), Severity::FATAL);
}

llvm::Error CheckPlatform(Platform platform, Platform expected) {
  if (platform != expected) {
    return MakeStringError(llvm::formatv(
        "Expected platform to be {0}, but got {1}", expected, platform));
  }
  return llvm::Error::success();
}

static llvm::Error CreatePlatformError(const char* cond, Platform platform) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      llvm::formatv("{0} platform {1}", cond, platform));
}

llvm::Error InvalidPlatform(Platform platform) {
  return CreatePlatformError("Invalid", platform);
}

llvm::Error UnsupportedPlatform(Platform platform) {
  return CreatePlatformError("Unsupported", platform);
}

llvm::Error MakeOomError(CurrentContext current, size_t size_bytes) {
  std::string message;
  llvm::raw_string_ostream oss(message);
  oss << "Out of memory trying to allocate "
      << HumanReadableNumBytes(size_bytes);
  if (auto mem_info = wrapper::MemGetInfo(current)) {
    oss << " (" << HumanReadableNumBytes(mem_info->free_bytes) << " of "
        << HumanReadableNumBytes(mem_info->total_bytes) << " available)";
  }
  if (auto device = wrapper::CtxGetDevice(current))
    oss << " on GPU " << *device;
  oss << ".";
  return MakeStringError(std::move(oss.str()));
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ResourceType type) {
  switch (type) {
    case ResourceType::kStream:
      return os << "stream";
    case ResourceType::kEvent:
      return os << "event";
    case ResourceType::kModule:
      return os << "module";
    case ResourceType::kDeviceMemory:
      return os << "device memory";
    case ResourceType::kHostMemory:
      return os << "host memory";
    case ResourceType::kRegisteredMemory:
      return os << "registered memory";
    default:
      return os << llvm::formatv("ResourceType({0})", static_cast<int>(type));
  }
}

ResourceMap::ResourceMap(Map* map, std::mutex* mutex)
    : map_(map), lock_(*mutex) {}

ResourceMap ResourceMap::Get() {
  static std::pair<Map*, std::mutex*> pair = {new Map, new std::mutex};
  return {pair.first, pair.second};
}

void ResourceMap::NotifyCreated(ResourceType type, void* resource) {
  auto context = CreateCurrentContext().context();
  if (!map_->emplace(resource, std::make_pair(type, context)).second)
    TFRT_LOG(FATAL) << StrCat("Resource ", resource, " already registered");
}

void ResourceMap::NotifyDestroyed(void* resource) {
  if (!map_->erase(resource))
    TFRT_LOG(FATAL) << StrCat("Resource ", resource, " not registered");
}

llvm::Error ResourceMap::CheckNoneDangling(Context context) {
  std::vector<Map::iterator> iters;
  for (auto it = map_->begin(); it != map_->end(); ++it) {
    if (std::get<Context>(it->second) == context) {
      iters.push_back(it);
    }
  }
  if (iters.empty()) return llvm::Error::success();

  std::string message =
      StrCat("Context ", context, " has dangling resources: ");
  for (auto it : iters) {
    llvm::raw_string_ostream(message)
        << it->first << " (" << std::get<ResourceType>(it->second) << "), ";
    map_->erase(it);
  }
  return MakeStringError(llvm::StringRef(message).drop_back(2));
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
