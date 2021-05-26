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

// Thin wrapper around the CUDA API adding llvm::Error and explicit context.
#include "tfrt/gpu/wrapper/cuda_wrapper.h"

#include "llvm/Support/FormatVariadic.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

template llvm::raw_ostream& internal::operator<<(llvm::raw_ostream&,
                                                 const ErrorData<CUresult>&);

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, CUresult result) {
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  if (name != nullptr) {
    os << name;
  } else {
    os << llvm::formatv("CUresult({0})", static_cast<int>(result));
  }
  const char* msg = nullptr;
  cuGetErrorString(result, &msg);
  if (msg != nullptr) os << " (" << msg << ")";
  return os;
}

// Convert wrapper types to CUDA types.
static CUdevice ToCuda(Device device) { return device.id(Platform::CUDA); }

template <typename T>
static CUdeviceptr ToDevicePtr(Pointer<T> ptr) {
  return reinterpret_cast<CUdeviceptr>(ToCuda(ptr));
}

llvm::Error CuInit() { return TO_ERROR(cuInit(/*flags=*/0)); }

llvm::Expected<int> CuDriverGetVersion() {
  int version;
  RETURN_IF_ERROR(cuDriverGetVersion(&version));
  return version;
}

llvm::Expected<int> CuDeviceGetCount() {
  int count;
  RETURN_IF_ERROR(cuDeviceGetCount(&count));
  return count;
}

llvm::Expected<Device> CuDeviceGet(int ordinal) {
  CUdevice device;
  RETURN_IF_ERROR(cuDeviceGet(&device, ordinal));
  return Device(device, Platform::CUDA);
}

llvm::Expected<std::string> CuDeviceGetName(Device device) {
  char name[100];
  RETURN_IF_ERROR(cuDeviceGetName(name, sizeof(name), ToCuda(device)));
  return std::string(name);
}

llvm::Expected<size_t> CuDeviceTotalMem(Device device) {
  size_t size_bytes;
  RETURN_IF_ERROR(cuDeviceTotalMem(&size_bytes, ToCuda(device)));
  return size_bytes;
}

llvm::Expected<int> CuDeviceGetAttribute(CUdevice_attribute attribute,
                                         Device device) {
  int value;
  RETURN_IF_ERROR(cuDeviceGetAttribute(&value, attribute, ToCuda(device)));
  return value;
}

llvm::Expected<int> CuDeviceCanAccessPeer(Device src_dev, Device dst_dev) {
  int result;
  RETURN_IF_ERROR(
      cuDeviceCanAccessPeer(&result, ToCuda(src_dev), ToCuda(dst_dev)));
  return result;
}

llvm::Expected<int> CuDeviceGetP2PAttribute(CUdevice_P2PAttribute attribute,
                                            Device src_dev, Device dst_dev) {
  int result;
  RETURN_IF_ERROR(cuDeviceGetP2PAttribute(&result, attribute, ToCuda(src_dev),
                                          ToCuda(dst_dev)));
  return result;
}

llvm::Expected<OwningContext> CuDevicePrimaryCtxRetain(Device device) {
  CUcontext cuda_ctx;
  RETURN_IF_ERROR(cuDevicePrimaryCtxRetain(&cuda_ctx, ToCuda(device)));
  kContextTls.cuda_may_skip_set_ctx = true;
  return OwningContext(cuda_ctx);
}

llvm::Error CuDevicePrimaryCtxRelease(Device device) {
  if (auto has_instance = CheckNoCurrentContext()) return has_instance;
  RETURN_IF_ERROR(cuDevicePrimaryCtxRelease(ToCuda(device)));
  // Releasing the primary context does not change the current context, but
  // decrements the internal reference count and deactivates it iff zero.
  kContextTls.cuda_may_skip_set_ctx = false;
#ifndef NDEBUG
  auto state = CuDevicePrimaryCtxGetState(device);
  if (!state) return state.takeError();
  if (!state->active) {
    auto context = CuDevicePrimaryCtxRetain(device);
    if (!context) return context.takeError();
    RETURN_IF_ERROR(cuDevicePrimaryCtxRelease(ToCuda(device)));
    return CheckNoDanglingResources(context->release());
  }
#endif
  return llvm::Error::success();
}

llvm::Error CuDevicePrimaryCtxReset(Device device) {
  if (kContextTls.platform == Platform::CUDA) {
    if (auto has_instance = CheckNoCurrentContext()) {
      // There is a CurrentContext instance, check that primary is not current.
      CUcontext context;
      RETURN_IF_ERROR(cuCtxGetCurrent(&context));
      if (kContextTls.cuda_ctx == context) return has_instance;
    }
  }
  Context context;
#ifndef NDEBUG
  auto context_or = CuDevicePrimaryCtxRetain(device);
  if (!context_or) return context_or.takeError();
  context = context_or->release();
  RETURN_IF_ERROR(cuDevicePrimaryCtxRelease(ToCuda(device)));
#endif
  RETURN_IF_ERROR(cuDevicePrimaryCtxReset(ToCuda(device)));
  return CheckNoDanglingResources(context);
}

llvm::Expected<ContextState> CuDevicePrimaryCtxGetState(Device device) {
  unsigned flags;
  int active;
  RETURN_IF_ERROR(cuDevicePrimaryCtxGetState(ToCuda(device), &flags, &active));
  return ContextState{static_cast<CtxFlags>(flags), active};
}

llvm::Error CuDevicePrimaryCtxSetFlags(Device device, CUctx_flags flags) {
  return TO_ERROR(cuDevicePrimaryCtxSetFlags(ToCuda(device), flags));
}

llvm::Expected<OwningContext> CuCtxCreate(CUctx_flags flags, Device device) {
  // Check no instance of CurrentContext exists in this thread.
  if (auto has_instance = CheckNoCurrentContext())
    return std::move(has_instance);
  RETURN_IF_ERROR(cuCtxCreate(&kContextTls.cuda_ctx, flags, ToCuda(device)));
  kContextTls.platform = Platform::CUDA;
  kContextTls.cuda_may_skip_set_ctx = true;
  return OwningContext(kContextTls.cuda_ctx);
}

llvm::Error CuCtxDestroy(CUcontext context) {
  if (context == nullptr) return llvm::Error::success();
  if (kContextTls.cuda_ctx == context &&
      kContextTls.platform == Platform::CUDA) {
    // Check that there is no CurrentContext instance.
    if (auto has_instance = CheckNoCurrentContext()) return has_instance;
  }
  RETURN_IF_ERROR(cuCtxDestroy(context));
  if (kContextTls.cuda_ctx == context) {
    // Destroying the current context makes the primary context current if there
    // is one for the same device. The primary context may be inactive.
    RETURN_IF_ERROR(cuCtxGetCurrent(&kContextTls.cuda_ctx));
    kContextTls.cuda_may_skip_set_ctx = false;
  }
  return CheckNoDanglingResources(context);
}

llvm::Expected<CurrentContext> CuCtxGetCurrent() {
  return CreateCurrentContext();
}

llvm::Expected<CurrentContext> CuCtxSetCurrent(CUcontext context) {
  // Check no instance of CurrentContext exists in this thread.
  if (auto has_instance = CheckNoCurrentContext())
    return std::move(has_instance);
  // Skip setting context if it's already current. This is an optimization
  // that requires users to not change the current context through the CUDA
  // API. This is validated through calls to CheckCudaContext().
  if (kContextTls.cuda_ctx != context || !kContextTls.cuda_may_skip_set_ctx) {
    RETURN_IF_ERROR(cuCtxSetCurrent(context));
    if (context != nullptr) {
      kContextTls.cuda_may_skip_set_ctx = true;
    } else {
      // Setting the null context makes the primary context current if there
      // is one for the current device. The primary context may be inactive.
      RETURN_IF_ERROR(cuCtxGetCurrent(&context));
      kContextTls.cuda_may_skip_set_ctx = (context == nullptr);
    }
    kContextTls.cuda_ctx = context;
  }
  kContextTls.platform = Platform::CUDA;
  auto current = CreateCurrentContext();
  // Catch false skipping of setting context above.
  CheckCudaContext(current);
  return current;
}

llvm::Error CuCtxSynchronize(CurrentContext current) {
  CheckCudaContext(current);
  return TO_ERROR(cuCtxSynchronize());
}

llvm::Expected<unsigned> CuCtxGetApiVersion(CUcontext context) {
  unsigned version;
  RETURN_IF_ERROR(cuCtxGetApiVersion(context, &version));
  return version;
}

llvm::Expected<Device> CuCtxGetDevice(CurrentContext current) {
  CheckCudaContext(current);
  CUdevice device;
  RETURN_IF_ERROR(cuCtxGetDevice(&device));
  return Device(device, Platform::CUDA);
}

llvm::Expected<CUctx_flags> CuCtxGetFlags(CurrentContext current) {
  CheckCudaContext(current);
  unsigned flags;
  RETURN_IF_ERROR(cuCtxGetFlags(&flags));
  return static_cast<CUctx_flags>(flags);
}

llvm::Expected<StreamPriorityRange> CuCtxGetStreamPriorityRange(
    CurrentContext current) {
  CheckCudaContext(current);
  StreamPriorityRange priority_range;
  RETURN_IF_ERROR(cuCtxGetStreamPriorityRange(&priority_range.least,
                                              &priority_range.greatest));
  return priority_range;
}

llvm::Expected<size_t> CuCtxGetLimit(CurrentContext current, CUlimit limit) {
  CheckCudaContext(current);
  size_t value;
  RETURN_IF_ERROR(cuCtxGetLimit(&value, limit));
  return value;
}

llvm::Error CuCtxSetLimit(CurrentContext current, CUlimit limit, size_t value) {
  CheckCudaContext(current);
  return TO_ERROR(cuCtxSetLimit(limit, value));
}

llvm::Expected<CUfunc_cache> CuCtxGetCacheConfig(CurrentContext current) {
  CheckCudaContext(current);
  CUfunc_cache config;
  RETURN_IF_ERROR(cuCtxGetCacheConfig(&config));
  return config;
}

llvm::Error CuCtxSetCacheConfig(CurrentContext current, CUfunc_cache config) {
  CheckCudaContext(current);
  return TO_ERROR(cuCtxSetCacheConfig(config));
}

llvm::Expected<CUsharedconfig> CuCtxGetSharedMemConfig(CurrentContext current) {
  CheckCudaContext(current);
  CUsharedconfig config;
  RETURN_IF_ERROR(cuCtxGetSharedMemConfig(&config));
  return config;
}

llvm::Error CuCtxSetSharedMemConfig(CurrentContext current,
                                    CUsharedconfig config) {
  CheckCudaContext(current);
  return TO_ERROR(cuCtxSetSharedMemConfig(config));
}

llvm::Error CuCtxEnablePeerAccess(CurrentContext current,
                                  CUcontext peer_context) {
  CheckCudaContext(current);
  return TO_ERROR(cuCtxEnablePeerAccess(peer_context, /*flags=*/0));
}

llvm::Error CuCtxDisablePeerAccess(CurrentContext current,
                                   CUcontext peer_context) {
  CheckCudaContext(current);
  return TO_ERROR(cuCtxDisablePeerAccess(peer_context));
}

llvm::Expected<OwningStream> CuStreamCreate(CurrentContext current,
                                            CUstream_flags flags) {
  CheckCudaContext(current);
  CUstream stream;
  RETURN_IF_ERROR(cuStreamCreate(&stream, flags));
  NotifyResourceCreated(ResourceType::kStream, stream);
  return OwningStream(stream);
}

llvm::Expected<OwningStream> CuStreamCreate(CurrentContext current,
                                            CUstream_flags flags,
                                            int priority) {
  CheckCudaContext(current);
  CUstream stream;
  RETURN_IF_ERROR(cuStreamCreateWithPriority(&stream, flags, priority));
  NotifyResourceCreated(ResourceType::kStream, stream);
  return OwningStream(stream);
}

llvm::Error CuStreamDestroy(CUstream stream) {
  if (stream == nullptr) return llvm::Error::success();
  RETURN_IF_ERROR(cuStreamDestroy(stream));
  NotifyResourceDestroyed(stream);
  return llvm::Error::success();
}

llvm::Expected<Context> CuStreamGetCtx(CUstream stream) {
  CUcontext context;
  RETURN_IF_ERROR(cuStreamGetCtx(stream, &context));
  return context;
}

llvm::Expected<CUstream_flags> CuStreamGetFlags(CUstream stream) {
  unsigned flags;
  RETURN_IF_ERROR(cuStreamGetFlags(stream, &flags));
  return static_cast<CUstream_flags>(flags);
}

llvm::Expected<int> CuStreamGetPriority(CUstream stream) {
  int priority;
  RETURN_IF_ERROR(cuStreamGetPriority(stream, &priority));
  return priority;
}

llvm::Error CuStreamSynchronize(CUstream stream) {
  return TO_ERROR(cuStreamSynchronize(stream));
}

llvm::Expected<bool> CuStreamQuery(CUstream stream) {
  auto result = cuStreamQuery(stream);
  if (result == CUDA_ERROR_NOT_READY) {
    return false;
  }
  RETURN_IF_ERROR(result);
  return true;
}

llvm::Error CuStreamWaitEvent(CUstream stream, CUevent event) {
  return TO_ERROR(cuStreamWaitEvent(stream, event, /*flags=*/0));
}

llvm::Expected<OwningEvent> CuEventCreate(CurrentContext current,
                                          CUevent_flags flags) {
  CheckCudaContext(current);
  CUevent event;
  RETURN_IF_ERROR(cuEventCreate(&event, flags));
  NotifyResourceCreated(ResourceType::kEvent, event);
  return OwningEvent(event);
}

llvm::Error CuEventDestroy(CUevent event) {
  if (event == nullptr) return llvm::Error::success();
  RETURN_IF_ERROR(cuEventDestroy(event));
  NotifyResourceDestroyed(event);
  return llvm::Error::success();
}

llvm::Error CuEventRecord(CUevent event, CUstream stream) {
  return TO_ERROR(cuEventRecord(event, stream));
}

llvm::Error CuEventSynchronize(CUevent event) {
  return TO_ERROR(cuEventSynchronize(event));
}

llvm::Expected<bool> CuEventQuery(CUevent event) {
  auto result = cuEventQuery(event);
  if (result == CUDA_ERROR_NOT_READY) {
    return false;
  }
  RETURN_IF_ERROR(result);
  return true;
}

llvm::Expected<float> CuEventElapsedTime(CUevent start, CUevent end) {
  float time_ms;
  RETURN_IF_ERROR(cuEventElapsedTime(&time_ms, start, end));
  return time_ms;
}

llvm::Expected<DeviceMemory<void>> CuMemAlloc(CurrentContext current,
                                              size_t size_bytes) {
  CheckCudaContext(current);
  void* ptr;
  if (auto error = TO_ERROR(
          cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&ptr), size_bytes))) {
    return llvm::handleErrors(
        std::move(error), [&](std::unique_ptr<ErrorInfo<CUresult>> info) {
          return GetResult(*info) == CUDA_ERROR_OUT_OF_MEMORY
                     ? MakeOomError(current, size_bytes)
                     : llvm::Error(std::move(info));
        });
  }
  NotifyResourceCreated(ResourceType::kDeviceMemory, ptr);
  return DeviceMemory<void>({ptr, Platform::CUDA});
}

llvm::Error CuMemFree(Pointer<void> pointer) {
  RETURN_IF_ERROR(cuMemFree(ToDevicePtr(pointer)));
  NotifyResourceDestroyed(ToCuda(pointer));
  return llvm::Error::success();
}

llvm::Expected<HostMemory<void>> CuMemHostAlloc(CurrentContext current,
                                                size_t size_bytes,
                                                CUmemhostalloc_flags flags) {
  CheckCudaContext(current);
  void* ptr;
  RETURN_IF_ERROR(cuMemHostAlloc(&ptr, size_bytes, flags));
  NotifyResourceCreated(ResourceType::kHostMemory, ptr);
  return HostMemory<void>({ptr, Platform::CUDA});
}

llvm::Error CuMemHostFree(Pointer<void> pointer) {
  RETURN_IF_ERROR(cuMemFreeHost(pointer.raw(Platform::CUDA)));
  NotifyResourceDestroyed(ToCuda(pointer));
  return llvm::Error::success();
}

llvm::Expected<RegisteredMemory<void>> CuMemHostRegister(
    CurrentContext current, void* ptr, size_t size_bytes,
    CUmemhostregister_flags flags) {
  CheckCudaContext(current);
  RETURN_IF_ERROR(cuMemHostRegister(ptr, size_bytes, flags));
  NotifyResourceCreated(ResourceType::kRegisteredMemory, ptr);
  return RegisteredMemory<void>({ptr, Platform::CUDA});
}

llvm::Error CuMemHostUnregister(Pointer<void> pointer) {
  RETURN_IF_ERROR(cuMemHostUnregister(pointer.raw(Platform::CUDA)));
  NotifyResourceDestroyed(ToCuda(pointer));
  return llvm::Error::success();
}

llvm::Expected<DeviceMemory<void>> CuMemAllocManaged(CurrentContext current,
                                                     size_t size_bytes,
                                                     CUmemAttach_flags flags) {
  CheckCudaContext(current);
  void* ptr;
  RETURN_IF_ERROR(cuMemAllocManaged(reinterpret_cast<CUdeviceptr*>(&ptr),
                                    size_bytes, flags));
  NotifyResourceCreated(ResourceType::kDeviceMemory, ptr);
  return DeviceMemory<void>({ptr, Platform::CUDA});
}

llvm::Expected<Pointer<void>> CuMemHostGetDevicePointer(
    Pointer<void> host_ptr) {
  CUdeviceptr dev_ptr;
  RETURN_IF_ERROR(cuMemHostGetDevicePointer(
      &dev_ptr, host_ptr.raw(Platform::CUDA), /*flags=*/0));
  return Pointer<void>(reinterpret_cast<void*>(dev_ptr), Platform::CUDA);
}

llvm::Expected<MemoryRange<void>> CuMemGetAddressRange(CurrentContext current,
                                                       Pointer<void> ptr) {
  CheckCudaContext(current);
  CUdeviceptr base;
  size_t size_bytes;
  RETURN_IF_ERROR(cuMemGetAddressRange(&base, &size_bytes, ToDevicePtr(ptr)));
  return MemoryRange<void>{{reinterpret_cast<void*>(base), Platform::CUDA},
                           size_bytes};
}

llvm::Expected<MemoryInfo> CuMemGetInfo(CurrentContext current) {
  CheckCudaContext(current);
  MemoryInfo info;
  RETURN_IF_ERROR(cuMemGetInfo(&info.free_bytes, &info.total_bytes));
  return info;
}

llvm::Expected<CUmemhostalloc_flags> CuMemHostGetFlags(CurrentContext current,
                                                       Pointer<void> ptr) {
  CheckCudaContext(current);
  unsigned flags;
  RETURN_IF_ERROR(cuMemHostGetFlags(&flags, ptr.raw(Platform::CUDA)));
  return static_cast<CUmemhostalloc_flags>(flags);
}

llvm::Error CuMemRangeGetAttribute(void* data, size_t data_size,
                                   CUmem_range_attribute attribute,
                                   Pointer<const void> ptr, size_t size_bytes) {
  return TO_ERROR(cuMemRangeGetAttribute(data, data_size, attribute,
                                         ToDevicePtr(ptr), size_bytes));
}
llvm::Error CuMemRangeGetAttributes(
    llvm::ArrayRef<void*> data, llvm::ArrayRef<size_t> data_sizes,
    llvm::ArrayRef<CUmem_range_attribute> attributes, Pointer<const void> ptr,
    size_t size_bytes) {
  if (data.size() != data_sizes.size() || data.size() != attributes.size()) {
    return MakeStringError("Mismatching array sizes");
  }
  return TO_ERROR(cuMemRangeGetAttributes(
      const_cast<void**>(data.data()), const_cast<size_t*>(data_sizes.data()),
      const_cast<CUmem_range_attribute*>(attributes.data()), data.size(),
      ToDevicePtr(ptr), size_bytes));
}

llvm::Error CuPointerGetAttribute(void* data, CUpointer_attribute attribute,
                                  Pointer<const void> ptr) {
  return TO_ERROR(cuPointerGetAttribute(data, attribute, ToDevicePtr(ptr)));
}

llvm::Error CuPointerGetAttributes(
    llvm::ArrayRef<void*> data, llvm::ArrayRef<CUpointer_attribute> attributes,
    Pointer<const void> ptr) {
  if (data.size() != attributes.size()) {
    return MakeStringError("Mismatching array sizes");
  }
  return TO_ERROR(cuPointerGetAttributes(
      data.size(), const_cast<CUpointer_attribute*>(attributes.data()),
      const_cast<void**>(data.data()), ToDevicePtr(ptr)));
}

llvm::Error CuMemcpy(CurrentContext current, Pointer<void> dst,
                     Pointer<const void> src, size_t count_bytes) {
  CheckCudaContext(current);
  return TO_ERROR(cuMemcpy(ToDevicePtr(dst), ToDevicePtr(src), count_bytes));
}

llvm::Error CuMemcpyAsync(CurrentContext current, Pointer<void> dst,
                          Pointer<const void> src, size_t count_bytes,
                          CUstream stream) {
  CheckCudaContext(current);
  return TO_ERROR(
      cuMemcpyAsync(ToDevicePtr(dst), ToDevicePtr(src), count_bytes, stream));
}

llvm::Error CuMemcpyPeer(Pointer<void> dst_ptr, CUcontext dst_ctx,
                         Pointer<const void> src_ptr, CUcontext src_ctx,
                         size_t count_bytes) {
  return TO_ERROR(cuMemcpyPeer(ToDevicePtr(dst_ptr), dst_ctx,
                               ToDevicePtr(src_ptr), src_ctx, count_bytes));
}

llvm::Error CuMemcpyPeerAsync(Pointer<void> dst_ptr, CUcontext dst_ctx,
                              Pointer<const void> src_ptr, CUcontext src_ctx,
                              size_t count_bytes, CUstream stream) {
  return TO_ERROR(cuMemcpyPeerAsync(ToDevicePtr(dst_ptr), dst_ctx,
                                    ToDevicePtr(src_ptr), src_ctx, count_bytes,
                                    stream));
}

llvm::Error CuMemsetD8(CurrentContext current, Pointer<void> dst,
                       std::uint8_t value, size_t count) {
  CheckCudaContext(current);
  return TO_ERROR(cuMemsetD8(ToDevicePtr(dst), value, count));
}

llvm::Error CuMemsetD16(CurrentContext current, Pointer<void> dst,
                        std::uint16_t value, size_t count) {
  CheckCudaContext(current);
  return TO_ERROR(cuMemsetD16(ToDevicePtr(dst), value, count));
}

llvm::Error CuMemsetD32(CurrentContext current, Pointer<void> dst,
                        std::uint32_t value, size_t count) {
  CheckCudaContext(current);
  return TO_ERROR(cuMemsetD32(ToDevicePtr(dst), value, count));
}

llvm::Error CuMemsetD8Async(CurrentContext current, Pointer<void> dst,
                            std::uint8_t value, size_t count, CUstream stream) {
  CheckCudaContext(current);
  return TO_ERROR(cuMemsetD8Async(ToDevicePtr(dst), value, count, stream));
}

llvm::Error CuMemsetD16Async(CurrentContext current, Pointer<void> dst,
                             std::uint16_t value, size_t count,
                             CUstream stream) {
  CheckCudaContext(current);
  return TO_ERROR(cuMemsetD16Async(ToDevicePtr(dst), value, count, stream));
}

llvm::Error CuMemsetD32Async(CurrentContext current, Pointer<void> dst,
                             std::uint32_t value, size_t count,
                             CUstream stream) {
  CheckCudaContext(current);
  return TO_ERROR(cuMemsetD32Async(ToDevicePtr(dst), value, count, stream));
}

llvm::Expected<OwningModule> CuModuleLoadDataEx(
    CurrentContext current, const void* image,
    llvm::ArrayRef<CUjit_option> jit_options,
    llvm::ArrayRef<void*> jit_option_values) {
  CheckCudaContext(current);
  if (jit_options.size() != jit_option_values.size()) {
    return MakeStringError("Mismatching array sizes");
  }
  CUmodule module;
  RETURN_IF_ERROR(
      cuModuleLoadDataEx(&module, image, jit_options.size(),
                         const_cast<CUjit_option*>(jit_options.data()),
                         const_cast<void**>(jit_option_values.data())));
  NotifyResourceCreated(ResourceType::kModule, module);
  return OwningModule(module);
}

llvm::Expected<OwningModule> CuModuleLoadData(CurrentContext current,
                                              const void* image) {
  CheckCudaContext(current);
  CUmodule module;
  RETURN_IF_ERROR(cuModuleLoadData(&module, image));
  NotifyResourceCreated(ResourceType::kModule, module);
  return OwningModule(module);
}

llvm::Error CuModuleUnload(CUmodule module) {
  if (module == nullptr) return llvm::Error::success();
  RETURN_IF_ERROR(cuModuleUnload(module));
  NotifyResourceDestroyed(module);
  return llvm::Error::success();
}

llvm::Expected<Function> CuModuleGetFunction(CUmodule module,
                                             const char* name) {
  CUfunction function;
  RETURN_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}

llvm::Expected<int> CuFuncGetAttribute(CurrentContext current,
                                       CUfunction_attribute attribute,
                                       CUfunction function) {
  CheckCudaContext(current);
  int value;
  RETURN_IF_ERROR(cuFuncGetAttribute(&value, attribute, function));
  return value;
}

llvm::Error CuFuncSetAttribute(CurrentContext current, CUfunction function,
                               CUfunction_attribute attribute, int value) {
  CheckCudaContext(current);
  return TO_ERROR(cuFuncSetAttribute(function, attribute, value));
}

llvm::Error CuFuncSetCacheConfig(CurrentContext current, CUfunction function,
                                 CUfunc_cache config) {
  CheckCudaContext(current);
  return TO_ERROR(cuFuncSetCacheConfig(function, config));
}

llvm::Error CuFuncSetSharedMemConfig(CurrentContext current,
                                     CUfunction function,
                                     CUsharedconfig config) {
  CheckCudaContext(current);
  return TO_ERROR(cuFuncSetSharedMemConfig(function, config));
}

llvm::Error CuLaunchKernel(CurrentContext current, CUfunction function,
                           unsigned grid_dim_x, unsigned grid_dim_y,
                           unsigned grid_dim_z, unsigned block_dim_x,
                           unsigned block_dim_y, unsigned block_dim_z,
                           unsigned shared_memory_size_bytes, CUstream stream,
                           llvm::ArrayRef<const void*> arguments,
                           llvm::ArrayRef<const void*> extras) {
  CheckCudaContext(current);
  return TO_ERROR(cuLaunchKernel(
      function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
      block_dim_z, shared_memory_size_bytes, stream,
      const_cast<void**>(arguments.data()), const_cast<void**>(extras.data())));
}

llvm::Error CuLaunchCooperativeKernel(
    CurrentContext current, CUfunction function, unsigned grid_dim_x,
    unsigned grid_dim_y, unsigned grid_dim_z, unsigned block_dim_x,
    unsigned block_dim_y, unsigned block_dim_z,
    unsigned shared_memory_size_bytes, CUstream stream,
    llvm::ArrayRef<const void*> arguments) {
  CheckCudaContext(current);
  return TO_ERROR(cuLaunchCooperativeKernel(
      function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
      block_dim_z, shared_memory_size_bytes, stream,
      const_cast<void**>(arguments.data())));
}

llvm::Expected<int> CuOccupancyMaxActiveBlocksPerMultiprocessor(
    CurrentContext current, CUfunction function, int block_size,
    size_t dynamic_shared_memory_size) {
  return CuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      current, function, block_size, dynamic_shared_memory_size,
      CU_OCCUPANCY_DEFAULT);
}

llvm::Expected<int> CuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    CurrentContext current, CUfunction function, int block_size,
    size_t dynamic_shared_memory_size, CUoccupancy_flags flags) {
  CheckCudaContext(current);
  int num_blocks;
  RETURN_IF_ERROR(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &num_blocks, function, block_size, dynamic_shared_memory_size, flags));
  return num_blocks;
}

llvm::Expected<MaxPotentialBlockSize> CuOccupancyMaxPotentialBlockSize(
    CurrentContext current, CUfunction function,
    const std::function<size_t(int)>& block_size_to_dynamic_shared_memory_size,
    int block_size_limit) {
  return CuOccupancyMaxPotentialBlockSizeWithFlags(
      current, function, block_size_to_dynamic_shared_memory_size,
      block_size_limit, CU_OCCUPANCY_DEFAULT);
}

llvm::Expected<MaxPotentialBlockSize> CuOccupancyMaxPotentialBlockSizeWithFlags(
    CurrentContext current, CUfunction function,
    const std::function<size_t(int)>& block_size_to_dynamic_shared_memory_size,
    int block_size_limit, CUoccupancy_flags flags) {
  CheckCudaContext(current);
  thread_local static const std::function<size_t(int)>* occupancy_callback_ptr;
  occupancy_callback_ptr = &block_size_to_dynamic_shared_memory_size;
  CUoccupancyB2DSize callback = [](int block_size) -> size_t {
    return (*occupancy_callback_ptr)(block_size);
  };
  MaxPotentialBlockSize result;
  size_t ignored_dynamic_shared_memory_size = 0;
  RETURN_IF_ERROR(cuOccupancyMaxPotentialBlockSizeWithFlags(
      &result.min_num_blocks, &result.block_size, function, callback,
      ignored_dynamic_shared_memory_size, block_size_limit, flags));
  return result;
}

// Definitions from wrapper_detail.h.

llvm::Expected<CUdevice> CuCtxGetDevice(CUcontext context) {
  CUdevice device;
  if (kContextTls.cuda_ctx == context) {
    RETURN_IF_ERROR(cuCtxGetDevice(&device));
  } else {
    RETURN_IF_ERROR(cuCtxPushCurrent(context));
    auto result = cuCtxGetDevice(&device);
    RETURN_IF_ERROR(cuCtxPopCurrent(nullptr));
    RETURN_IF_ERROR(result);
  }
  return device;
}

void CheckCudaContext(CurrentContext) {
#ifndef NDEBUG
  DieIfError([&]() -> llvm::Error {
    if (auto error = CheckPlatform(kContextTls.platform, Platform::CUDA))
      return error;
    CUcontext cuda_ctx;
    RETURN_IF_ERROR(cuCtxGetCurrent(&cuda_ctx));
    if (kContextTls.cuda_ctx != cuda_ctx) {
      std::string msg = llvm::formatv("Expected context to be {0}, but got {1}",
                                      kContextTls.cuda_ctx, cuda_ctx);
      return llvm::createStringError(llvm::inconvertibleErrorCode(), msg);
    }
    return llvm::Error::success();
  }());
#endif
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
