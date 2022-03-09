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

// Thin wrapper around the HIP API adding llvm::Error and explicit context.
#include "tfrt/gpu/wrapper/hip_wrapper.h"

#include <cstddef>

#include "llvm/Support/FormatVariadic.h"
#include "tfrt/gpu/wrapper/hip_stub.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& Print(llvm::raw_ostream& os, hipError_t error) {
  const char* msg = hipGetErrorName(error);
  if (msg != nullptr) {
    os << msg;
  } else {
    os << llvm::formatv("hipError_t({0})", static_cast<int>(error));
  }
  msg = hipGetErrorString(error);
  if (msg != nullptr) os << " (" << msg << ")";
  return os;
}

llvm::raw_ostream& Print(llvm::raw_ostream& os, hiprtcResult result) {
  const char* msg = hiprtcGetErrorString(result);
  if (msg != nullptr) os << "hiprtc Error: (" << msg << ")";
  return os;
}

// Convert wrapper types to HIP types.
static hipDevice_t ToRocm(Device device) { return device.id(Platform::ROCm); }

llvm::Error HipInit() { return TO_ERROR(hipInit(/*flags=*/0)); }

llvm::Error HipFree(std::nullptr_t) { return TO_ERROR(hipFree(nullptr)); }

llvm::Expected<int> HipDriverGetVersion() {
  int version = -1;
  RETURN_IF_ERROR(hipDriverGetVersion(&version));
  return version;
}

llvm::Expected<int> HipRuntimeGetVersion() {
  int version = -1;
  RETURN_IF_ERROR(hipRuntimeGetVersion(&version));
  return version;
}

llvm::Error HipGetLastError(CurrentContext current) {
  CheckHipContext(current);
  return TO_ERROR(hipGetLastError());
}

llvm::Error HipPeekAtLastError(CurrentContext current) {
  CheckHipContext(current);
  return TO_ERROR(hipPeekAtLastError());
}

llvm::Expected<hipDeviceProp_t> HipGetDeviceProperties(CurrentContext current) {
  CheckHipContext(current);
  int deviceId;
  RETURN_IF_ERROR(hipGetDevice(&deviceId));
  hipDeviceProp_t properties;
  RETURN_IF_ERROR(hipGetDeviceProperties(&properties, deviceId));
  return properties;
}

llvm::Expected<int> HipDeviceGetCount() {
  int count;
  RETURN_IF_ERROR(hipGetDeviceCount(&count));
  return count;
}

llvm::Expected<Device> HipDeviceGet(int ordinal) {
  hipDevice_t device;
  RETURN_IF_ERROR(hipDeviceGet(&device, ordinal));
  return Device(device, Platform::ROCm);
}

llvm::Expected<std::string> HipDeviceGetName(Device device) {
  char name[100];
  RETURN_IF_ERROR(hipDeviceGetName(name, sizeof(name), ToRocm(device)));
  return std::string(name);
}

llvm::Expected<size_t> HipDeviceTotalMem(Device device) {
  size_t size_bytes;
  RETURN_IF_ERROR(hipDeviceTotalMem(&size_bytes, ToRocm(device)));
  return size_bytes;
}

llvm::Expected<int> HipDeviceGetAttribute(hipDeviceAttribute_t attribute,
                                          Device device) {
  int value;
  RETURN_IF_ERROR(hipDeviceGetAttribute(&value, attribute, ToRocm(device)));
  return value;
}

llvm::Expected<std::string> HipDeviceGetPCIBusId(Device device) {
  char bus_id[100];
  RETURN_IF_ERROR(hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), ToRocm(device)));
  return std::string(bus_id);
}

llvm::Expected<size_t> HipDeviceGetLimit(CurrentContext current,
                                         hipLimit_t limit) {
  CheckHipContext(current);
  size_t value;
  RETURN_IF_ERROR(hipDeviceGetLimit(&value, limit));
  return value;
}

llvm::Expected<StreamPriorityRange> HipDeviceGetStreamPriorityRange(
    CurrentContext current) {
  CheckHipContext(current);
  StreamPriorityRange priority_range;
  RETURN_IF_ERROR(hipDeviceGetStreamPriorityRange(&priority_range.least,
                                                  &priority_range.greatest));
  return priority_range;
}

llvm::Error HipDeviceSynchronize(CurrentContext current) {
  CheckHipContext(current);
  return TO_ERROR(hipDeviceSynchronize());
}

llvm::Expected<int> HipDeviceCanAccessPeer(Device src_dev, Device dst_dev) {
  int result;
  RETURN_IF_ERROR(
      hipDeviceCanAccessPeer(&result, ToRocm(src_dev), ToRocm(dst_dev)));
  return result;
}

llvm::Error HipDeviceEnablePeerAccess(CurrentContext current,
                                      Device peer_device) {
  CheckHipContext(current);
  return TO_ERROR(hipDeviceEnablePeerAccess(ToRocm(peer_device), /*flags=*/0));
}

llvm::Error HipDeviceDisablePeerAccess(CurrentContext current,
                                       Device peer_device) {
  CheckHipContext(current);
  return TO_ERROR(hipDeviceDisablePeerAccess(peer_device.id(Platform::ROCm)));
}

llvm::Expected<OwningContext> HipDevicePrimaryCtxRetain(Device device) {
  hipCtx_t hip_ctx;
  RETURN_IF_ERROR(hipDevicePrimaryCtxRetain(&hip_ctx, ToRocm(device)));
  kContextTls.hip_may_skip_set_ctx = true;
  return OwningContext(hip_ctx);
}

llvm::Error HipDevicePrimaryCtxRelease(Device device) {
  if (auto has_instance = CheckNoCurrentContext()) return has_instance;
  RETURN_IF_ERROR(hipDevicePrimaryCtxRelease(ToRocm(device)));
  // Releasing the primary context does not change the current context, but
  // decrements the internal reference count and deactivates it iff zero.
  kContextTls.hip_may_skip_set_ctx = false;
#ifndef NDEBUG
  auto state = HipDevicePrimaryCtxGetState(device);
  if (!state) return state.takeError();
  if (!state->active) {
    auto context = HipDevicePrimaryCtxRetain(device);
    if (!context) return context.takeError();
    RETURN_IF_ERROR(hipDevicePrimaryCtxRelease(ToRocm(device)));
    return CheckNoDanglingResources(context->release());
  }
#endif
  return llvm::Error::success();
}

llvm::Error HipDevicePrimaryCtxReset(Device device) {
  if (kContextTls.platform == Platform::ROCm) {
    if (auto has_instance = CheckNoCurrentContext()) {
      // There is a CurrentContext instance, check that primary is not current.
      hipCtx_t context;
      RETURN_IF_ERROR(hipCtxGetCurrent(&context));
      if (kContextTls.hip_ctx == context) return has_instance;
    }
  }
  Context context;
#ifndef NDEBUG
  auto context_or = HipDevicePrimaryCtxRetain(device);
  if (!context_or) return context_or.takeError();
  context = context_or->release();
  RETURN_IF_ERROR(hipDevicePrimaryCtxRelease(ToRocm(device)));
#endif
  RETURN_IF_ERROR(hipDevicePrimaryCtxReset(ToRocm(device)));
  return CheckNoDanglingResources(context);
}

llvm::Expected<ContextState> HipDevicePrimaryCtxGetState(Device device) {
  unsigned flags;
  int active;
  RETURN_IF_ERROR(hipDevicePrimaryCtxGetState(ToRocm(device), &flags, &active));
  return ContextState{static_cast<hipDeviceFlags_t>(flags), active};
}

llvm::Error HipDevicePrimaryCtxSetFlags(Device device, hipDeviceFlags_t flags) {
  return TO_ERROR(hipDevicePrimaryCtxSetFlags(ToRocm(device), flags));
}

llvm::Expected<OwningContext> HipCtxCreate(hipDeviceFlags_t flags,
                                           Device device) {
  // Check no instance of CurrentContext exists in this thread.
  if (auto has_instance = CheckNoCurrentContext())
    return std::move(has_instance);
  RETURN_IF_ERROR(hipCtxCreate(&kContextTls.hip_ctx, flags, ToRocm(device)));
  kContextTls.platform = Platform::ROCm;
  kContextTls.hip_may_skip_set_ctx = true;
  return OwningContext(kContextTls.hip_ctx);
}

llvm::Error HipCtxDestroy(hipCtx_t context) {
  if (context == nullptr) return llvm::Error::success();
  if (kContextTls.hip_ctx == context &&
      kContextTls.platform == Platform::ROCm) {
    // Check that there is no CurrentContext instance.
    if (auto has_instance = CheckNoCurrentContext()) return has_instance;
  }
  RETURN_IF_ERROR(hipCtxDestroy(context));
  if (kContextTls.hip_ctx == context) {
    // Destroying the current context makes the primary context current if there
    // is one for the same device. The primary context may be inactive.
    RETURN_IF_ERROR(hipCtxGetCurrent(&kContextTls.hip_ctx));
    kContextTls.hip_may_skip_set_ctx = false;
  }
  return CheckNoDanglingResources(context);
}

llvm::Expected<CurrentContext> HipCtxGetCurrent() {
  return CreateCurrentContext();
}

llvm::Expected<CurrentContext> HipCtxSetCurrent(hipCtx_t context) {
  // Check no instance of CurrentContext exists in this thread.
  if (auto has_instance = CheckNoCurrentContext())
    return std::move(has_instance);
  // Skip setting context if it's already current. This is an optimization
  // that requires users to not change the current context through the HIP
  // API. This is validated through calls to CheckHipContext().
  if (kContextTls.hip_ctx != context || !kContextTls.hip_may_skip_set_ctx) {
    RETURN_IF_ERROR(hipCtxSetCurrent(context));
    if (context != nullptr) {
      kContextTls.hip_may_skip_set_ctx = true;
    } else {
      // Setting the null context makes the primary context current if there
      // is one for the current device. The primary context may be inactive.
      RETURN_IF_ERROR(hipCtxGetCurrent(&context));
      kContextTls.hip_may_skip_set_ctx = (context == nullptr);
    }
    kContextTls.hip_ctx = context;
  }
  kContextTls.platform = Platform::ROCm;
  auto current = CreateCurrentContext();
  // Catch false skipping of setting context above.
  CheckHipContext(current);
  return current;
}

llvm::Expected<unsigned> HipCtxGetApiVersion(hipCtx_t context) {
  int version;
  RETURN_IF_ERROR(hipCtxGetApiVersion(context, &version));
  return version;
}

llvm::Expected<Device> HipCtxGetDevice(CurrentContext current) {
  CheckHipContext(current);
  hipDevice_t device;
  RETURN_IF_ERROR(hipCtxGetDevice(&device));
  return Device(device, Platform::ROCm);
}

llvm::Expected<hipDeviceFlags_t> HipCtxGetFlags(CurrentContext current) {
  CheckHipContext(current);
  unsigned flags;
  RETURN_IF_ERROR(hipCtxGetFlags(&flags));
  return static_cast<hipDeviceFlags_t>(flags);
}

llvm::Expected<OwningStream> HipStreamCreate(CurrentContext current,
                                             hipStreamFlags_t flags) {
  CheckHipContext(current);
  hipStream_t stream;
  RETURN_IF_ERROR(hipStreamCreateWithFlags(&stream, flags));
  NotifyResourceCreated(ResourceType::kStream, stream);
  return OwningStream(stream);
}

llvm::Expected<OwningStream> HipStreamCreate(CurrentContext current,
                                             hipStreamFlags_t flags,
                                             int priority) {
  CheckHipContext(current);
  hipStream_t stream;
  RETURN_IF_ERROR(hipStreamCreateWithPriority(&stream, flags, priority));
  NotifyResourceCreated(ResourceType::kStream, stream);
  return OwningStream(stream);
}

llvm::Error HipStreamDestroy(hipStream_t stream) {
  if (stream == nullptr) return llvm::Error::success();
  RETURN_IF_ERROR(hipStreamDestroy(stream));
  NotifyResourceDestroyed(stream);
  return llvm::Error::success();
}

llvm::Expected<int> HipStreamGetPriority(hipStream_t stream) {
  int priority;
  RETURN_IF_ERROR(hipStreamGetPriority(stream, &priority));
  return priority;
}

llvm::Expected<hipStreamFlags_t> HipStreamGetFlags(hipStream_t stream) {
  unsigned flags;
  RETURN_IF_ERROR(hipStreamGetFlags(stream, &flags));
  return static_cast<hipStreamFlags_t>(flags);
}

llvm::Error HipStreamSynchronize(hipStream_t stream) {
  return TO_ERROR(hipStreamSynchronize(stream));
}

llvm::Expected<bool> HipStreamQuery(hipStream_t stream) {
  auto result = hipStreamQuery(stream);
  if (result == hipErrorNotReady) {
    return false;
  }
  RETURN_IF_ERROR(result);
  return true;
}

llvm::Error HipStreamWaitEvent(hipStream_t stream, hipEvent_t event) {
  return TO_ERROR(hipStreamWaitEvent(stream, event, /*flags=*/0));
}

llvm::Expected<OwningEvent> HipEventCreate(CurrentContext current,
                                           hipEventFlags_t flags) {
  CheckHipContext(current);
  hipEvent_t event;
  RETURN_IF_ERROR(hipEventCreateWithFlags(&event, flags));
  NotifyResourceCreated(ResourceType::kEvent, event);
  return OwningEvent(event);
}

llvm::Error HipEventDestroy(hipEvent_t event) {
  if (event == nullptr) return llvm::Error::success();
  RETURN_IF_ERROR(hipEventDestroy(event));
  NotifyResourceDestroyed(event);
  return llvm::Error::success();
}

llvm::Error HipEventRecord(hipEvent_t event, hipStream_t stream) {
  return TO_ERROR(hipEventRecord(event, stream));
}

llvm::Error HipEventSynchronize(hipEvent_t event) {
  return TO_ERROR(hipEventSynchronize(event));
}

llvm::Expected<bool> HipEventQuery(hipEvent_t event) {
  auto result = hipEventQuery(event);
  if (result == hipErrorNotReady) {
    return false;
  }
  RETURN_IF_ERROR(result);
  return true;
}

llvm::Expected<float> HipEventElapsedTime(hipEvent_t start, hipEvent_t end) {
  float time_ms;
  RETURN_IF_ERROR(hipEventElapsedTime(&time_ms, start, end));
  return time_ms;
}

llvm::Expected<DeviceMemory<void>> HipMemAlloc(CurrentContext current,
                                               size_t size_bytes) {
  CheckHipContext(current);
  hipDeviceptr_t ptr;
  if (auto error = TO_ERROR(hipMalloc(&ptr, size_bytes))) {
    return llvm::handleErrors(std::move(error),
                              [&](std::unique_ptr<ErrorInfo<hipError_t>> info) {
                                return GetResult(*info) == hipErrorOutOfMemory
                                           ? MakeOomError(current, size_bytes)
                                           : llvm::Error(std::move(info));
                              });
  }
  NotifyResourceCreated(ResourceType::kDeviceMemory, ptr);
  return DeviceMemory<void>({ptr, Platform::ROCm});
}

llvm::Error HipMemFree(Pointer<void> pointer) {
  RETURN_IF_ERROR(hipFree(ToRocm(pointer)));
  NotifyResourceDestroyed(ToRocm(pointer));
  return llvm::Error::success();
}

llvm::Expected<HostMemory<void>> HipMemHostAlloc(CurrentContext current,
                                                 size_t size_bytes,
                                                 hipHostMallocFlags_t flags) {
  CheckHipContext(current);
  void* ptr;
  RETURN_IF_ERROR(hipHostMalloc(&ptr, size_bytes, flags));
  NotifyResourceCreated(ResourceType::kHostMemory, ptr);
  return HostMemory<void>({ptr, Platform::ROCm});
}

llvm::Error HipMemHostFree(Pointer<void> pointer) {
  RETURN_IF_ERROR(hipHostFree(ToRocm(pointer)));
  NotifyResourceDestroyed(ToRocm(pointer));
  return llvm::Error::success();
}

llvm::Expected<RegisteredMemory<void>> HipMemHostRegister(
    CurrentContext current, void* ptr, size_t size_bytes,
    hipHostRegisterFlags_t flags) {
  CheckHipContext(current);
  RETURN_IF_ERROR(hipHostRegister(ptr, size_bytes, flags));
  NotifyResourceCreated(ResourceType::kRegisteredMemory, ptr);
  return RegisteredMemory<void>({ptr, Platform::ROCm});
}

llvm::Error HipMemHostUnregister(Pointer<void> pointer) {
  RETURN_IF_ERROR(hipHostUnregister(ToRocm(pointer)));
  NotifyResourceDestroyed(ToRocm(pointer));
  return llvm::Error::success();
}

llvm::Expected<DeviceMemory<void>> HipMemAllocManaged(
    CurrentContext current, size_t size_bytes, hipMemAttachFlags_t flags) {
  CheckHipContext(current);
  hipDeviceptr_t ptr;
  RETURN_IF_ERROR(hipMallocManaged(&ptr, size_bytes, flags));
  NotifyResourceCreated(ResourceType::kDeviceMemory, ptr);
  return DeviceMemory<void>({ptr, Platform::ROCm});
}

llvm::Expected<Pointer<void>> HipMemHostGetDevicePointer(
    Pointer<void> host_ptr) {
  hipDeviceptr_t device_ptr;
  RETURN_IF_ERROR(
      hipHostGetDevicePointer(&device_ptr, ToRocm(host_ptr), /*flags=*/0));
  return Pointer<void>(device_ptr, Platform::ROCm);
}

llvm::Expected<MemoryRange<void>> HipMemGetAddressRange(CurrentContext current,
                                                        Pointer<void> ptr) {
  CheckHipContext(current);
  hipDeviceptr_t base;
  size_t size_bytes;
  RETURN_IF_ERROR(hipMemGetAddressRange(&base, &size_bytes, ToRocm(ptr)));
  return MemoryRange<void>{{base, Platform::ROCm}, size_bytes};
}

llvm::Expected<MemoryInfo> HipMemGetInfo(CurrentContext current) {
  CheckHipContext(current);
  MemoryInfo info;
  RETURN_IF_ERROR(hipMemGetInfo(&info.free_bytes, &info.total_bytes));
  return info;
}

llvm::Expected<hipPointerAttribute_t> HipPointerGetAttributes(
    Pointer<const void> ptr) {
  hipPointerAttribute_t attributes;
  RETURN_IF_ERROR(hipPointerGetAttributes(&attributes, ToRocm(ptr)));
  return attributes;
}

llvm::Error HipMemcpy(CurrentContext current, Pointer<void> dst,
                      Pointer<const void> src, size_t count_bytes) {
  CheckHipContext(current);
  return TO_ERROR(
      hipMemcpy(ToRocm(dst), ToRocm(src), count_bytes, hipMemcpyDefault));
}

llvm::Error HipMemcpyAsync(CurrentContext current, Pointer<void> dst,
                           Pointer<const void> src, size_t count_bytes,
                           hipStream_t stream) {
  CheckHipContext(current);
  return TO_ERROR(hipMemcpyAsync(ToRocm(dst), ToRocm(src), count_bytes,
                                 hipMemcpyDefault, stream));
}

llvm::Error HipMemcpyPeer(Pointer<void> dst_ptr, Device dst_dev,
                          Pointer<const void> src_ptr, Device src_dev,
                          size_t count_bytes) {
  return TO_ERROR(hipMemcpyPeer(ToRocm(dst_ptr), ToRocm(dst_dev),
                                ToRocm(src_ptr), ToRocm(src_dev), count_bytes));
}

llvm::Error HipMemcpyPeerAsync(Pointer<void> dst_ptr, Device dst_dev,
                               Pointer<const void> src_ptr, Device src_dev,
                               size_t count_bytes, hipStream_t stream) {
  return TO_ERROR(hipMemcpyPeerAsync(ToRocm(dst_ptr), ToRocm(dst_dev),
                                     ToRocm(src_ptr), ToRocm(src_dev),
                                     count_bytes, stream));
}

llvm::Error HipMemsetD8(CurrentContext current, Pointer<void> dst,
                        std::uint8_t value, size_t count) {
  CheckHipContext(current);
  return TO_ERROR(hipMemset(ToRocm(dst), value, count));
}

llvm::Error HipMemsetD32(CurrentContext current, Pointer<void> dst,
                         std::uint32_t value, size_t count) {
  CheckHipContext(current);
  return TO_ERROR(hipMemsetD32(ToRocm(dst), value, count));
}

llvm::Error HipMemsetD8Async(CurrentContext current, Pointer<void> dst,
                             std::uint8_t value, size_t count,
                             hipStream_t stream) {
  CheckHipContext(current);
  return TO_ERROR(hipMemsetAsync(ToRocm(dst), value, count, stream));
}

llvm::Error HipMemsetD32Async(CurrentContext current, Pointer<void> dst,
                              std::uint32_t value, size_t count,
                              hipStream_t stream) {
  CheckHipContext(current);
  return TO_ERROR(hipMemsetD32Async(ToRocm(dst), value, count, stream));
}

llvm::Expected<OwningModule> HipModuleLoadData(CurrentContext current,
                                               const void* image) {
  CheckHipContext(current);
  hipModule_t module;
  RETURN_IF_ERROR(hipModuleLoadData(&module, image));
  NotifyResourceCreated(ResourceType::kModule, module);
  return OwningModule(module);
}

llvm::Expected<OwningModule> HipRTCModuleLoadData(CurrentContext current,
                                                  const void* image) {
  CheckHipContext(current);
  hiprtcProgram prog;
  //auto img = reinterpret_cast<const char*>(const_cast<void*>(image));
  auto kernel = static_cast<const char*>(image);
  std::string kname(kernel);
  kname += ".cu";
  RETURN_IF_ERROR(hiprtcCreateProgram(&prog,
                                      kernel,
                                      kname.c_str(),
                                      0,
                                      nullptr,
                                      nullptr
                                     ));
  hiprtcResult compileResult = hiprtcCompileProgram(prog, 0, nullptr);
  if (compileResult != HIPRTC_SUCCESS) {
    size_t logSize;
    hiprtcGetProgramLogSize(prog, &logSize);
    if (logSize) {
      std::string log(logSize, '\0');
      hiprtcGetProgramLog(prog, &log[0]);
      MakeStringError(log.c_str());
    }
  }

  size_t code_size;
  RETURN_IF_ERROR(hiprtcGetCodeSize(prog, &code_size));
  std::vector<char> code(code_size);
  RETURN_IF_ERROR(hiprtcGetCode(prog, code.data()));
  RETURN_IF_ERROR(hiprtcDestroyProgram(&prog));

  hipModule_t module;
  RETURN_IF_ERROR(hipModuleLoadData(&module, code.data()));

  NotifyResourceCreated(ResourceType::kModule, module);
  return OwningModule(module);
}

llvm::Expected<OwningModule> HipModuleLoadDataEx(
    CurrentContext current, const void* image,
    llvm::ArrayRef<hipJitOption> options, llvm::ArrayRef<void*> option_values) {
  CheckHipContext(current);
  hipModule_t module;
  RETURN_IF_ERROR(hipModuleLoadDataEx(
      &module, image, options.size(), const_cast<hipJitOption*>(options.data()),
      const_cast<void**>(option_values.data())));
  NotifyResourceCreated(ResourceType::kModule, module);
  return OwningModule(module);
}

llvm::Error HipModuleUnload(hipModule_t module) {
  if (module == nullptr) return llvm::Error::success();
  RETURN_IF_ERROR(hipModuleUnload(module));
  NotifyResourceDestroyed(module);
  return llvm::Error::success();
}

llvm::Expected<Function> HipModuleGetFunction(hipModule_t module,
                                              const char* name) {
  hipFunction_t function;
  RETURN_IF_ERROR(hipModuleGetFunction(&function, module, name));
  return function;
}

llvm::Expected<MemoryRange<void>> HipModuleGetGlobal(hipModule_t module,
                                                     const char* name) {
  void* ptr;
  size_t size;
  RETURN_IF_ERROR(hipModuleGetGlobal(&ptr, &size, module, name));
  return MemoryRange<void>{{ptr, Platform::ROCm}, size};
}

llvm::Expected<hipFuncAttributes> HipFuncGetAttributes(CurrentContext current,
                                                       hipFunction_t function) {
  CheckHipContext(current);
  hipFuncAttributes attributes;
  RETURN_IF_ERROR(hipFuncGetAttributes(&attributes, function));
  return attributes;
}

llvm::Error HipLaunchKernel(CurrentContext current, hipFunction_t function,
                            unsigned grid_dim_x, unsigned grid_dim_y,
                            unsigned grid_dim_z, unsigned block_dim_x,
                            unsigned block_dim_y, unsigned block_dim_z,
                            unsigned shared_memory_size_bytes,
                            hipStream_t stream, llvm::ArrayRef<void*> arguments,
                            llvm::ArrayRef<void*> extras) {
  CheckHipContext(current);
  return TO_ERROR(hipModuleLaunchKernel(
      function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
      block_dim_z, shared_memory_size_bytes, stream,
      const_cast<void**>(arguments.data()), const_cast<void**>(extras.data())));
}

llvm::Error HipLaunchCooperativeKernel(
    CurrentContext current, hipFunction_t function, unsigned grid_dim_x,
    unsigned grid_dim_y, unsigned grid_dim_z, unsigned block_dim_x,
    unsigned block_dim_y, unsigned block_dim_z,
    unsigned shared_memory_size_bytes, hipStream_t stream,
    llvm::ArrayRef<void*> arguments) {
  CheckHipContext(current);
  hipDim3_t grid_dim = {grid_dim_x, grid_dim_y, grid_dim_z};
  hipDim3_t block_dim = {block_dim_x, block_dim_y, block_dim_z};
  return TO_ERROR(hipLaunchCooperativeKernel(
      function, grid_dim, block_dim, const_cast<void**>(arguments.data()),
      shared_memory_size_bytes, stream));
}

llvm::Expected<int> HipOccupancyMaxActiveBlocksPerMultiprocessor(
    CurrentContext current, hipFunction_t function, int block_size,
    size_t dynamic_shared_memory_size) {
  CheckHipContext(current);
  int32_t num_blocks;
  RETURN_IF_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, function, block_size, dynamic_shared_memory_size));
  return num_blocks;
}

llvm::Expected<MaxPotentialBlockSize> HipOccupancyMaxPotentialBlockSize(
    CurrentContext current, hipFunction_t function,
    size_t dynamic_shared_memory_size, int block_size_limit) {
  CheckHipContext(current);
  int32_t min_num_blocks, block_size;
  RETURN_IF_ERROR(hipOccupancyMaxPotentialBlockSize(
      &min_num_blocks, &block_size, function, dynamic_shared_memory_size,
      block_size_limit));
  return MaxPotentialBlockSize{min_num_blocks, block_size};
}

// Definitions from wrapper_detail.h.

llvm::Expected<hipDevice_t> HipCtxGetDevice(hipCtx_t context) {
  hipDevice_t device;
  if (kContextTls.hip_ctx == context) {
    RETURN_IF_ERROR(hipCtxGetDevice(&device));
  } else {
    RETURN_IF_ERROR(hipCtxPushCurrent(context));
    auto result = hipCtxGetDevice(&device);
    RETURN_IF_ERROR(hipCtxPopCurrent(nullptr));
    RETURN_IF_ERROR(result);
  }
  return device;
}

void CheckHipContext(CurrentContext) {
#ifndef NDEBUG
  DieIfError([&]() -> llvm::Error {
    if (auto error = CheckPlatform(kContextTls.platform, Platform::ROCm))
      return error;
    hipCtx_t hip_ctx;
    RETURN_IF_ERROR(hipCtxGetCurrent(&hip_ctx));
    if (kContextTls.hip_ctx != hip_ctx) {
      std::string msg = llvm::formatv("Expected context to be {0}, but got {1}",
                                      kContextTls.hip_ctx, hip_ctx);
      return llvm::createStringError(llvm::inconvertibleErrorCode(), msg);
    }
    return llvm::Error::success();
  }());
#endif
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
