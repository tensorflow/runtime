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

//===- stream_wrapper.h -----------------------------------------*- C++ -*-===//
//
// Thin abstraction layer for CUDA and HIP.
//
// The stream wrapper provides a safer and more convenient access to CUDA and
// ROCm platform APIs as a very thin layer. It also provides an abstraction
// of the two platforms where their functionality overlap. For this, resources
// (contexts, streams, events, modules, buffers) are wrapped in discriminated
// union types. The ROCm platform targets the HIP API which shares a lot of
// commonality with CUDA (see http://www.google.com/search?q=hip+porting+guide).
// Most user code can therefore be platform agnostic and platform specific
// sections can easily be mixed in.
//
// Here is some example code which launches a kernel operating on a temp buffer:
//
//   llvm::Error LaunchMyKernel(Context ctx, Stream stream, Function func) {
//     llvm::ExitOnError die_if_error;
//     CurrentContext current = die_if_error(CtxSetCurrent(ctx));
//     size_t dynamic_shared_memory = 42;
//     auto config = die_if_error(OccupancyMaxPotentialBlockSize(current,
//         func, dynamic_shared_memory, /*block_size_limit=*/1024));
//     if (current.platform() == Platform::CUDA) {
//       die_if_error(CuFuncSetAttribute(current, func,
//           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
//           dynamic_shared_memory));
//     }
//     DeviceMemory<void> buffer = die_if_error(MemAlloc(current, 1234567));
//     Pointer<void> pointer = buffer.get();
//     die_if_error(LaunchKernel(current, func, {{config.min_num_blocks, 1, 1}},
//         {{config.block_size, 1, 1}}, dynamic_shared_memory, stream,
//         pointer.raw(current.platform())));
//     return StreamSynchronize(stream);
//   }
//
// This will hopefully make it easier to parse the design rationale below.
//
// Return Error/Expected:
// The CUDA/HIP return codes are converted to llvm::Error. Return values are not
// passed as pointer arguments but wrapped in llvm::Expected. The die_if_error
// functor is only used for illustration here, non-sample code should return the
// error if one occurs.
//
// RAII types:
// Created resources are returned as owning handles. For example, the buffer
// instance above deallocates the memory when it goes out of scope. The RAII
// types are std::unique_ptrs (with custom deleters) and therefore support
// normal move semantics and allow taking ownership.
//
// Runtime platform:
// All handles can be queried for their target platform. For example, the above
// code is platform agnostic but configures a function attribute on CUDA, which
// are only available there. There is no abstraction for functionality that only
// exists for one platform (note the Cu prefix on CuFuncSetAttribute()).
//
// Wrapped pointers:
// Pointers are wrapped in almost fancy pointer types that support the normal
// pointer arithmetics and casting semantics but requires explicit casting to
// the underlying pointer to avoid accidental dereferencing on the host.
//
// Explicit current context:
// Setting the current context returns a handle that is passed to functions that
// use the current context state. Setting another context while a CurrentContext
// instance exists is an error that is enforced in debug builds.
//
// Safe casting:
// Platform specific APIs take platform specific types. The 'func' parameter in
// the CuFuncSetAttribute() call is implicitly casted to a CUfunction. Implicit
// casts are checked for correctness in debug builds. Pointer and device types
// are slightly different, because the CUDA/HIP API types are the same.
// Retrieving the underlying raw pointer or device ID requires specifying the
// target platform (which is again checked in debug builds). The platform
// specific APIs do this for the user.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_GPU_STREAM_STREAM_WRAPPER_H_
#define TFRT_GPU_STREAM_STREAM_WRAPPER_H_

#include <cstddef>
#include <memory>
#include <type_traits>

#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Error.h"
#include "tfrt/gpu/stream/cuda_forwards.h"
#include "tfrt/gpu/stream/hip_forwards.h"

namespace tfrt {
namespace gpu {
namespace stream {
// Enum of the abstracted platforms.
enum class Platform {
  NONE,
  CUDA,
  ROCm,
};
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, Platform platform);

// Enums that can safely be static_cast'ed to the corresponding CUDA/HIP types.
enum class CtxFlags {
  SCHED_AUTO = 0x0,
  SCHED_SPIN = 0x1,
  SCHED_YIELD = 0x2,
  SCHED_BLOCKING_SYNC = 0x4,
  MAP_HOST = 0x8,
  LMEM_RESIZE_TO_MAX = 0x10,
};
enum class StreamFlags {
  DEFAULT = 0x0,
  NON_BLOCKING = 0x1,
};
enum class EventFlags {
  DEFAULT = 0x0,
  BLOCKING_SYNC = 0x1,
  DISABLE_TIMING = 0x2,
  INTERPROCESS = 0x4,
};
enum class MemHostAllocFlags {
  DEFAULT = 0x0,
  PORTABLE = 0x1,
  DEVICEMAP = 0x2,
  WRITECOMBINED = 0x4,
};
enum class MemHostRegisterFlags {
  DEFAULT = 0x0,
  PORTABLE = 0x1,
  DEVICEMAP = 0x2,
};
enum class MemAttachFlags {
  GLOBAL = 1,
  HOST = 2,
};
// Print enumerator value to os.
template <typename E>
typename std::enable_if<std::is_enum<E>::value, llvm::raw_ostream>::type&
operator<<(llvm::raw_ostream& os, E item) {
  return os << static_cast<unsigned>(item);
}

namespace internal {
template <typename E>
using IfEnumOperators =
    std::enable_if_t<!std::is_same<E, MemAttachFlags>::value ||
                         std::is_same<E, StreamFlags>::value ||
                         std::is_same<E, EventFlags>::value ||
                         std::is_same<E, MemHostAllocFlags>::value ||
                         std::is_same<E, MemHostRegisterFlags>::value,
                     E>;
}  // namespace internal

// Binary operators for bitmask enums.
template <typename E>
internal::IfEnumOperators<E> operator|(E lhs, E rhs) {
  using underlying = typename std::underlying_type<E>::type;
  return static_cast<E>(static_cast<underlying>(lhs) |
                        static_cast<underlying>(rhs));
}

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

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, Device device) {
    return os << device.device_id_ << " (" << device.platform() << ")";
  }
};

// Resource union type.
template <typename CudaT, typename HipT>
class Resource {
 public:
  Resource() = default;
  Resource(std::nullptr_t, Platform platform) : pair_(nullptr, platform) {}
  // This constructor is for creating invalid Resources, e.g. to use as
  // sentinel values in maps.
  explicit Resource(void* ptr) : pair_(ptr, Platform::NONE) {}
  Resource(CudaT ptr) : pair_(ptr, Platform::CUDA) {}
  Resource(HipT ptr) : pair_(ptr, Platform::ROCm) {}
  // Required for std::unique_ptr<Resource>.
  operator bool() const { return *this != nullptr; }
  operator CudaT() const {
    assert(platform() == Platform::CUDA);
    return static_cast<CudaT>(pair_.getPointer());
  }
  operator HipT() const {
    assert(platform() == Platform::ROCm);
    return static_cast<HipT>(pair_.getPointer());
  }
  Platform platform() const { return pair_.getInt(); }
  bool operator==(std::nullptr_t) const {
    return pair_.getPointer() == nullptr;
  }
  bool operator!=(std::nullptr_t) const {
    return pair_.getPointer() != nullptr;
  }
  bool operator==(Resource other) const { return pair_ == other.pair_; }
  bool operator!=(Resource other) const { return pair_ != other.pair_; }

  size_t hash() const noexcept {
    return std::hash<void*>()(pair_.getOpaqueValue());
  }

 private:
  llvm::PointerIntPair<void*, 2, Platform> pair_;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const Resource& resource) {
    return os << resource.pair_.getPointer() << " (" << resource.platform()
              << ")";
  }
};

// Non-owning handles of GPU resources.
using Context = Resource<CUcontext, hipCtx_t>;
using Module = Resource<CUmodule, hipModule_t>;
using Stream = Resource<CUstream, hipStream_t>;
using Event = Resource<CUevent, hipEvent_t>;
using Function = Resource<CUfunction, hipFunction_t>;

// Serves as a contract that the context has been set on the current thread.
//
// Use CtxSetCurrent() to create an instance. The current context cannot be
// changed while an instance of this class exists in the current
// thread. Instances of this class cannot be transferred across threads. This
// means TFRT kernels cannot produce or consume instances of this class since
// there is no way to force a TFRT kernel to run on a specific host thread.
//
// Code which operates on a context should have access to a Context instance (if
// the context has not yet been set) or a CurrentContext instance (if the
// context has already been set). As a general rule, functions which primarily
// operate on the current context should take a CurrentContext parameter and
// pass it down (all functions in this API which use CUDA's or HIP's internal
// current context state take a CurrentContext parameter). The CurrentContext
// instance should be created somewhere higher up in the call stack at the
// beginning of a section which operates on one single context.
//
// Calling CtxGetCurrent() is discouraged: CtxGetCurrent() can create a
// CurrentContext instance from the internal current context state, but
// tunneling an implicit context through the call stack as internal state is
// bad style. Instead, pass down an instance of Context or CurrentContext as
// explained above.
//
// In NDEBUG builds, this class is trivially copyable and destructible.
class CurrentContext {
#ifndef NDEBUG
  CurrentContext();

 public:
  CurrentContext(const CurrentContext&);
  ~CurrentContext();
#else
  CurrentContext() = default;
#endif

 public:
  CurrentContext& operator=(const CurrentContext&) = default;
  Context context() const;
  Platform platform() const;
  bool operator==(std::nullptr_t) const { return context() == nullptr; }
  bool operator!=(std::nullptr_t) const { return context() != nullptr; }

 private:
  friend CurrentContext CreateCurrentContext();
};
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, CurrentContext current);

// Non-owning handle to GPU memory.
//
// The location is unspecified: the memory may reside on host or device, or
// transparently migrate between the two.
//
// Accessibility is unspecified: the memory may be accessible exclusively from
// host or device, or from both.
//
// Under the (generally true) assumption that all devices support unified
// addressing, the memory may be one of:
// - device-resident memory allocated through cuMemAlloc(). It is accessible by
//   that device and its peers if peer access has been enabled.
// - host-resident memory allocated through MemHostAlloc(). It is accessible by
//   the host and all devices. If allocated with
//   MemHostAllocFlags::WRITECOMBINED, the memory employs disjoint addressing*.
// - ordinary host memory temporarily page-locked through MemHostRegister().
//   It is accessible by the host and all devices using disjoint addressing*.
// - memory allocated through MemAllocManaged() which transparently migrates
//   between host and devices.
//
// * In contrast to unified virtual addressing, disjoint addressing uses one
//   address range to access memory from host and a different (but common)
//   address range to access the same memory from devices. Accessing memory
//   using the wrong address range throws a hardware exception on both host
//   and device. Pointer arguments to Memcpy etc can use either address range.
//
// The type of memory may be queried with PointerGetAttribute().
//
// This class does not model a random access iterator (i.e. fancy pointer) to
// avoid accidentally dereferencing addresses accessible from device only.
template <typename T>
class Pointer {
  // Replacement for llvm::PointerIntPair when T is less than 4 bytes aligned.
  class PointerPlatformPair {
   public:
    PointerPlatformPair() = default;
    PointerPlatformPair(T* ptr, Platform platform)
        : ptr_(ptr), int_(platform) {}
    T* getPointer() const { return ptr_; }
    Platform getInt() const { return int_; }
    void setPointer(T* ptr) { ptr_ = ptr; }
    void setPointerAndInt(T* ptr, Platform platform) {
      ptr_ = ptr;
      int_ = platform;
    }
    bool operator==(const PointerPlatformPair& other) const {
      return ptr_ == other.ptr_ && int_ == other.int_;
    }
    bool operator!=(const PointerPlatformPair& other) const {
      return !operator==(other);
    }

   private:
    T* ptr_ = nullptr;
    Platform int_ = Platform::NONE;
  };

  template <typename U>
  // Bool type iff instances of U* implicitly convert to T*.
  using if_implicitly_convertible_t =
      typename std::enable_if<std::is_convertible<U*, T*>::value, bool>::type;
  // Bool type iff instances of U* can and need be static_cast to T*.
  template <typename U>
  using if_explicitly_convertible_t = typename std::enable_if<
      !std::is_convertible<U*, T*>::value &&
          (std::is_constructible<T*, U*>::value ||
           std::is_same<typename std::remove_cv<U>::type, void>::value),
      bool>::type;

  template <typename U>
  friend class Pointer;

 public:
  Pointer() = default;
  Pointer(std::nullptr_t) : Pointer() {}
  Pointer(T* ptr, Platform platform) : pair_(ptr, platform) {}
  Pointer(const Pointer&) = default;
  template <typename U, if_implicitly_convertible_t<U> = true>
  Pointer(const Pointer<U>& other) : pair_(other.pointer(), other.platform()) {}
  template <typename U, if_explicitly_convertible_t<U> = false>
  explicit Pointer(const Pointer<U>& other)
      : pair_(static_cast<T*>(other.pointer()), other.platform()) {}

  Pointer& operator=(std::nullptr_t) { return *this = Pointer(); }
  Pointer& operator=(const Pointer&) = default;
  template <typename U>
  Pointer& operator=(const Pointer<U>& other) {
    pair_.setPointerAndInt(static_cast<T*>(other.pointer()), other.platform());
    return *this;
  }

  T* raw(Platform platform) const {
    assert(pair_.getInt() == platform);
    return pointer();
  }
  T* raw() const { return pointer(); }
  Platform platform() const { return pair_.getInt(); }

  Pointer operator+(std::ptrdiff_t offset) const {
    return Pointer(pointer() + offset, platform());
  }
  Pointer operator-(std::ptrdiff_t offset) const {
    return Pointer(pointer() - offset, platform());
  }
  Pointer& operator+=(std::ptrdiff_t offset) {
    pair_.setPointer(pointer() + offset);
    return *this;
  }
  Pointer& operator-=(std::ptrdiff_t offset) {
    pair_.setPointer(pointer() - offset);
    return *this;
  }
  Pointer& operator++() { return *this += 1; }
  Pointer& operator--() { return *this -= 1; }
  Pointer operator++(int) {
    Pointer result = *this;
    ++*this;
    return result;
  }
  Pointer operator--(int) {
    Pointer result = *this;
    --*this;
    return result;
  }

  explicit operator bool() const { return *this != nullptr; }
  bool operator!() const { return *this == nullptr; }

  bool operator==(std::nullptr_t) const { return pointer() == nullptr; }
  bool operator!=(std::nullptr_t) const { return pointer() != nullptr; }

  template <typename U>
  bool operator==(const Pointer<U>& other) const {
    return pair_ == other.pair_;
  }
  template <typename U>
  bool operator!=(const Pointer<U>& other) const {
    return pair_ != other.pair_;
  }

  template <typename U>
  bool operator<(const Pointer<U>& other) const {
    return pointer() < other.pointer();
  }
  template <typename U>
  bool operator<=(const Pointer<U>& other) const {
    return pointer() <= other.pointer();
  }

  template <typename U>
  bool operator>(const Pointer<U>& other) const {
    return pointer() > other.pointer();
  }
  template <typename U>
  bool operator>=(const Pointer<U>& other) const {
    return pointer() >= other.pointer();
  }

 private:
  T* pointer() const { return pair_.getPointer(); }

  // Store platform in lower pointer bits if available, otherwise separately.
  typename std::conditional<
      llvm::PointerLikeTypeTraits<T*>::NumLowBitsAvailable >= 2,
      llvm::PointerIntPair<T*, 2, Platform>, PointerPlatformPair>::type pair_;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const Pointer& pointer) {
    return os << pointer.pointer() << " (" << pointer.platform() << ")";
  }
};

namespace internal {
// Helper to wrap resources and memory into RAII types.
struct ContextDeleter {
  using pointer = Context;
  void operator()(Context context) const;
};
struct ModuleDeleter {
  using pointer = Module;
  void operator()(Module module) const;
};
struct StreamDeleter {
  using pointer = Stream;
  void operator()(Stream stream) const;
};
struct EventDeleter {
  using pointer = Event;
  void operator()(Event event) const;
};
template <typename Deleter>
using OwningResource = std::unique_ptr<typename Deleter::pointer, Deleter>;

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
using OwningContext = internal::OwningResource<internal::ContextDeleter>;
using OwningModule = internal::OwningResource<internal::ModuleDeleter>;
using OwningStream = internal::OwningResource<internal::StreamDeleter>;
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
  CtxFlags flags;
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
llvm::Error DevicePrimaryCtxSetFlags(Device device, CtxFlags flags);

llvm::Expected<OwningContext> CtxCreate(CtxFlags flags, Device device);
llvm::Error CtxDestroy(Context context);
// Avoid if possible. See documentation of CurrentContext.
llvm::Expected<CurrentContext> CtxGetCurrent();
llvm::Expected<CurrentContext> CtxSetCurrent(Context context);
llvm::Error CtxSynchronize(CurrentContext current);
llvm::Expected<unsigned> CtxGetApiVersion(Context context);
llvm::Expected<Device> CtxGetDevice(CurrentContext current);
llvm::Expected<CtxFlags> CtxGetFlags(CurrentContext current);
llvm::Expected<StreamPriorityRange> CtxGetStreamPriorityRange(
    CurrentContext current);

llvm::Expected<OwningStream> StreamCreate(CurrentContext current,
                                          StreamFlags flags);
llvm::Expected<OwningStream> StreamCreate(CurrentContext current,
                                          StreamFlags flags, int priority);
llvm::Error StreamDestroy(Stream stream);
llvm::Expected<int> StreamGetPriority(Stream stream);
llvm::Expected<StreamFlags> StreamGetFlags(Stream stream);
llvm::Error StreamSynchronize(Stream stream);
llvm::Expected<bool> StreamQuery(Stream stream);
llvm::Error StreamWaitEvent(Stream stream, Event event);

llvm::Expected<OwningEvent> EventCreate(CurrentContext current,
                                        EventFlags flags);
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
llvm::Error MemHostFree(Pointer<void> pointer);
llvm::Expected<RegisteredMemory<void>> MemHostRegister(
    CurrentContext current, void* ptr, size_t size_bytes,
    MemHostRegisterFlags flags);
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
llvm::Error ModuleUnload(Module module);
llvm::Expected<Function> ModuleGetFunction(Module module, const char* name);

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
                      shared_memory_size_bytes, stream, arg_ptrs, nullptr);
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

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_STREAM_STREAM_WRAPPER_H_
