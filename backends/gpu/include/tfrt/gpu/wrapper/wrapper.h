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

// Thin abstraction layer for CUDA and HIP.
//
// The wrapper provides a safer and more convenient access to CUDA and ROCm
// platform APIs as a very thin layer. It also provides an abstraction of the
// two platforms where their functionality overlap. For this, resources
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
#ifndef TFRT_GPU_WRAPPER_WRAPPER_H_
#define TFRT_GPU_WRAPPER_WRAPPER_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>

#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Error.h"
#include "tfrt/gpu/wrapper/cuda_forwards.h"
#include "tfrt/gpu/wrapper/hip_forwards.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace gpu {
namespace wrapper {
// Enum of the abstracted platforms.
enum class Platform {
  NONE,
  CUDA,
  ROCm,
};

template <typename T>
Expected<T> Parse(llvm::StringRef);
template <>
Expected<Platform> Parse<Platform>(llvm::StringRef platform);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, Platform platform);

namespace internal {
// Struct capturing a failed API call with result code type T.
template <typename T>
struct ErrorData {
  T result;
  const char* expr;
  StackTrace stack_trace;
};

template <typename T, typename... Args>
Error MakeError(T result, const char* expr, Args... args) {
  return llvm::make_error<TupleErrorInfo<ErrorData<T>>>(ErrorData<T>{
      result, expr, CreateStackTrace(), std::forward<Args>(args)...});
}

// Write ErrorData to raw_ostream.
template <typename T>
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const ErrorData<T>& data);

template <bool... Bs>
using AllFalse = std::is_same<std::integer_sequence<bool, false, Bs...>,
                              std::integer_sequence<bool, Bs..., false>>;
}  // namespace internal

// llvm::ErrorInfoBase payload of errors created with 'MakeError'.
template <typename T>
using ErrorInfo = TupleErrorInfo<internal::ErrorData<T>>;

// Create error from failed API call 'expr' with returned 'result' code.
template <typename T>
Error MakeError(T result, const char* expr) {
  return internal::MakeError(result, expr);
}

// Return result code contained in above 'ErrorInfo'.
template <typename T>
T GetResult(const ErrorInfo<T>& info) {
  return info.template get<internal::ErrorData<T>>().result;
}

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
  Resource& operator=(std::nullptr_t) {
    pair_.setPointer(nullptr);
    return *this;
  }
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

  // For member access from std::unique_ptr.
  const Resource* operator->() const { return this; }

  // Free function which hashes a resource instance.
  friend std::size_t hash(const Resource& resource) {
    return std::hash<void*>{}(resource.pair_.getOpaqueValue());
  }

 private:
  llvm::PointerIntPair<void*, 2, Platform> pair_;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const Resource& resource) {
    return os << resource.pair_.getPointer() << " (" << resource.platform()
              << ")";
  }
  friend class std::hash<Resource>;
};

template <Platform platform>
using PlatformType = std::integral_constant<Platform, platform>;

using CudaPlatformType = PlatformType<Platform::CUDA>;
using RocmPlatformType = PlatformType<Platform::ROCm>;

// Maps platform-specific type T for Tag to CUDA or ROCm.
template <typename Tag, typename T>
struct PlatformTypeTraits : public PlatformType<Platform::NONE> {};

// Enum union type.
//
// For a CUDA/ROCm pair of enums with different enumerators, instantiate
// this template with an opaque tag type (e.g. `struct FooTag;`) and specialize
// the PlatformTypeTraits struct in the CUDA/ROCm wrapper header, e.g.:
// template <>
// PlatformTypeTraits<FooTag, cudaFooEnum> : public CudaPlatformType {};
//
// Tag may define a 'type' member to override the value's type (default is int).
template <typename Tag>
class Enum {
  template <typename T, typename Tag_>
  using IsCudaOrRocm =
      std::enable_if_t<PlatformTypeTraits<Tag_, T>::value != Platform::NONE,
                       int>;

  template <typename T>
  static constexpr auto get_value_type(T*) -> typename T::type;
  static constexpr auto get_value_type(...) -> int;  // defaults to int.
  using ValueType = decltype(get_value_type(static_cast<Tag*>(nullptr)));

 public:
  Enum() : Enum({}, Platform::NONE) {}
  Enum(ValueType value, Platform platform)
      : value_(value), platform_(platform) {}
  template <typename T, typename Tag_ = Tag, IsCudaOrRocm<T, Tag_> = 0>
  Enum(T value)
      : Enum(static_cast<ValueType>(value),
             PlatformTypeTraits<Tag_, T>::value) {}
  template <typename T, typename Tag_ = Tag, IsCudaOrRocm<T, Tag_> = 0>
  operator T() const {
    using PlatformType = PlatformTypeTraits<Tag_, T>;
    assert(platform_ == PlatformType::value);
    return static_cast<T>(value_);
  }
  Platform platform() const { return platform_; }
  bool operator==(Enum other) const {
    return value_ == other.value_ && platform_ == other.platform_;
  }
  bool operator!=(Enum other) const { return !(*this == other); }
  template <typename T, typename Tag_ = Tag, IsCudaOrRocm<T, Tag_> = 0>
  bool operator==(T value) const {
    return Enum(value) == *this;
  }
  template <typename T, typename Tag_ = Tag, IsCudaOrRocm<T, Tag_> = 0>
  bool operator!=(T value) const {
    return Enum(value) != *this;
  }

  ValueType ToOpaqueValue() const {
    auto result = value_ << 2 | static_cast<ValueType>(platform_);
    assert(*this == FromOpaqueValue(result) && "roundtrip failed");
    return result;
  }
  static Enum FromOpaqueValue(ValueType opaque) {
    return Enum(opaque >> 2, static_cast<Platform>(opaque & 0x3));
  }

 private:
  ValueType value_;
  Platform platform_;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const Enum& pair) {
    return os << pair.value_ << " (" << pair.platform_ << ")";
  }
};

// Non-owning handles of GPU resources. This header only exposes the types that
// are commonly used by libraries. See the *_wrapper.h files for more specific
// resource types.
using Context = Resource<CUcontext, hipCtx_t>;
// Note: does not support CU_STREAM_LEGACY (0x1) or CU_STREAM_PER_THREAD (0x2).
// Those special handle values use the lower two bits and therefore interfere
// with PointerIntPair. Will fail runtime asserts if used.
using Stream = Resource<CUstream, hipStream_t>;

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
#ifdef NDEBUG
  CurrentContext() = default;
#else
  CurrentContext();

 public:
  CurrentContext(const CurrentContext&);
  ~CurrentContext();
#endif

 public:
  // Restrict construction to implementation-defined factory.
  struct Factory;

 public:
  CurrentContext& operator=(const CurrentContext&) = default;
  Context context() const;
  Platform platform() const;
  bool operator==(std::nullptr_t) const { return context() == nullptr; }
  bool operator!=(std::nullptr_t) const { return context() != nullptr; }
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
  T* raw() const { return pointer(); }  // TODO(csigg): Remove.
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
struct StreamDeleter {
  using pointer = Stream;
  void operator()(Stream stream) const;
};
template <typename Deleter>
using OwningResource = std::unique_ptr<typename Deleter::pointer, Deleter>;
}  // namespace internal

// RAII wrappers for resources. Instances own the underlying resource.
//
// They are implemented as std::unique_ptrs with custom deleters.
//
// Use get() and release() to access the non-owning handle, please use with
// appropriate care.
using OwningContext = internal::OwningResource<internal::ContextDeleter>;
using OwningStream = internal::OwningResource<internal::StreamDeleter>;

struct LibraryVersion {
  int major;
  int minor;
  int patch;
};

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_WRAPPER_H_
