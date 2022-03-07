/*
 * Copyright 2022 The TensorFlow Runtime Authors
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

// Provides types used in ccl_wrapper.h without including nccl.h. This allows
// TensorFlow (which includes nccl.h from NCCL or RCCL, depending on the
// configuration) to safely include this file in any order. On the other hand,
// ccl_wrapper.h cannot be included before nccl.h from RCCL.
#ifndef TFRT_GPU_WRAPPER_CCL_TYPES_H_
#define TFRT_GPU_WRAPPER_CCL_TYPES_H_

#include <type_traits>

#include "tfrt/gpu/wrapper/wrapper.h"

namespace llvm {
template <>
struct PointerLikeTypeTraits<ncclComm_t> {
  static void* getAsVoidPointer(ncclComm_t comm) { return comm; }
  static ncclComm_t getFromVoidPointer(void* ptr) {
    return static_cast<ncclComm_t>(ptr);
  }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr int NumLowBitsAvailable = 2;
};
}  // namespace llvm

namespace tfrt {
namespace gpu {
namespace wrapper {

namespace internal {
template <typename Tag, typename T>
class IsCclType : public std::false_type {};
}  // namespace internal

// Similar to wrapper::Enum, provides a class that is explicitly constructible
// from and implicitly convertible to a type that is only defined later. We
// cannot use wrapper::Enum because NCCL and RCCL use the same types.
template <typename ValueType, typename Tag>
class CclType {
  template <typename T>
  using EnableIf =
      std::enable_if_t<internal::IsCclType<CclType, T>::value, int>;

  explicit CclType(ValueType value) : value_(value) {}

 public:
  CclType() = default;
  template <typename T, EnableIf<T> = 0>
  // NOLINTNEXTLINE(google-explicit-constructor)
  CclType(T value) : value_(reinterpret_cast<const ValueType&>(value)) {}
  template <typename T, EnableIf<T> = 0>
  operator T() const {  // NOLINT(google-explicit-constructor)
    return reinterpret_cast<const T&>(value_);
  }

  ValueType ToOpaqueValue() const { return value_; }
  static CclType FromOpaqueValue(ValueType opaque) { return CclType(opaque); }

  static Expected<CclType> Parse(llvm::StringRef name) {
    auto result = internal::EnumStream<CclType, Platform::NONE>::Parse(name);
    if (result) return CclType(*result);
    return result.takeError();
  }

  friend raw_ostream& operator<<(raw_ostream& os, const CclType& value) {
    return internal::EnumStream<CclType, Platform::NONE>::Print(os, value);
  }

 private:
  ValueType value_;
};

struct CclUniqueIdTag;
using CclUniqueId = CclType<std::array<char, 128>, CclUniqueIdTag>;

struct CclDataTypeTag;
using CclDataType = CclType<int, CclDataTypeTag>;

struct CclReductionOpTag;
using CclReductionOp = CclType<int, CclReductionOpTag>;

// Non-owning NCCL communicator for a specific platform.
class CclComm {
 public:
  CclComm() = default;
  explicit CclComm(std::nullptr_t) : pair_(nullptr, Platform::NONE) {}
  CclComm(ncclComm_t comm, Platform platform) : pair_(comm, platform) {}
  // Required for std::unique_ptr<Resource>.
  CclComm& operator=(std::nullptr_t) {
    pair_.setPointer(nullptr);
    return *this;
  }
  // Required for std::unique_ptr<Resource>.
  operator bool() const {  // NOLINT(google-explicit-constructor)
    return *this != nullptr;
  }
  operator ncclComm_t() const {  // NOLINT(google-explicit-constructor)
    return static_cast<ncclComm_t>(pair_.getPointer());
  }
  Platform platform() const { return pair_.getInt(); }
  bool operator==(std::nullptr_t) const {
    return pair_.getPointer() == nullptr;
  }
  bool operator!=(std::nullptr_t) const {
    return pair_.getPointer() != nullptr;
  }
  bool operator==(CclComm other) const { return pair_ == other.pair_; }
  bool operator!=(CclComm other) const { return pair_ != other.pair_; }

  // For member access from std::unique_ptr.
  const CclComm* operator->() const { return this; }

 private:
  llvm::PointerIntPair<ncclComm_t, 2, Platform> pair_;

  friend raw_ostream& operator<<(raw_ostream& os, const CclComm& comm) {
    return os << comm.pair_.getPointer() << " (" << comm.platform() << ")";
  }
};

namespace internal {
// Helper to wrap resources and memory into RAII types.
struct CclCommDeleter {
  using pointer = CclComm;
  void operator()(CclComm comm) const;
};
}  // namespace internal

// RAII wrappers for resources. Instances own the underlying resource.
//
// They are implemented as std::unique_ptrs with custom deleters.
//
// Use get() and release() to access the non-owning handle, please use with
// appropriate care.
using OwningCclComm = internal::OwningResource<internal::CclCommDeleter>;

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CCL_TYPES_H_
