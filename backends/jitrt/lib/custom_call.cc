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

//===- custom_call.cc - ---------------------------------------------------===//
// JitRt custom calls library.
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/custom_call.h"

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

namespace tfrt {
namespace jitrt {

using mlir::failure;
using mlir::TypeID;

raw_ostream& operator<<(raw_ostream& os, const FlatMemrefView& view) {
  return os << "FlatMemrefView: dtype: " << view.dtype
            << " size_in_bytes: " << view.size_in_bytes;
}

struct CustomCallRegistry::Impl {
  llvm::StringMap<std::unique_ptr<CustomCall>> custom_calls;
};

CustomCallRegistry::CustomCallRegistry() : impl_(std::make_unique<Impl>()) {}

void CustomCallRegistry::Register(std::unique_ptr<CustomCall> custom_call) {
  llvm::StringRef key = custom_call->name();
  auto inserted = impl_->custom_calls.insert({key, std::move(custom_call)});
  assert(inserted.second && "duplicate custom call registration");
  (void)inserted;
}

CustomCall* CustomCallRegistry::Find(llvm::StringRef callee) const {
  auto it = impl_->custom_calls.find(callee);
  if (it == impl_->custom_calls.end()) return nullptr;
  return it->second.get();
}

static std::vector<CustomCallRegistry::RegistrationFunction>*
GetCustomCallRegistrations() {
  static auto* ret = new std::vector<CustomCallRegistry::RegistrationFunction>;
  return ret;
}

void RegisterStaticCustomCalls(CustomCallRegistry* custom_call_registry) {
  for (auto func : *GetCustomCallRegistrations()) func(custom_call_registry);
}

void AddStaticCustomCallRegistration(
    CustomCallRegistry::RegistrationFunction registration) {
  GetCustomCallRegistrations()->push_back(registration);
}

static mlir::FailureOr<DType> ScalarTypeIdToDType(TypeID type_id) {
  if (TypeID::get<uint8_t>() == type_id) return DType::UI8;
  if (TypeID::get<uint32_t>() == type_id) return DType::UI32;
  if (TypeID::get<uint64_t>() == type_id) return DType::UI64;
  if (TypeID::get<int32_t>() == type_id) return DType::I32;
  if (TypeID::get<int64_t>() == type_id) return DType::I64;
  if (TypeID::get<float>() == type_id) return DType::F32;
  if (TypeID::get<double>() == type_id) return DType::F64;
  assert(false && "unsupported data type");
  return failure();
}

template <typename T, int rank>
static ArrayRef<int64_t> Sizes(StridedMemRefType<T, rank>* memref) {
  return llvm::makeArrayRef(memref->sizes);
}

template <typename T>
static ArrayRef<int64_t> Sizes(StridedMemRefType<T, 0>* memref) {
  return {};
}

template <typename T, int rank>
static ArrayRef<int64_t> Strides(StridedMemRefType<T, rank>* memref) {
  return llvm::makeArrayRef(memref->strides);
}

template <typename T>
static ArrayRef<int64_t> Strides(StridedMemRefType<T, 0>* memref) {
  return {};
}

mlir::FailureOr<MemrefDesc> CustomCallArgDecoding<MemrefDesc>::Decode(
    mlir::TypeID type_id, void* value) {
  // Check that encoded value holds the correct type id.
  if (type_id != mlir::TypeID::get<MemrefDesc>()) return failure();

  // Cast opaque memory to exected encoding.
  auto* encoded = reinterpret_cast<internal::EncodedMemref*>(value);

  // Get the memref element data type.
  void* opaque = reinterpret_cast<void*>(encoded->element_type_id);
  TypeID element_type_id = TypeID::getFromOpaquePointer(opaque);

  auto dtype = ScalarTypeIdToDType(element_type_id);
  if (mlir::failed(dtype)) return failure();

  // Unpack the StridedMemRefType into the MemrefDesc.
  auto unpack_strided_memref = [&](auto rank_tag) -> MemrefDesc {
    constexpr int rank = decltype(rank_tag)::value;

    using Descriptor = StridedMemRefType<float, rank>;
    auto* descriptor = reinterpret_cast<Descriptor*>(encoded->descriptor);

    return MemrefDesc(*dtype, descriptor->data, descriptor->offset,
                      Sizes(descriptor), Strides(descriptor));
  };

  // Dispatch based on the memref rank.
  switch (encoded->rank) {
    case 0:
      return unpack_strided_memref(std::integral_constant<int, 0>{});
    case 1:
      return unpack_strided_memref(std::integral_constant<int, 1>{});
    case 2:
      return unpack_strided_memref(std::integral_constant<int, 2>{});
    case 3:
      return unpack_strided_memref(std::integral_constant<int, 3>{});
    case 4:
      return unpack_strided_memref(std::integral_constant<int, 4>{});
    case 5:
      return unpack_strided_memref(std::integral_constant<int, 5>{});
    default:
      assert(false && "unsupported memref rank");
      return failure();
  }
}

mlir::FailureOr<FlatMemrefView> CustomCallArgDecoding<FlatMemrefView>::Decode(
    mlir::TypeID type_id, void* value) {
  FlatMemrefView memref;

  // Check that the encoded value holds the correct type id.
  if (type_id != TypeID::get<MemrefDesc>()) return failure();

  // Cast opaque memory to the encoded memref.
  auto* encoded = reinterpret_cast<internal::EncodedMemref*>(value);

  // Get the memref element data type.
  void* opaque = reinterpret_cast<void*>(encoded->element_type_id);
  TypeID element_type_id = TypeID::getFromOpaquePointer(opaque);

  auto dtype = ScalarTypeIdToDType(element_type_id);
  if (mlir::failed(dtype)) return failure();
  memref.dtype = *dtype;

  // Unpack the StridedMemRefType into the FlatMemrefView.
  auto unpack_strided_memref = [&](auto rank_tag) {
    constexpr int rank = decltype(rank_tag)::value;

    using Descriptor = StridedMemRefType<float, rank>;
    auto* descriptor = reinterpret_cast<Descriptor*>(encoded->descriptor);

    memref.data = descriptor->data;
    memref.size_in_bytes =
        std::accumulate(Sizes(descriptor).begin(), Sizes(descriptor).end(),
                        GetHostSize(memref.dtype), std::multiplies<int64_t>());
  };

  // Dispatch based on the memref rank.
  switch (encoded->rank) {
    case 0:
      unpack_strided_memref(std::integral_constant<int, 0>{});
      break;
    case 1:
      unpack_strided_memref(std::integral_constant<int, 1>{});
      break;
    case 2:
      unpack_strided_memref(std::integral_constant<int, 2>{});
      break;
    case 3:
      unpack_strided_memref(std::integral_constant<int, 3>{});
      break;
    case 4:
      unpack_strided_memref(std::integral_constant<int, 4>{});
      break;
    case 5:
      unpack_strided_memref(std::integral_constant<int, 5>{});
      break;
    default:
      assert(false && "unsupported memref rank");
      return failure();
  }

  return memref;
}

}  // namespace jitrt
}  // namespace tfrt
