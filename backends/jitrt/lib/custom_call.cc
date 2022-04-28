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

namespace tfrt {
namespace jitrt {

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

namespace internal {

// Decode arguments encoded by the `rt-to-llvm` pass. Decoding/encoding scheme
// must be consistent between lowering to LLVM pass and this function.
llvm::SmallVector<DecodedArg> DecodeArgs(void** args) {
  int64_t num_args = *reinterpret_cast<int64_t*>(args[0]);

  llvm::SmallVector<DecodedArg> decoded;
  decoded.reserve(num_args);

  for (int64_t i = 0; i < num_args; ++i) {
    void** arg_base = args + 1 + i * 2;

    DecodedArg arg;
    arg.type_id = DecodeTypeid(arg_base[0]);
    arg.value = arg_base[1];

    decoded.push_back(arg);
  }

  return decoded;
}

mlir::TypeID DecodeTypeid(void* type_id) {
  std::uintptr_t encoded_type_id = *reinterpret_cast<std::uintptr_t*>(type_id);
  void* opaque_type_id = reinterpret_cast<void*>(encoded_type_id);
  return mlir::TypeID::getFromOpaquePointer(opaque_type_id);
}

mlir::FailureOr<DType> TypeIdToDType(mlir::TypeID type_id) {
  if (mlir::TypeID::get<float>() == type_id) return DType::F32;
  assert(false && "unsupported data type");
  return mlir::failure();
}

}  // namespace internal
}  // namespace jitrt
}  // namespace tfrt
