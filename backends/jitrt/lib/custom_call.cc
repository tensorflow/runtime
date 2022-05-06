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

raw_ostream& operator<<(raw_ostream& os, const MemrefView& view) {
  auto print_arr = [&](string_view name, ArrayRef<int64_t> arr) {
    os << " " << name << ": [";
    if (!arr.empty()) {
      os << arr[0];
      for (int i = 1; i < arr.size(); ++i) os << ", " << arr[i];
    }
    os << "]";
  };

  os << "MemrefView: dtype: " << view.dtype << " offset: " << view.offset;
  print_arr("sizes", view.sizes);
  print_arr("strides", view.strides);

  return os;
}

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

}  // namespace jitrt
}  // namespace tfrt
