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

//===- tensor_type_registration.cc ----------------------------------------===//
//
// This file implements Tensor type and its registration.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tensor/tensor_type_registration.h"

#include <limits>
#include <tuple>

#include "llvm/Support/raw_ostream.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const TensorType& tensor_type) {
  os << "TensorType<name=" << tensor_type.name() << ">";
  return os;
}

// Registry that contains all the TensorType registered.
class TensorTypeRegistry {
 public:
  TensorTypeRegistry(const TensorTypeRegistry&) = delete;
  TensorTypeRegistry& operator=(const TensorTypeRegistry&) = delete;

  // Each process should only have one TensorTypeRegistry.
  static TensorTypeRegistry* GetInstance();

  // Registers a new tensor type name. The tensor type name must be unique and
  // haven't registered before.
  TensorType RegisterTensorType(string_view tensor_type_name);

  // Returns the registered tensor type.
  TensorType GetRegisteredTensorType(string_view tensor_type_name) const;

  string_view GetTensorTypeName(TensorType tensor_type) const;

 private:
  TensorTypeRegistry() = default;
  // TODO(b/165864222): change to reader-write lock instead of mutex.
  mutable mutex mu_;

  llvm::SmallVector<std::string, 10> name_list_ TFRT_GUARDED_BY(mu_);
  llvm::StringMap<int8_t> name_to_id_map_ TFRT_GUARDED_BY(mu_);
};

TensorTypeRegistry* TensorTypeRegistry::GetInstance() {
  static TensorTypeRegistry* registry = new TensorTypeRegistry();
  return registry;
}

TensorType TensorTypeRegistry::RegisterTensorType(
    string_view tensor_type_name) {
  mutex_lock l(mu_);
  auto result = name_to_id_map_.try_emplace(tensor_type_name);
  assert(result.second && "Re-registering existing tensor type name");
  result.first->second = name_list_.size();
  name_list_.emplace_back(tensor_type_name);
  return TensorType(result.first->second);
}

TensorType TensorTypeRegistry::GetRegisteredTensorType(
    string_view tensor_type_name) const {
  mutex_lock l(mu_);
  llvm::StringMap<int8_t>::const_iterator itr =
      name_to_id_map_.find(tensor_type_name);
  assert((itr != name_to_id_map_.end()) && "Invalid tensor type name");
  return TensorType(itr->second);
}

string_view TensorTypeRegistry::GetTensorTypeName(
    TensorType tensor_type) const {
  mutex_lock l(mu_);
  int8_t id = tensor_type.id();
  assert(id < name_list_.size() && "Invalid tensor type id");
  return name_list_[id];
}

TensorType RegisterStaticTensorType(string_view tensor_type_name) {
  return TensorTypeRegistry::GetInstance()->RegisterTensorType(
      tensor_type_name);
}

TensorType GetStaticTensorType(string_view tensor_type_name) {
  return TensorTypeRegistry::GetInstance()->GetRegisteredTensorType(
      tensor_type_name);
}

string_view TensorType::name() const {
  return TensorTypeRegistry::GetInstance()->GetTensorTypeName(*this);
}

const TensorType TensorType::kUnknownTensorType =
    RegisterStaticTensorType("Unknown");

}  // namespace tfrt
