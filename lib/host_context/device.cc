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

//===- device.cc ------------------------------------------------*- C++ -*-===//
//
// This file contains implementation of Device abstraction.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/device.h"

#include "llvm/ADT/FunctionExtras.h"

namespace tfrt {

void DeviceTypeRegistry::RegisterDeviceType(string_view type) {
  for (auto& dt : types_) {
    if (dt.name() == type) {
      assert(false && "re-registered existing device type.");
    }
  }
  types_.push_back(DeviceType(type));
}

const DeviceType& DeviceTypeRegistry::GetDeviceType(string_view type) const {
  for (auto& dt : types_) {
    if (dt.name() == type) {
      return dt;
    }
  }
  llvm_unreachable("invalid device type");
}

DeviceTypeRegistry* DeviceTypeRegistry::GetStaticDeviceTypeRegistry() {
  static DeviceTypeRegistry* ret = new DeviceTypeRegistry();
  return ret;
}

DeviceTypeRegistration::DeviceTypeRegistration(string_view name) {
  DeviceTypeRegistry::GetStaticDeviceTypeRegistry()->RegisterDeviceType(name);
}

const DeviceType& GetStaticDeviceType(string_view type) {
  return DeviceTypeRegistry::GetStaticDeviceTypeRegistry()->GetDeviceType(type);
}
}  // namespace tfrt
