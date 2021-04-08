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

// This file contains implementation of creating remote devices.

#include "tfrt/distributed_runtime/remote_device.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace {
constexpr char kRemoteDeviceTypePrefix[] = "remote_";

// Singleton to help construct remote device of the specified types.
class RemoteDeviceMaker {
 public:
  RemoteDeviceMaker(const RemoteDeviceMaker&) = delete;
  RemoteDeviceMaker& operator=(const RemoteDeviceMaker&) = delete;

  using MakeRemoteDeviceFunc = ::llvm::unique_function<Expected<RemoteDevice*>(
      string_view name, TaskHandle task_handle)>;
  static Expected<RemoteDevice*> MakeTypedRemoteDevice(string_view name,
                                                       string_view type,
                                                       TaskHandle task_handle) {
    static RemoteDeviceMaker* maker = new RemoteDeviceMaker();

    // When creating remote device, the type name can be with or without the
    // "remote_" prefix, depending on how the device info is serialized:
    // (1) When the master sends GetDevices RPC, remote workers serialize their
    //     local devices in GetDevicesResponse. When the master adds them as
    //     RemoteDevices, the device type name will not have the "remote_"
    //     prefix since they were serialized from worker's host device manager;
    // (2) When the master sends CreateContext and serializes all devices in
    //     distributed context's cluster device manager, workers will see device
    //     type name with "remote_" prefix when adding them as RemoteDevices.
    std::string remote_type = type.str();
    if (!type.startswith(kRemoteDeviceTypePrefix)) {
      remote_type = StrCat(kRemoteDeviceTypePrefix + type);
    }
    auto it = maker->remote_device_type_func_.find(remote_type);
    if (it == maker->remote_device_type_func_.end()) {
      return llvm::make_error<UnimplementedErrorInfo>(
          StrCat("Remote device ", name, " of type ", type,
                 " is not supported in distributed runtime."));
    }
    return it->second(name, task_handle);
  }

 private:
  RemoteDeviceMaker() {
    remote_device_type_func_.try_emplace(
        RemoteCpuDevice::type_name(),
        [](string_view name, TaskHandle task_handle) {
          return new RemoteCpuDevice(name, task_handle);
        });
    remote_device_type_func_.try_emplace(
        RemoteTpuDevice::type_name(),
        [](string_view name, TaskHandle task_handle) {
          return new RemoteTpuDevice(name, task_handle);
        });
    remote_device_type_func_.try_emplace(
        RemoteTpuSystemDevice::type_name(),
        [](string_view name, TaskHandle task_handle) {
          return new RemoteTpuSystemDevice(name, task_handle);
        });
  }
  llvm::StringMap<MakeRemoteDeviceFunc> remote_device_type_func_;
};
}  // namespace

Expected<Device*> NewRemoteDevice(string_view name, string_view type,
                                  TaskHandle task_handle) {
  return RemoteDeviceMaker::MakeTypedRemoteDevice(name, type, task_handle);
}
}  // namespace tfrt
