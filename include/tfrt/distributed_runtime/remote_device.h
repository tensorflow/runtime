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

// Remote Device
//
// This file defines remote devices, which represent devices on remote tasks.
#ifndef TFRT_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_
#define TFRT_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_

#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/host_context/device.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
class RemoteDevice : public Device {
 public:
  RemoteDevice(const DeviceType& type, string_view name, TaskHandle task_handle)
      : Device(type, name), task_handle_(task_handle) {}

  TaskHandle GetTaskHandle() const { return task_handle_; }

 private:
  // The remote task where this device resides.
  TaskHandle task_handle_;
};

class RemoteCpuDevice : public RemoteDevice,
                        public DeviceTraits<RemoteCpuDevice> {
 public:
  static const char* type_name() {
    static constexpr char kName[] = "remote_cpu";
    return kName;
  }

  explicit RemoteCpuDevice(string_view name, TaskHandle task_handle)
      : RemoteDevice(kDeviceType, name, task_handle) {}

  ~RemoteCpuDevice() override {}
};

class RemoteTpuDevice : public RemoteDevice,
                        public DeviceTraits<RemoteTpuDevice> {
 public:
  static const char* type_name() {
    static constexpr char kName[] = "remote_tpu";
    return kName;
  }

  explicit RemoteTpuDevice(string_view name, TaskHandle task_handle)
      : RemoteDevice(kDeviceType, name, task_handle) {}

  ~RemoteTpuDevice() override {}
};

// This is a vritual device. If a host has TPU connected, it will have one
// instance of this device. Current TF has the same design.
class RemoteTpuSystemDevice : public RemoteDevice,
                              public DeviceTraits<RemoteTpuSystemDevice> {
 public:
  static const char* type_name() {
    static constexpr char kName[] = "remote_tpu_system";
    return kName;
  }

  explicit RemoteTpuSystemDevice(string_view name, TaskHandle task_handle)
      : RemoteDevice(kDeviceType, name, task_handle) {}

  ~RemoteTpuSystemDevice() override {}
};

// TODO(tfrt-dev): Add remote GPU device when there are valid use cases.

Expected<Device*> NewRemoteDevice(string_view name, string_view type,
                                  TaskHandle task_handle);

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_
