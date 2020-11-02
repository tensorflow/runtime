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

//===- remote_device.h - Remote Device --------------------------*- C++ -*-===//
//
// This file defines RemoteCpuDevice which represents a Cpu Device in a remote
// worker
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_
#define TFRT_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_

#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/host_context/device.h"

namespace tfrt {
class RemoteCpuDevice : public Device, public DeviceTraits<RemoteCpuDevice> {
 public:
  static const char* type_name() {
    static constexpr char kName[] = "remote_cpu";
    return kName;
  }

  explicit RemoteCpuDevice(string_view name, HostId host_id)
      : Device(kDeviceType, name), host_id_(host_id) {}

  ~RemoteCpuDevice() override {}

  HostId GetHostId() const { return host_id_; }

 private:
  // The Host where this device resides.
  HostId host_id_;
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_