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

//===- device/device_util.cc ----------------------------------------------===//
//
// This file implements GPU device util functions for managing GPU devices.
//
//===----------------------------------------------------------------------===//
#include "tfrt/gpu/device/device_util.h"

#include "tfrt/host_context/device.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/string_util.h"

namespace tfrt {
namespace gpu {

RCReference<Device> CreateGpuDevice(int gpu_ordinal, HostContext* host) {
  static DeviceTypeRegistration register_device_type_cpu("gpu");
  auto device_name = StrCat("GPU:", gpu_ordinal);
  auto existing_device = host->GetDeviceManager()->GetDeviceRef(device_name);
  if (existing_device) {
    return existing_device;
  }
  return host->GetDeviceManager()->MaybeAddDevice(
      TakeRef(new Device(GetStaticDeviceType("gpu"), device_name)));
}

}  // namespace gpu
}  // namespace tfrt
