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

//===- device.h - device abstraction ----------------------------*- C++ -*-===//
//
// Device abstract that are used in both op-by-op and graph execution.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_HOST_CONTEXT_DEVICE_H_
#define TFRT_HOST_CONTEXT_DEVICE_H_

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {

class HostContext;
class DeviceTypeRegistry;

// A thin wrapper of device type string like "gpu", "cpu", etc. We don't use
// enum because it requires a central place that lists all the supported value
// which is not very extensible.
class DeviceType {
 public:
  string_view name() const { return name_; }
  bool IsValid() const { return !name_.empty(); }

  bool operator==(const DeviceType& other) const {
    // We can compare their address directly because all instances are managed
    // by DeviceTypeRegistry.
    return this == &other;
  }

 private:
  friend DeviceTypeRegistry;

  // Default constructor that creates an invalid DeviceType.
  DeviceType() {}

  // It is hidden because all instances are managed by DeviceTypeRegistry.
  explicit DeviceType(string_view name) : name_(name) {}

  std::string name_;
};

// Represents a device that a tensor can be placed on. E.g. "gpu:0", "tpu:1".
class Device : public ReferenceCounted<Device> {
 public:
  Device(const DeviceType& type, string_view name) : type_(type), name_(name) {
    assert(type_.IsValid() &&
           "Cannot create a Device with invalid device type");
    assert(!name_.empty() && "Cannot create a Device with empty device name");
  }

  ~Device() {}

  // This class is not copyable or assignable. If we add a copy operation it
  // will likely be explicit - copying a Tensor can be a very expensive
  // operation.
  Device(const Device& other) = delete;
  Device& operator=(const Device&) = delete;

  const DeviceType& type() const { return type_; }
  string_view name() const { return name_; }

 private:
  const DeviceType& type_;
  std::string name_;
};

// A central place to manage all active devices.
class DeviceManager {
 public:
  ~DeviceManager() = default;
  DeviceManager(const DeviceManager&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;

  // Add a new device if it doesn't exist. If it doesn't exist, return the newly
  // added device, otherwise, return the existing device.
  RCReference<Device> MaybeAddDevice(RCReference<Device> device);

  // Lookup a device by its name. Return an empty RCReference if not found.
  RCReference<Device> GetDeviceRef(string_view device_name) const;

 private:
  DeviceManager() = default;
  friend class HostContext;

  mutable mutex mu_;
  llvm::StringMap<RCReference<Device>> device_map_ TFRT_GUARDED_BY(mu_);
};

// Contains all the DeviceType that are supported.
class DeviceTypeRegistry {
 public:
  // Each process should only have one DeviceTypeRegistry;
  static DeviceTypeRegistry* GetStaticDeviceTypeRegistry();

  ~DeviceTypeRegistry();

  void RegisterDeviceType(string_view type);
  const DeviceType& GetDeviceType(string_view type) const;

 private:
  DeviceType invalid_type_;

  // We use an array instead of map because we don't expected to have many
  // device types. And it is not on the performance critical path.
  SmallVector<DeviceType, 4> types_;
};

// A helper class for registering a new DeviceType.
struct DeviceTypeRegistration {
  explicit DeviceTypeRegistration(string_view name);
};

const DeviceType& GetStaticDeviceType(string_view type);

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_DEVICE_H_
