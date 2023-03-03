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

// This file implements various helpers to configure GPU OpHandler.

#include "tfrt/gpu/device/gpu_config.h"

#include <functional>
#include <optional>
#include <unordered_map>

#include "tfrt/gpu/memory/bfc_gpu_allocator.h"
#include "tfrt/support/mutex.h"

namespace tfrt {
namespace gpu {

namespace {

class GpuResourcesMap {
 public:
  void SetResources(wrapper::Device device, GpuResources resources) {
    mutex_lock lock(mu_);
    auto it = map_.emplace(std::make_pair(device, std::move(resources)));
    static_cast<void>(it);
    assert(it.second && "Only one set of resources per gpu ordinal is allowed");
  }

  std::optional<GpuResources> GetResources(wrapper::Device device) {
    mutex_lock lock(mu_);
    auto it = map_.find(device);
    if (it != map_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

 private:
  struct DeviceHash {
    // Algorithm of boost::hash_combine
    static std::size_t HashCombine(std::size_t x, std::size_t y) {
      return x + 0x9e3779b9 + (y << 6) + (y >> 2);
    }
    std::size_t operator()(const wrapper::Device& device) const {
      auto platform = device.platform();
      auto id = device.id(platform);
      std::hash<int> hash;
      return HashCombine(hash(id), hash(static_cast<int>(platform)));
    }
  };

  mutable mutex mu_;
  std::unordered_map<wrapper::Device, GpuResources, DeviceHash> map_;
};

}  // namespace

// TODO(zhangqiaorjc): For multitenant case, a global resource map may not be
// able to provide the resource isolation we want.
GpuResourcesMap* GetGpuResourcesMap() {
  static GpuResourcesMap* registry = new GpuResourcesMap;
  return registry;
}

void SetTfrtGpuResources(wrapper::Device device, GpuResources resources) {
  GetGpuResourcesMap()->SetResources(device, resources);
}

std::optional<GpuResources> GetTfrtGpuResources(wrapper::Device device) {
  return GetGpuResourcesMap()->GetResources(device);
}

}  // namespace gpu
}  // namespace tfrt
