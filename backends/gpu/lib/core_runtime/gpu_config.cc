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

//===- gpu_config.cc ------------------------------------------------------===//
//
// This file implements various helpers to configure GPU OpHandler.
//
//===----------------------------------------------------------------------===//

#include "tfrt/gpu/core_runtime/gpu_config.h"

#include <unordered_map>

#include "llvm/ADT/Optional.h"
#include "tfrt/gpu/memory/bfc_gpu_allocator.h"
#include "tfrt/support/mutex.h"

namespace tfrt {
namespace gpu {

namespace {

class GpuResourcesMap {
 public:
  void SetResources(int gpu_ordinal, GpuResources resources) {
    mutex_lock lock(mu_);
    assert(map_.count(gpu_ordinal) == 0 &&
           "Only one set of resources per gpu ordinal is allowed");
    map_.emplace(std::make_pair(gpu_ordinal, std::move(resources)));
  }

  llvm::Optional<GpuResources> GetResources(int gpu_ordinal) {
    mutex_lock lock(mu_);
    auto it = map_.find(gpu_ordinal);
    if (it != map_.end()) {
      return it->second;
    }
    return llvm::None;
  }

  bool Empty() const {
    mutex_lock lock(mu_);
    return map_.empty();
  }

 private:
  mutable mutex mu_;
  std::unordered_map<int, GpuResources> map_;
};

}  // namespace

// TODO(zhangqiaorjc): For multitenant case, a global resource map may not be
// able to provide the resource isolation we want.
GpuResourcesMap* GetGpuResourcesMap() {
  static GpuResourcesMap* registry = new GpuResourcesMap;
  return registry;
}

void SetTfrtGpuResources(int gpu_ordinal, GpuResources resources) {
  GetGpuResourcesMap()->SetResources(gpu_ordinal, resources);
}

llvm::Optional<GpuResources> GetTfrtGpuResources(int gpu_ordinal) {
  return GetGpuResourcesMap()->GetResources(gpu_ordinal);
}

}  // namespace gpu
}  // namespace tfrt
