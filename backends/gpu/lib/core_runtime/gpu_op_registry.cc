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

// This file implements the GpuOpRegistry.

#include <vector>

#include "gpu_op_registry_impl.h"

namespace tfrt {
namespace gpu {

GpuOpRegistry::GpuOpRegistry() : impl_(std::make_unique<Impl>()) {}

GpuOpRegistry::~GpuOpRegistry() {}

GpuOpRegistry::GpuOpRegistry(GpuOpRegistry&& other) = default;

GpuOpRegistry& GpuOpRegistry::operator=(GpuOpRegistry&& other) = default;

// Add an op with the specified dispatch function.  This style of dispatch
// function does not require a shape function.
void GpuOpRegistry::AddOp(string_view op_name, GpuDispatchFn dispatch_fn) {
  impl_->AddOp(op_name, dispatch_fn, GpuOpFlags{}, {});
}

void GpuOpRegistry::AddOp(string_view op_name, GpuDispatchFn dispatch_fn,
                          ArrayRef<string_view> attr_names) {
  impl_->AddOp(op_name, dispatch_fn, GpuOpFlags{}, attr_names);
}

// Set a metadata function for the specified op_name.  All metadata functions
// are required to be semantically equal, so multiple registrations for the
// same op are allowed (making static initialization easier).
void GpuOpRegistry::AddMetadataFn(string_view op_name,
                                  OpMetadataFn metadata_fn) {
  impl_->AddMetadataFn(op_name, metadata_fn);
}

static std::vector<GpuOpRegistration>* GetStaticGpuOpRegistrations() {
  static std::vector<GpuOpRegistration>* ret =
      new std::vector<GpuOpRegistration>;
  return ret;
}

void AddStaticGpuOpRegistration(GpuOpRegistration func) {
  GetStaticGpuOpRegistrations()->push_back(func);
}

void RegisterStaticGpuOps(GpuOpRegistry* op_reg) {
  for (auto func : *GetStaticGpuOpRegistrations()) {
    func(op_reg);
  }
}

}  // namespace gpu
}  // namespace tfrt
