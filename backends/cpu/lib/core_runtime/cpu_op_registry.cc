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

// This file implements the CpuOpRegistry.

#include <vector>

#include "cpu_op_registry_impl.h"

namespace tfrt {

CpuOpRegistry::CpuOpRegistry() : impl_(std::make_unique<Impl>()) {}

CpuOpRegistry::~CpuOpRegistry() {}

CpuOpRegistry::CpuOpRegistry(CpuOpRegistry&& other) = default;

CpuOpRegistry& CpuOpRegistry::operator=(CpuOpRegistry&& other) = default;

// Add an op with the specified dispatch function.  This style of dispatch
// function does not require a shape function.
void CpuOpRegistry::AddOp(string_view op_name, CpuDispatchFn dispatch_fn,
                          CpuOpFlags flags, ArrayRef<string_view> attr_names) {
  impl_->AddOp(op_name, dispatch_fn, flags, attr_names);
}

void CpuOpRegistry::AddOp(string_view op_name, CpuDispatchFn dispatch_fn,
                          CpuOpFlags flags) {
  impl_->AddOp(op_name, dispatch_fn, flags, {});
}

// Set a metadata function for the specified op_name.  All metadata functions
// are required to be semantically equal, so multiple registrations for the
// same op are allowed (making static initialization easier).
void CpuOpRegistry::AddMetadataFn(string_view op_name,
                                  OpMetadataFn metadata_fn) {
  impl_->AddMetadataFn(op_name, metadata_fn);
}

static std::vector<CpuOpRegistration>* GetStaticCpuOpRegistrations() {
  static std::vector<CpuOpRegistration>* ret =
      new std::vector<CpuOpRegistration>;
  return ret;
}

void AddStaticCpuOpRegistration(CpuOpRegistration func) {
  GetStaticCpuOpRegistrations()->push_back(func);
}

void RegisterStaticCpuOps(CpuOpRegistry* op_reg) {
  for (auto func : *GetStaticCpuOpRegistrations()) {
    func(op_reg);
  }
}

}  // namespace tfrt
