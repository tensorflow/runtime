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

// This file implements the KernelRegistry.

#include "tfrt/host_context/kernel_registry.h"

#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "tfrt/host_context/type_name.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"

namespace tfrt {

using llvm::StringMap;
using llvm::StringSet;

struct KernelRegistry::Impl {
  StringMap<KernelImplementation> implementations;
  StringSet<> type_names TFRT_GUARDED_BY(mu);
  mutex mu;
};

KernelRegistry::KernelRegistry() : impl_(std::make_unique<Impl>()) {}

KernelRegistry::~KernelRegistry() {}

void KernelRegistry::AddKernel(string_view kernel_name,
                               AsyncKernelImplementation fn) {
  bool added =
      impl_->implementations.try_emplace(kernel_name, KernelImplementation{fn})
          .second;
  (void)added;
  assert(added && "Re-registered existing kernel_name for async kernel");
}

void KernelRegistry::AddSyncKernel(string_view kernel_name,
                                   SyncKernelImplementation fn) {
  bool added =
      impl_->implementations.try_emplace(kernel_name, KernelImplementation{fn})
          .second;
  (void)added;
  assert(added && "Re-registered existing kernel_name for sync kernel");
}

KernelImplementation KernelRegistry::GetKernel(string_view kernel_name) const {
  auto it = impl_->implementations.find(kernel_name);
  return it == impl_->implementations.end() ? KernelImplementation()
                                            : it->second;
}

TypeName KernelRegistry::GetType(string_view type_name) const {
  mutex_lock lock(impl_->mu);
  auto it = impl_->type_names.insert(type_name).first;
  return TypeName(it->getKeyData());
}

static std::vector<KernelRegistration>* GetStaticKernelRegistrations() {
  static std::vector<KernelRegistration>* ret =
      new std::vector<KernelRegistration>;
  return ret;
}

void AddStaticKernelRegistration(KernelRegistration func) {
  GetStaticKernelRegistrations()->push_back(func);
}

void RegisterStaticKernels(KernelRegistry* kernel_reg) {
  for (auto func : *GetStaticKernelRegistrations()) {
    func(kernel_reg);
  }
}

}  // namespace tfrt
