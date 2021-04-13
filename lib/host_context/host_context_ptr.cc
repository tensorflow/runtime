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

// Compact pointer to HostContext
//
// This file defines HostContextPtr, a compact pointer representation for
// HostContext.

#include "tfrt/host_context/host_context_ptr.h"

#include "tfrt/host_context/host_context.h"

namespace tfrt {

HostContextPtr HostContextPool::AllocateForHostContext(HostContext* host) {
  mutex_lock lock(mutex_);

  for (int i = 0; i < all_host_contexts_.size(); ++i) {
    if (!all_host_contexts_[i]) {
      all_host_contexts_[i] = host;
      return HostContextPtr{i};
    }
  }

  llvm_unreachable("Created too many HostContext instances");
}

void HostContextPool::FreeHostContext(HostContext* host) {
  mutex_lock lock(mutex_);
  all_host_contexts_[host->instance_ptr().index()] = nullptr;
}

HostContext* HostContextPool::GetHostContextByIndex(int index) const {
  // Note that we do not need to lock the mutex here as
  // all_host_contexts_[index] is guranteed to filled when this function is
  // called.
  assert(index < all_host_contexts_.size());
  assert(all_host_contexts_[index]);
  return all_host_contexts_[index];
}

HostContextPtr::HostContextPtr(HostContext* host)
    : HostContextPtr{host->instance_ptr()} {}

HostContext* HostContextPtr::get() const {
  return HostContextPool::instance().GetHostContextByIndex(index_);
}

}  // namespace tfrt
