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

//===- host_context.cc - CPU thread and memory abstraction ----------------===//
//
// This file implements the generic interface for thread pool abstractions.

#include "tfrt/host_context/host_context.h"

#include "llvm/Support/Error.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/shared_context.h"
#include "tfrt/support/string_util.h"

namespace tfrt {

std::atomic<int> HostContext::num_shared_context_types_{0};
const char* const HostContext::kDefaultHostDeviceName = "CPU:0";

HostContext::HostContext(
    std::function<void(const DecodedDiagnostic&)> diag_handler,
    std::unique_ptr<HostAllocator> allocator,
    std::unique_ptr<ConcurrentWorkQueue> work_queue,
    string_view host_device_name)
    : diag_handler_(std::move(diag_handler)),
      allocator_(std::move(allocator)),
      work_queue_(std::move(work_queue)),
      shared_context_mgr_(std::make_unique<SharedContextManager>(this)),
      instance_ptr_{HostContextPool::instance().AllocateForHostContext(this)} {
  ReadyChain::Get().Construct(this);
  host_device_ =
      device_mgr_.MaybeAddDevice(MakeRef<CpuDevice>(host_device_name));
}

HostContext::HostContext(
    std::function<void(const DecodedDiagnostic&)> diag_handler,
    std::unique_ptr<HostAllocator> allocator,
    std::unique_ptr<ConcurrentWorkQueue> work_queue)
    : HostContext(std::move(diag_handler), std::move(allocator),
                  std::move(work_queue), kDefaultHostDeviceName) {}

HostContext::~HostContext() {
  // Wait for the completion of all async tasks managed by this host context.
  Quiesce();
  // We need to free the ready chain AsyncValue first, as the destructor of the
  // AsyncValue calls the HostContext to free its memory.
  ReadyChain::Get().Destruct(this);
  HostContextPool::instance().FreeHostContext(this);
}

void Function::VtableAnchor() {}

//===----------------------------------------------------------------------===//
// Error Reporting
//===----------------------------------------------------------------------===//

// Emit an error for a specified decoded diagnostic, which gets funneled
// through a location handler.
void HostContext::EmitError(const DecodedDiagnostic& diagnostic) {
  // Emit the message to the global handler, guaranteeing that it will be seen
  // by the handler registered with the HostContext.
  diag_handler_(diagnostic);
}

//===----------------------------------------------------------------------===//
// Memory Management
//===----------------------------------------------------------------------===//

// Allocate the specified number of bytes at the specified alignment.
void* HostContext::AllocateBytes(size_t size, size_t alignment) {
  return allocator_->AllocateBytes(size, alignment);
}

// Deallocate the specified pointer, that had the specified size.
void HostContext::DeallocateBytes(void* ptr, size_t size) {
  allocator_->DeallocateBytes(ptr, size);
}

//===----------------------------------------------------------------------===//
// Concurrency
//===----------------------------------------------------------------------===//

void HostContext::Quiesce() { work_queue_->Quiesce(); }

void HostContext::Await(ArrayRef<RCReference<AsyncValue>> values) {
  work_queue_->Await(values);
}

int HostContext::GetNumWorkerThreads() const {
  return work_queue_->GetParallelismLevel();
}

bool HostContext::IsInWorkerThread() const {
  return work_queue_->IsInWorkerThread();
}

//===----------------------------------------------------------------------===//
// SharedContext management
//===----------------------------------------------------------------------===//

class HostContext::SharedContextManager {
 public:
  explicit SharedContextManager(HostContext* host) : host_{host} {}
  // Returns the shared context instance with the given shared_context_id.
  // Create one if the requested shared context instance does not exist yet.
  SharedContext& GetOrCreateSharedContext(int shared_context_id,
                                          SharedContextFactory factory) {
    assert(shared_context_id < shared_context_instances_.size() &&
           "The requested SharedContext ID exceeds the maximum allowed");

    auto& item = shared_context_instances_[shared_context_id];

    std::call_once(item.second, [&] {
      assert(!item.first);
      item.first = factory(host_);
    });

    return *item.first;
  }

 private:
  HostContext* const host_;
  // We allow up to 256 ShareContext instances.
  std::array<std::pair<std::unique_ptr<SharedContext>, std::once_flag>, 256>
      shared_context_instances_{};
};

SharedContext& HostContext::GetOrCreateSharedContext(
    int shared_context_id, SharedContextFactory factory) {
  return shared_context_mgr_->GetOrCreateSharedContext(shared_context_id,
                                                       factory);
}

//===----------------------------------------------------------------------===//
// Device Manager
//===----------------------------------------------------------------------===//
RCReference<Device> HostContext::GetHostDeviceRef() { return host_device_; }

const Device& HostContext::GetHostDevice() { return *host_device_; }

void HostContext::ResetHostDevice(CpuDevice* device) {
  host_device_.reset(device);
}

}  // namespace tfrt
