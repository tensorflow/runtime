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
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/host_context.h"

#include "llvm/Support/Error.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/shared_context.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/string_util.h"

namespace tfrt {

void LocationHandler::VtableAnchor() {}

std::atomic<int> HostContext::num_shared_context_types_{0};
static std::atomic<int> next_host_context_index{0};
HostContext* HostContext::all_host_contexts_[HostContextPtr::kDummyIndex];

HostContext::HostContext(
    std::function<void(const DecodedDiagnostic&)> diag_handler,
    std::unique_ptr<HostAllocator> allocator,
    std::unique_ptr<ConcurrentWorkQueue> work_queue)
    : diag_handler_(std::move(diag_handler)),
      allocator_(std::move(allocator)),
      work_queue_(std::move(work_queue)),
      shared_context_mgr_(std::make_unique<SharedContextManager>(this)),
      instance_ptr_{next_host_context_index.fetch_add(1)} {
  assert(instance_index() < HostContextPtr::kDummyIndex &&
         "Created too many HostContext instances");
  all_host_contexts_[instance_index()] = this;
  ReadyChain::Get().Construct(this);
  // Add a CPU:0 device by default.
  static DeviceTypeRegistration cpu_type("cpu");
  // TODO(b/160264760): Pick a better device name than "CPU:0".
  host_device_ = device_mgr_.MaybeAddDevice(TakeRef(new CpuDevice("CPU:0")));
}

HostContext::~HostContext() {
  // Wait for the completion of all async tasks managed by this host context.
  Quiesce();
  // We need to free the ready chain AsyncValue first, as the destructor of the
  // AsyncValue calls the HostContext to free its memory.
  ReadyChain::Get().Destruct(this);
  all_host_contexts_[instance_index()] = nullptr;
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

// Add some work to the workqueue managed by this CPU device.
void HostContext::EnqueueWork(llvm::unique_function<void()> work) {
  work_queue_->AddTask(TaskFunction(std::move(work)));
}

// Add some work to the workqueue managed by this CPU device.
bool HostContext::EnqueueBlockingWork(llvm::unique_function<void()> work) {
  Optional<TaskFunction> task = work_queue_->AddBlockingTask(
      TaskFunction(std::move(work)), /*allow_queuing=*/true);
  return !task.hasValue();
}

// Runs blocking work on a work_queue managed by this CPU device.
bool HostContext::RunBlockingWork(llvm::unique_function<void()> work) {
  Optional<TaskFunction> task = work_queue_->AddBlockingTask(
      TaskFunction(std::move(work)), /*allow_queuing=*/false);
  return !task.hasValue();
}

int HostContext::GetNumWorkerThreads() const {
  return work_queue_->GetParallelismLevel();
}

bool HostContext::IsInWorkerThread() const {
  return work_queue_->IsInWorkerThread();
}

// Run the specified function when the specified set of AsyncValue's are all
// resolved.  This is a set-version of "AndThen".
void HostContext::RunWhenReady(ArrayRef<AsyncValue*> values,
                               llvm::unique_function<void()> callee) {
  // Perform a quick scan of the arguments.  If they are all available, or if
  // any is already an error, then we can run the callee synchronously.
  SmallVector<AsyncValue*, 4> unavailable_values;
  for (auto i : values) {
    if (!i->IsAvailable()) unavailable_values.push_back(i);
  }

  // If we can synchronously call 'callee', then do it and we're done.
  if (unavailable_values.empty()) return callee();

  // If there is exactly one unavailable value, then we can just AndThen it.
  if (unavailable_values.size() == 1) {
    unavailable_values[0]->AndThen(
        [callee = std::move(callee)]() mutable { callee(); });
    return;
  }

  struct CounterAndCallee {
    std::atomic<size_t> counter;
    llvm::unique_function<void()> callee;
  };

  // Otherwise, we have multiple unavailable values.  Put a counter on the heap
  // and have each unavailable value decrement and test it.
  auto* data =
      new CounterAndCallee{{unavailable_values.size()}, std::move(callee)};

  for (auto* val : unavailable_values) {
    val->AndThen([data]() {
      // Decrement the counter unless we're the last to be here.
      if (data->counter.fetch_sub(1) != 1) return;

      // If we are the last one, then run the callee and free the data.
      data->callee();
      delete data;
    });
  }
}

void HostContext::RunWhenReady(ArrayRef<RCReference<AsyncValue>> values,
                               llvm::unique_function<void()> callee) {
  auto mapped = llvm::map_range(
      values, [](const RCReference<AsyncValue>& ref) -> AsyncValue* {
        return ref.get();
      });
  SmallVector<AsyncValue*, 8> values_ptr(mapped.begin(), mapped.end());
  RunWhenReady(values_ptr, std::move(callee));
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
RCReference<Device> HostContext::GetHostDeviceRef() {
  return host_device_.CopyRef();
}

const Device& HostContext::GetHostDevice() { return *host_device_; }

}  // namespace tfrt
