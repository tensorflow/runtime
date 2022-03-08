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

// Implementation of the types used in the tfrt_gpu dialect.
#include "tfrt/gpu/gpu_types.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/ccl_wrapper.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/gpu/wrapper/dnn_wrapper.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"
#include "tfrt/gpu/wrapper/hip_wrapper.h"
#include "tfrt/gpu/wrapper/solver_wrapper.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {

Expected<wrapper::HostMemory<void>> GpuContext::HostMemoryPool::Allocate(
    wrapper::CurrentContext current, size_t size_bytes) {
  mutex_lock lock(mutex_);
  auto it = pool_.lower_bound(size_bytes);
  // Heuristic: use pool entry if it's at most 4 times the requested size.
  if (it != pool_.end() && it->first <= size_bytes * 4) {
    auto result = std::move(it->second);
    pool_.erase(it);
    return std::move(result);
  }
  // Write-combined results in faster transfers from/to GPU, but the pages are
  // not cached on CPU and should therefore be read/written at once.
  return wrapper::MemHostAllocWriteCombined(current, size_bytes);
}

void GpuContext::HostMemoryPool::Deallocate(GpuPointer pointer,
                                            size_t size_bytes) {
  mutex_lock lock(mutex_);
  pool_.emplace(size_bytes, pointer);
}

// Implements 'AddEventualCallback()' and 'MaybeInvokeCallbacks()'.
//
// Holds a list of event/callback pairs and queries (on-demand and regularly)
// each event and if it's ready, invokes the corresponding callback.
class GpuContext::CallbackManager : public ReferenceCounted<CallbackManager> {
  struct Callback {
    GpuEvent event;
    llvm::unique_function<void()> callback;
  };

 public:
  ~CallbackManager() {
    // Wait for any InvokeImpl() calls to complete.
    mutex_lock lock(invoke_mutex_);
    TFRT_LOG_IF(ERROR, !callbacks_.empty()) << "Not all callbacks were invoked";
  }

  // Destroys events in pool. This needs to be called before destroying the gpu
  // context because a pending invoke task extends the lifetime of 'this'.
  void ClearPool() {
    mutex_lock lock(pool_mutex_);
    event_pool_.clear();
  }

  // Records an event into 'stream' and invokes 'callback' when it's ready.
  Error Add(wrapper::CurrentContext current, const GpuStream& stream,
            llvm::unique_function<void()> callback, HostContext* host) {
    mutex_lock lock(invoke_mutex_);
    Expected<GpuEvent> event = CreateEvent(current, stream.context());
    if (!event) return event.takeError();
    if (auto error = wrapper::EventRecord(event->get(), stream.get()))
      return error;
    callbacks_.push_back(Callback{std::move(*event), std::move(callback)});
    if (pending_invoke_) return Error::success();
    return InvokeImpl(host);
  }

  // Invoke pending callbacks if their event is ready. Returns whether all
  // pending callbacks have been invoked.
  Expected<bool> Invoke() {
    mutex_lock lock(invoke_mutex_);
    if (auto error = InvokeImpl(nullptr)) return std::move(error);
    return callbacks_.empty();
  }

 private:
  // Sleeps for a 100ms before calling InvokeImpl(). Called from blocking tasks.
  Error InvokeLater(HostContext* host) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    mutex_lock lock(invoke_mutex_);
    pending_invoke_ = false;
    if (callbacks_.empty()) return Error::success();
    return InvokeImpl(host);
  }

  // Queries each event and if it's ready, invokes the corresponding callback.
  // If any event is not ready yet and host != nullptr, schedule a task to
  // check back later.
  Error InvokeImpl(HostContext* host) TFRT_REQUIRES(invoke_mutex_) {
    // Prevent 'this' from being destroyed along with the GpuContext if all
    // events are erased. This ref-count is moved to the blocking task below.
    auto manager = FormRef(this);
    // For each event that is ready, invoke the callback and remove the element.
    Error error = Error::success();
    pool_mutex_.lock();
    auto it = llvm::remove_if(callbacks_, [&](Callback& callback) TFRT_REQUIRES(
                                              invoke_mutex_, pool_mutex_) {
      auto result = wrapper::EventQuery(callback.event.get());
      if (!result) {
        error = llvm::joinErrors(std::move(error), result.takeError());
        return false;
      }
      if (!*result) return false;
      callback.callback();
      event_pool_.push_back(wrapper::OwningEvent(callback.event.release()));
      return true;
    });
    // Erase the removed elements after unlocking the pool_mutex_, because
    // destroying the callbacks might call ClearPool() from ~GpuContext().
    pool_mutex_.unlock();
    callbacks_.erase(it, callbacks_.end());
    if (error) return error;
    if (callbacks_.empty() || pending_invoke_ || !host) return Error::success();
    // Enqueue a blocking task that calls this function again after some
    // delay.
    pending_invoke_ = true;
    bool enqueued =
        EnqueueBlockingWork(host, [host, manager = std::move(manager)] {
          if (auto error = manager->InvokeLater(host))
            host->EmitError(DecodedDiagnostic(std::move(error)));
        });
    if (!enqueued) return MakeStringError("Failed to enqueue blocking work.");
    return Error::success();
  }

  Expected<GpuEvent> CreateEvent(wrapper::CurrentContext current,
                                 AsyncValueRef<GpuContext> gpu_context) {
    mutex_lock lock(pool_mutex_);
    // Amortize the cost of event instancing with a pool.
    if (event_pool_.empty()) {
      auto event = wrapper::EventCreateNoTiming(current);
      if (!event) return event.takeError();
      event_pool_.emplace_back(std::move(*event));
    }
    GpuEvent event(std::move(gpu_context), std::move(event_pool_.back()));
    event_pool_.pop_back();
    return std::move(event);
  }

  std::vector<Callback> callbacks_ TFRT_GUARDED_BY(invoke_mutex_);
  bool pending_invoke_ TFRT_GUARDED_BY(invoke_mutex_) = false;
  mutex invoke_mutex_;

  std::vector<wrapper::OwningEvent> event_pool_ TFRT_GUARDED_BY(pool_mutex_);
  mutex pool_mutex_;
};

GpuContext::GpuContext(wrapper::OwningContext context)
    : context_(std::move(context)),
      host_memory_pool_(new HostMemoryPool()),
      callback_manager_(TakeRef(new CallbackManager)) {}

GpuContext::~GpuContext() {
  if (callback_manager_) callback_manager_->ClearPool();
}

GpuContext::GpuContext(GpuContext&&) = default;
GpuContext& GpuContext::operator=(GpuContext&&) = default;

wrapper::Context GpuContext::release() {
  callback_manager_->ClearPool();
  callback_manager_.reset();
  host_memory_pool_.reset();
  return context_.release();
}

Error GpuContext::AddEventualCallback(wrapper::CurrentContext current,
                                      const GpuStream& stream,
                                      llvm::unique_function<void()> callback,
                                      HostContext* host) {
  return stream.context()->callback_manager_->Add(current, stream,
                                                  std::move(callback), host);
}

Expected<bool> GpuContext::MaybeInvokeCallbacks() const {
  return callback_manager_->Invoke();
}

GpuStream::GpuStream(AsyncValueRef<GpuContext> context,
                     wrapper::OwningStream stream)
    : context_(std::move(context)), stream_(std::move(stream)) {}

GpuStream::~GpuStream() = default;

wrapper::Stream GpuStream::release() {
  context_.reset();
  return stream_.release();
}

GpuEvent::GpuEvent(AsyncValueRef<GpuContext> context,
                   wrapper::OwningEvent event)
    : context_(std::move(context)), event_(std::move(event)) {}

GpuEvent::~GpuEvent() = default;

wrapper::Event GpuEvent::release() {
  context_.reset();
  return event_.release();
}

GpuModule::GpuModule(AsyncValueRef<GpuContext> context,
                     wrapper::OwningModule module)
    : context_(std::move(context)), module_(std::move(module)) {}

GpuModule::~GpuModule() = default;

GpuFunction::GpuFunction(AsyncValueRef<GpuModule> module,
                         wrapper::Function function)
    : module_(std::move(module)), function_(function) {}

GpuFunction::~GpuFunction() = default;

GpuDefaultAllocator::GpuDefaultAllocator(AsyncValueRef<GpuContext> context)
    : context_(std::move(context)) {}

GpuDefaultAllocator::~GpuDefaultAllocator() = default;

Expected<GpuPointer> GpuDefaultAllocator::Allocate(size_t size,
                                                   wrapper::Stream) {
  auto current = wrapper::CtxSetCurrent(context_->get());
  if (!current) return current.takeError();
  auto memory = wrapper::MemAlloc(*current, size);
  if (!memory) return memory.takeError();
  return memory->release();
}

Error GpuDefaultAllocator::Deallocate(GpuPointer pointer, wrapper::Stream) {
  return wrapper::MemFree(pointer);
}

GpuOneShotAllocator<void>::GpuOneShotAllocator(GpuPointer pointer)
    : pointer_(pointer) {}

GpuOneShotAllocator<void>::GpuOneShotAllocator(GpuOneShotAllocator&& other)
    : pointer_(other.pointer_) {
  other.pointer_ = nullptr;
}

GpuOneShotAllocator<void>& GpuOneShotAllocator<void>::operator=(
    GpuOneShotAllocator&& other) {
  pointer_ = other.pointer_;
  other.pointer_ = nullptr;
  return *this;
}

Expected<GpuPointer> GpuOneShotAllocator<void>::Allocate(
    size_t size, wrapper::Stream stream) {
  if (size == 0) {
    return GpuPointer(nullptr, pointer_.platform());
  }
  if (!pointer_) {
    return MakeStringError(
        "Trying to allocate from GpuOneShotAllocator with null pointer.");
  }
  GpuPointer result = pointer_;
  pointer_ = GpuPointer(nullptr, pointer_.platform());
  return result;
}

Error GpuOneShotAllocator<void>::Deallocate(GpuPointer pointer,
                                            wrapper::Stream stream) {
  if (pointer != nullptr) {
    pointer_ = pointer;
  }
  return Error::success();
}

GpuBuffer::GpuBuffer(AsyncValueRef<GpuAllocator> allocator, GpuPointer pointer,
                     size_t size)
    : allocator_(std::move(allocator)), pointer_(pointer), size_(size) {}

GpuBuffer::GpuBuffer() = default;

GpuBuffer::~GpuBuffer() {
  if (auto error = Deallocate()) TFRT_LOG(ERROR) << error;
}

GpuBuffer::GpuBuffer(GpuBuffer&& buffer)
    : allocator_(std::move(buffer.allocator_)),
      pointer_(buffer.pointer_),
      size_(buffer.size_) {
  buffer.pointer_ = nullptr;
  buffer.size_ = 0;
}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& buffer) {
  if (auto error = Deallocate()) TFRT_LOG(ERROR) << error;
  allocator_ = std::move(buffer.allocator_);
  std::swap(pointer_, buffer.pointer_);
  std::swap(size_, buffer.size_);
  return *this;
}

Expected<GpuBuffer> GpuBuffer::Allocate(AsyncValueRef<GpuAllocator> allocator,
                                        size_t size, wrapper::Stream stream) {
  auto pointer = allocator->Allocate(size, stream);
  if (!pointer) return pointer.takeError();
  return GpuBuffer(std::move(allocator), *pointer, size);
}

Error GpuBuffer::Deallocate(wrapper::Stream stream) {
  size_ = 0;
  if (!pointer_) return Error::success();  // Skip virtual function call.
  auto pointer = pointer_;
  pointer_ = GpuPointer();
  return allocator_->Deallocate(pointer, stream);
}

GpuBlasHandle::GpuBlasHandle(AsyncValueRef<GpuContext> context,
                             wrapper::OwningBlasHandle handle)
    : context_(std::move(context)), handle_(std::move(handle)) {}

GpuBlasHandle::~GpuBlasHandle() = default;

GpuCclHandle::GpuCclHandle(
    AsyncValueRef<GpuContext> context, wrapper::OwningCclComm comm,
    llvm::unique_function<void(wrapper::CclComm)> custom_deleter)
    : context_(std::move(context)),
      comm_(std::move(comm)),
      custom_deleter_(std::move(custom_deleter)) {}

GpuCclHandle::~GpuCclHandle() {
  if (custom_deleter_) {
    // Using the custom deleter instead.
    custom_deleter_(comm_.release());
  }
}

void GpuCclHandle::AddCallback(Callback callback) {
  callbacks_.push_back(std::move(callback));
}

llvm::Error GpuCclHandle::ExecuteCallbacks(wrapper::CurrentContext current,
                                           wrapper::Stream stream) {
  if (auto error = wrapper::CclGroupStart(current.platform())) return error;
  for (auto& callback : callbacks_)
    if (auto error = callback(current, stream, comm_.get())) return error;
  if (auto error = wrapper::CclGroupEnd(current.platform())) return error;
  callbacks_.clear();
  return llvm::Error::success();
}

GpuDnnHandle::GpuDnnHandle(AsyncValueRef<GpuContext> context,
                           wrapper::OwningDnnHandle handle)
    : context_(std::move(context)), handle_(std::move(handle)) {}

GpuDnnHandle::~GpuDnnHandle() = default;

GpuSolverHandle::GpuSolverHandle(AsyncValueRef<GpuContext> context,
                                 wrapper::OwningSolverHandle handle)
    : context_(std::move(context)), handle_(std::move(handle)) {}

GpuSolverHandle::~GpuSolverHandle() = default;

GpuFftHandle::GpuFftHandle(AsyncValueRef<GpuContext> context,
                           wrapper::OwningFftHandle handle,
                           wrapper::FftType type)
    : context_(std::move(context)), handle_(std::move(handle)), type_(type) {}

GpuFftHandle::~GpuFftHandle() = default;

}  // namespace gpu
}  // namespace tfrt
