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

#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/dnn_wrapper.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {

Expected<wrapper::HostMemory<void>> GpuContext::HostMemoryPool::Allocate(
    wrapper::CurrentContext current, size_t size_bytes) {
  mutex_lock lock(mutex_);
  auto it = pool_.lower_bound(size_bytes);
  if (it != pool_.end()) {
    auto result = std::move(it->second);
    pool_.erase(it);
    return std::move(result);
  }
  // Write-combined results in faster transfers from/to GPU, but the pages are
  // not cached on CPU and should therefore be read/written at once.
  wrapper::MemHostAllocFlags flags = wrapper::MemHostAllocFlags::DEVICEMAP |
                                     wrapper::MemHostAllocFlags::WRITECOMBINED;
  return wrapper::MemHostAlloc(current, size_bytes, flags);
}

void GpuContext::HostMemoryPool::Deallocate(GpuPointer pointer,
                                            size_t size_bytes) {
  mutex_lock lock(mutex_);
  pool_.emplace(size_bytes, pointer);
}

GpuContext::GpuContext(wrapper::OwningContext context)
    : context_(std::move(context)), host_memory_pool_(new HostMemoryPool) {}

GpuContext::~GpuContext() = default;

wrapper::Context GpuContext::release() {
  auto result = context_.release();
  host_memory_pool_.reset();
  return result;
}

GpuStream::GpuStream(AsyncValueRef<GpuContext> context,
                     wrapper::OwningStream stream)
    : context_(std::move(context)), stream_(std::move(stream)) {}

GpuStream::~GpuStream() = default;

wrapper::Stream GpuStream::release() {
  context_.reset();
  return stream_.release();
}

BorrowedGpuStream::BorrowedGpuStream(wrapper::Context context,
                                     wrapper::Stream stream)
    : context_(MakeAvailableAsyncValueRef<GpuContext>(
          wrapper::OwningContext(context))),
      stream_(MakeAvailableAsyncValueRef<GpuStream>(
          context_.CopyRef(), wrapper::OwningStream(stream))) {}

BorrowedGpuStream::~BorrowedGpuStream() {
  stream_->release();
  context_->release();
}

GpuEvent::GpuEvent(AsyncValueRef<GpuContext> context,
                   wrapper::OwningEvent event)
    : context_(std::move(context)), event_(std::move(event)) {}

GpuEvent::~GpuEvent() = default;

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

GpuBlasHandle::GpuBlasHandle(AsyncValueRef<GpuStream> stream,
                             wrapper::OwningBlasHandle handle)
    : stream_(std::move(stream)), handle_(std::move(handle)) {}

GpuBlasHandle::~GpuBlasHandle() = default;

GpuCclHandle::GpuCclHandle(AsyncValueRef<GpuContext> context,
                           wrapper::OwningCclComm comm)
    : context_(std::move(context)), comm_(std::move(comm)) {}

GpuCclHandle::~GpuCclHandle() = default;

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

wrapper::CclComm GpuCclHandle::release() {
  context_.reset();
  return comm_.release();
}

GpuDnnHandle::GpuDnnHandle(AsyncValueRef<GpuStream> stream,
                           wrapper::OwningDnnHandle handle)
    : stream_(std::move(stream)), handle_(std::move(handle)) {}

GpuDnnHandle::~GpuDnnHandle() = default;

GpuSolverHandle::GpuSolverHandle(AsyncValueRef<GpuStream> stream,
                                 wrapper::OwningSolverHandle handle)
    : stream_(std::move(stream)), handle_(std::move(handle)) {}

GpuSolverHandle::~GpuSolverHandle() = default;

}  // namespace gpu
}  // namespace tfrt
