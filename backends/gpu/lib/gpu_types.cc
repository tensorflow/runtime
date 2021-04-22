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
#include "tfrt/support/error_util.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {
GpuContext::GpuContext(wrapper::OwningContext context)
    : context_(std::move(context)) {}

GpuContext::~GpuContext() = default;

// Wrapper for module loading that prints logs when in debug mode.
static llvm::Expected<wrapper::OwningModule> LoadModule(
    wrapper::CurrentContext current, string_view data) {
#ifdef NDEBUG
  return wrapper::ModuleLoadData(current, data.data());
#else
  std::string info_log;
  std::string error_log;

  wrapper::ModuleLoadOptions options{&info_log, &error_log, 1};
  auto maybe_module = wrapper::ModuleLoadDataEx(current, data.data(), options);
  if (!info_log.empty()) {
    TFRT_LOG_INFO << "GPU JIT info log: " << info_log;
  }
  if (!maybe_module) {
    TFRT_LOG_ERROR << "GPU JIT error log: " << error_log;
  }
  return maybe_module;
#endif
}

Expected<GpuFunction> GpuContext::GetFunction(uint64_t key, string_view data,
                                              string_view name) {
  auto it = functions_.find(key);
  if (it != functions_.end()) {
    // Returned cached function.
    return std::get<GpuFunction>(it->second);
  }

  if (data.empty() || data.back() != 0)
    return MakeStringError("data attribute must be null-terminated");
  if (name.empty() || name.back() != 0)
    return MakeStringError("name attribute must be null-terminated");

  auto current = wrapper::CtxSetCurrent(context_.get());
  if (!current) return current.takeError();

  auto module = LoadModule(*current, data);
  if (!module) return module.takeError();

  auto function = wrapper::ModuleGetFunction(module->get(), name.data());
  if (!function) return function.takeError();

  auto pair = functions_.try_emplace(
      key, std::make_pair(std::move(*module), *function));
  assert(pair.second && "failed to insert into map");

  return std::get<GpuFunction>(pair.first->second);
}

GpuStream::GpuStream(AsyncValueRef<GpuContext> context,
                     wrapper::OwningStream stream)
    : context_(std::move(context)), stream_(std::move(stream)) {}

GpuStream::~GpuStream() = default;

GpuEvent::GpuEvent(AsyncValueRef<GpuContext> context,
                   wrapper::OwningEvent event)
    : context_(std::move(context)), event_(std::move(event)) {}

GpuEvent::~GpuEvent() = default;

GpuAllocator::GpuAllocator(AsyncValueRef<GpuContext> context)
    : context_(std::move(context)) {}

GpuAllocator::~GpuAllocator() = default;

Expected<GpuPointer> GpuAllocator::Allocate(size_t size, wrapper::Stream) {
  auto current = wrapper::CtxSetCurrent(context_->get());
  if (!current) return current.takeError();
  auto memory = wrapper::MemAlloc(*current, size);
  if (!memory) return memory.takeError();
  return memory->release();
}

Error GpuAllocator::Deallocate(GpuPointer pointer, wrapper::Stream) {
  return wrapper::MemFree(pointer);
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
