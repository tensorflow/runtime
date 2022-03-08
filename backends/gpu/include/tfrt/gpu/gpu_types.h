/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MLIR op types for cuda_ops library
//
// This file declares the types used in the `tfrt_gpu` dialect.

#ifndef TFRT_GPU_GPU_TYPES_H_
#define TFRT_GPU_GPU_TYPES_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>

#include "llvm/ADT/DenseMap.h"
#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/ccl_types.h"
#include "tfrt/gpu/wrapper/dnn_wrapper.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"
#include "tfrt/gpu/wrapper/fft_wrapper.h"
#include "tfrt/gpu/wrapper/solver_wrapper.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace gpu {

class GpuStream;  // Forward declaration.

// Types that do not need a wrapper class go here.
using GpuPointer = wrapper::Pointer<void>;
using GpuDnnTensorDesc = wrapper::OwningDnnTensorDescriptor;
using GpuCclId = wrapper::CclUniqueId;

class GpuContext {
  class HostMemoryPool {
   public:
    Expected<wrapper::HostMemory<void>> Allocate(
        wrapper::CurrentContext current, size_t size_bytes);
    void Deallocate(GpuPointer pointer, size_t size_bytes);

   private:
    std::multimap<size_t, wrapper::HostMemory<void>> pool_
        TFRT_GUARDED_BY(mutex_);
    mutex mutex_;
  };

  template <typename T>
  struct HostPoolMemoryDeleter {
    using pointer = wrapper::Pointer<T>;
    void operator()(GpuPointer pointer) const {
      pool->Deallocate(pointer, size_bytes);
    }
    HostMemoryPool* pool;
    size_t size_bytes;
  };

  class CallbackManager;

 public:
  template <typename T>
  using HostPoolMemory = std::unique_ptr<T, HostPoolMemoryDeleter<T>>;

  explicit GpuContext(wrapper::OwningContext context);
  ~GpuContext();

  GpuContext(GpuContext&&);
  GpuContext& operator=(GpuContext&&);

  const wrapper::OwningContext& operator->() const { return context_; }
  wrapper::Context get() const { return context_.get(); }
  wrapper::Context release();

  // Allocates write-combined page-locked array of `count` elements.
  // The user is responsible to destroy the result before the context.
  //
  // The memory is not deallocated until the context is. The indended use is to
  // repeatedly allocate small amounts for device access or async memcopy.
  template <typename T>
  Expected<HostPoolMemory<T>> AllocateHostPoolMemory(
      wrapper::CurrentContext current, size_t count) const {
    size_t size_bytes = count * sizeof(T);
    auto memory = host_memory_pool_->Allocate(current, size_bytes);
    if (!memory) return memory.takeError();
    auto pointer = static_cast<wrapper::Pointer<T>>(memory->release());
    return HostPoolMemory<T>(pointer, {host_memory_pool_.get(), size_bytes});
  }

  // Adds a 'callback' that will be invoked (from an unspecified thread) after
  // work that is currently enqueued on the 'stream' has completed.
  // Work enqueued afterwards does not wait for the callback to complete.
  // Callbacks are permitted to hold ref-counts to e.g. a GpuBuffer.
  static Error AddEventualCallback(wrapper::CurrentContext current,
                                   const GpuStream& stream,
                                   llvm::unique_function<void()> callback,
                                   HostContext* host);

  // Invokes callbacks that are ready (all work enqueued on the stream when the
  // callback was added is known to have completed). All synchronized callbacks
  // (the stream has been host-synchronized after the callback was added) are
  // guaranteed to be called. Callbacks that are not synchronized may or may
  // not be called. Returns whether all pending callbacks were invoked.
  Expected<bool> MaybeInvokeCallbacks() const;

 private:
  wrapper::OwningContext context_;
  std::unique_ptr<HostMemoryPool> host_memory_pool_;
  RCReference<CallbackManager> callback_manager_;
};

class GpuStream {
 public:
  explicit GpuStream(AsyncValueRef<GpuContext> context,
                     wrapper::OwningStream stream);
  ~GpuStream();

  GpuStream(GpuStream&&) = default;
  GpuStream& operator=(GpuStream&&) = default;

  const wrapper::OwningStream& operator->() const { return stream_; }
  wrapper::Stream get() const { return stream_.get(); }
  wrapper::Stream release();

  const AsyncValueRef<GpuContext>& context() const { return context_; }

 private:
  AsyncValueRef<GpuContext> context_;
  wrapper::OwningStream stream_;
};

class GpuEvent {
 public:
  explicit GpuEvent(AsyncValueRef<GpuContext> context,
                    wrapper::OwningEvent event);
  ~GpuEvent();

  GpuEvent(GpuEvent&&) = default;
  GpuEvent& operator=(GpuEvent&&) = default;

  const wrapper::OwningEvent& operator->() const { return event_; }
  wrapper::Event get() const { return event_.get(); }
  wrapper::Event release();

  const AsyncValueRef<GpuContext>& context() const { return context_; }

 private:
  AsyncValueRef<GpuContext> context_;
  wrapper::OwningEvent event_;
};

class GpuModule {
 public:
  explicit GpuModule(AsyncValueRef<GpuContext> context,
                     wrapper::OwningModule module);
  ~GpuModule();

  GpuModule(GpuModule&&) = default;
  GpuModule& operator=(GpuModule&&) = default;

  const wrapper::OwningModule& operator->() const { return module_; }
  wrapper::Module get() const { return module_.get(); }

 private:
  AsyncValueRef<GpuContext> context_;
  wrapper::OwningModule module_;
};

class GpuFunction {
 public:
  explicit GpuFunction(AsyncValueRef<GpuModule> module,
                       wrapper::Function function);
  ~GpuFunction();

  GpuFunction(GpuFunction&&) = default;
  GpuFunction& operator=(GpuFunction&&) = default;

  wrapper::Function get() const { return function_; }

 private:
  AsyncValueRef<GpuModule> module_;
  wrapper::Function function_;
};

// GpuAllocator base class.
class GpuAllocator {
  friend class GpuBuffer;

 public:
  // Buffers returned by subclasses must be aligned at least to `kAlignment`.
  // NOTE: Kernels should not assume that all buffers passed in will be aligned
  // at least to `kAlignment`. Compiler can allocate a large buffer and pass
  // less aligned parts of it to kernels.
  static const size_t kAlignment = 256;

  virtual ~GpuAllocator() = default;

 protected:
  static Expected<GpuPointer> Allocate(GpuAllocator* allocator, size_t size,
                                       wrapper::Stream stream) {
    assert(allocator != nullptr);
    return allocator->Allocate(size, stream);
  }

  static Error Deallocate(GpuAllocator* allocator, GpuPointer pointer,
                          wrapper::Stream stream) {
    assert(allocator != nullptr);
    return allocator->Deallocate(pointer, stream);
  }

 private:
  // Allocates memory of at least `size` bytes. If `stream` is the default, the
  // memory is accessible on any stream. Otherwise, accessing the memory on
  // other streams requires synchronization.
  virtual Expected<GpuPointer> Allocate(size_t size,
                                        wrapper::Stream stream) = 0;

  // Deallocates memory. If `stream` is not the default, any other stream
  // accessing the memory needs to be to be synchronized with `stream`.
  virtual Error Deallocate(GpuPointer pointer, wrapper::Stream stream) = 0;
};

class GpuDefaultAllocator : public GpuAllocator {
 public:
  explicit GpuDefaultAllocator(AsyncValueRef<GpuContext> context);

  ~GpuDefaultAllocator() override;

  GpuDefaultAllocator(GpuDefaultAllocator&&) = default;
  GpuDefaultAllocator& operator=(GpuDefaultAllocator&&) = default;

  const AsyncValueRef<GpuContext>& context() const { return context_; }

 private:
  Expected<GpuPointer> Allocate(size_t size, wrapper::Stream stream) override;
  Error Deallocate(GpuPointer pointer, wrapper::Stream stream) override;

 private:
  AsyncValueRef<GpuContext> context_;
};

// Allocator which can allocate exactly once. Holds an instance of T which could
// be an RAII type owning the underlaying memory.
template <typename T>
class GpuOneShotAllocator;

template <>
class GpuOneShotAllocator<void> : public GpuAllocator {
 public:
  explicit GpuOneShotAllocator(GpuPointer pointer);

  GpuOneShotAllocator(GpuOneShotAllocator&& other);
  GpuOneShotAllocator& operator=(GpuOneShotAllocator&& other);

 private:
  Expected<GpuPointer> Allocate(size_t size, wrapper::Stream stream) override;
  Error Deallocate(GpuPointer pointer, wrapper::Stream stream) override;

 private:
  GpuPointer pointer_;
};

template <typename T>
class GpuOneShotAllocator : public GpuOneShotAllocator<void> {
 public:
  explicit GpuOneShotAllocator(GpuPointer pointer, T value)
      : GpuOneShotAllocator<void>(pointer), value_(std::move(value)) {}

  GpuOneShotAllocator(GpuOneShotAllocator&&) = default;
  GpuOneShotAllocator& operator=(GpuOneShotAllocator&&) = default;

  const T& value() const { return value_; }
  T& value() { return value_; }

 private:
  T value_;
};

// GpuBuffer points to a range of GPU memory. It can be either owning the memory
// (produced by a GpuAllocator) or non-owning.
class GpuBuffer {
  // Creates a buffer with base `pointer`, holding `size` bytes, that will be
  // deallocated using `allocator` when destroyed.
  GpuBuffer(AsyncValueRef<GpuAllocator> allocator, GpuPointer pointer,
            size_t size);

 public:
  GpuBuffer();
  ~GpuBuffer();

  GpuBuffer(GpuBuffer&& buffer);
  GpuBuffer& operator=(GpuBuffer&& buffer);

  // Creates a buffer of at least `size` bytes. If `stream` is the default, the
  // buffer is accessible on any stream. Otherwise, accessing the buffer on
  // other streams requires synchronization.
  static Expected<GpuBuffer> Allocate(AsyncValueRef<GpuAllocator> allocator,
                                      size_t size, wrapper::Stream stream = {});

  // Deallocates the buffer. If `stream` is not the default, any other stream
  // accessing the buffer needs to be to be synchronized with `stream`.
  Error Deallocate(wrapper::Stream stream = {});

  explicit operator bool() const { return pointer_ != nullptr; }
  wrapper::Pointer<void> pointer() const { return pointer_; }
  size_t size() const { return size_; }

 private:
  AsyncValueRef<GpuAllocator> allocator_;
  wrapper::Pointer<void> pointer_;
  size_t size_ = 0;
};

class GpuBlasHandle {
 public:
  explicit GpuBlasHandle(AsyncValueRef<GpuContext> context,
                         wrapper::OwningBlasHandle handle);
  ~GpuBlasHandle();

  GpuBlasHandle(GpuBlasHandle&&) = default;
  GpuBlasHandle& operator=(GpuBlasHandle&&) = default;

  const wrapper::OwningBlasHandle& operator->() const { return handle_; }
  wrapper::BlasHandle get() const { return handle_.get(); }

  const AsyncValueRef<GpuContext>& context() const { return context_; }

 private:
  AsyncValueRef<GpuContext> context_;
  wrapper::OwningBlasHandle handle_;
};

// Handle for storing collective ops to be fused into a single group call. Owns
// the NCCL communicator.
class GpuCclHandle {
 public:
  using Callback = llvm::unique_function<llvm::Error(
      wrapper::CurrentContext current, wrapper::Stream stream,
      wrapper::CclComm comm)>;

  GpuCclHandle(
      AsyncValueRef<GpuContext> context, wrapper::OwningCclComm comm,
      llvm::unique_function<void(wrapper::CclComm)> custom_deleter = {});
  ~GpuCclHandle();

  GpuCclHandle(GpuCclHandle&&) = default;
  GpuCclHandle& operator=(GpuCclHandle&&) = default;

  void AddCallback(Callback callback);

  // Executes and clears all accumulated callbacks.
  llvm::Error ExecuteCallbacks(wrapper::CurrentContext current,
                               wrapper::Stream stream);

  const wrapper::OwningCclComm& operator->() const { return comm_; }
  wrapper::CclComm get() const { return comm_.get(); }

  const AsyncValueRef<GpuContext>& context() const { return context_; }

 private:
  AsyncValueRef<GpuContext> context_;
  wrapper::OwningCclComm comm_;
  llvm::unique_function<void(wrapper::CclComm)> custom_deleter_;
  std::vector<Callback> callbacks_;
};

class GpuDnnHandle {
 public:
  explicit GpuDnnHandle(AsyncValueRef<GpuContext> context,
                        wrapper::OwningDnnHandle handle);
  ~GpuDnnHandle();

  GpuDnnHandle(GpuDnnHandle&&) = default;
  GpuDnnHandle& operator=(GpuDnnHandle&&) = default;

  const wrapper::OwningDnnHandle& operator->() const { return handle_; }
  wrapper::DnnHandle get() const { return handle_.get(); }

  const AsyncValueRef<GpuContext>& context() const { return context_; }

 private:
  AsyncValueRef<GpuContext> context_;
  wrapper::OwningDnnHandle handle_;
};

class GpuSolverHandle {
 public:
  explicit GpuSolverHandle(AsyncValueRef<GpuContext> context,
                           wrapper::OwningSolverHandle handle);
  ~GpuSolverHandle();

  GpuSolverHandle(GpuSolverHandle&&) = default;
  GpuSolverHandle& operator=(GpuSolverHandle&&) = default;

  const wrapper::OwningSolverHandle& operator->() const { return handle_; }
  wrapper::SolverHandle get() const { return handle_.get(); }

  const AsyncValueRef<GpuContext>& context() const { return context_; }

 private:
  AsyncValueRef<GpuContext> context_;
  wrapper::OwningSolverHandle handle_;
};

class GpuFftHandle {
 public:
  explicit GpuFftHandle(AsyncValueRef<GpuContext> context,
                        wrapper::OwningFftHandle handle, wrapper::FftType type);
  ~GpuFftHandle();

  GpuFftHandle(GpuFftHandle&&) = default;
  GpuFftHandle& operator=(GpuFftHandle&&) = default;

  const wrapper::OwningFftHandle& operator->() const { return handle_; }
  wrapper::FftHandle get() const { return handle_.get(); }

  wrapper::FftType type() const { return type_; }
  const AsyncValueRef<GpuContext>& context() const { return context_; }

 private:
  AsyncValueRef<GpuContext> context_;
  wrapper::OwningFftHandle handle_;
  wrapper::FftType type_;
};

template <typename T>
T* GetRawPointer(const GpuBuffer& buffer) {
  return static_cast<T*>(buffer.pointer().raw());
}

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_GPU_TYPES_H_
