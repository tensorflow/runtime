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
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/dnn_wrapper.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"
#include "tfrt/gpu/wrapper/solver_wrapper.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {

// Types that do not need a wrapper class go here.
using GpuFunction = wrapper::Function;
using GpuPointer = wrapper::Pointer<void>;

class GpuContext {
 public:
  explicit GpuContext(wrapper::OwningContext context);
  ~GpuContext();

  GpuContext(GpuContext&&) = default;
  GpuContext& operator=(GpuContext&&) = default;

  const wrapper::OwningContext& operator->() const { return context_; }
  wrapper::Context get() const { return context_.get(); }

  Expected<GpuFunction> GetFunction(uint64_t key, string_view data,
                                    string_view name);

 private:
  wrapper::OwningContext context_;
  llvm::DenseMap<uint64_t, std::pair<wrapper::OwningModule, GpuFunction>>
      functions_;
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

  wrapper::Context context() const { return context_->get(); }

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

 private:
  AsyncValueRef<GpuContext> context_;
  wrapper::OwningEvent event_;
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

  explicit GpuAllocator(AsyncValueRef<GpuContext> context);
  virtual ~GpuAllocator();

  GpuAllocator(GpuAllocator&&) = default;
  GpuAllocator& operator=(GpuAllocator&&) = default;

  wrapper::Context context() const { return context_->get(); }

 private:
  // Allocates memory of at least `size` bytes. If `stream` is the default, the
  // memory is accessible on any stream. Otherwise, accessing the memory on
  // other streams requires synchronization.
  virtual Expected<GpuPointer> Allocate(size_t size, wrapper::Stream stream);

  // Deallocates memory. If `stream` is not the default, any other stream
  // accessing the memory needs to be to be synchronized with `stream`.
  virtual Error Deallocate(GpuPointer pointer, wrapper::Stream stream);

 private:
  AsyncValueRef<GpuContext> context_;
};

// GpuBuffer owns a range of GPU memory produced by a GpuAllocator.
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
  const GpuAllocator& allocator() const { return *allocator_; }

 private:
  AsyncValueRef<GpuAllocator> allocator_;
  wrapper::Pointer<void> pointer_;
  size_t size_ = 0;
};

class GpuBlasHandle {
 public:
  explicit GpuBlasHandle(AsyncValueRef<GpuStream> stream,
                         wrapper::OwningBlasHandle handle);
  ~GpuBlasHandle();

  GpuBlasHandle(GpuBlasHandle&&) = default;
  GpuBlasHandle& operator=(GpuBlasHandle&&) = default;

  const wrapper::OwningBlasHandle& operator->() const { return handle_; }
  wrapper::BlasHandle get() const { return handle_.get(); }

  wrapper::Context context() const { return stream_->context(); }

 private:
  AsyncValueRef<GpuStream> stream_;
  wrapper::OwningBlasHandle handle_;
};

class GpuDnnHandle {
 public:
  explicit GpuDnnHandle(AsyncValueRef<GpuStream> stream,
                        wrapper::OwningDnnHandle handle);
  ~GpuDnnHandle();

  GpuDnnHandle(GpuDnnHandle&&) = default;
  GpuDnnHandle& operator=(GpuDnnHandle&&) = default;

  const wrapper::OwningDnnHandle& operator->() const { return handle_; }
  wrapper::DnnHandle get() const { return handle_.get(); }

  wrapper::Context context() const { return stream_->context(); }

 private:
  AsyncValueRef<GpuStream> stream_;
  wrapper::OwningDnnHandle handle_;
};

class GpuSolverHandle {
 public:
  explicit GpuSolverHandle(AsyncValueRef<GpuStream> stream,
                           wrapper::OwningSolverHandle handle);
  ~GpuSolverHandle();

  GpuSolverHandle(GpuSolverHandle&&) = default;
  GpuSolverHandle& operator=(GpuSolverHandle&&) = default;

  const wrapper::OwningSolverHandle& operator->() const { return handle_; }
  wrapper::SolverHandle get() const { return handle_.get(); }

  wrapper::Context context() const { return stream_->context(); }

 private:
  AsyncValueRef<GpuStream> stream_;
  wrapper::OwningSolverHandle handle_;
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_GPU_TYPES_H_
