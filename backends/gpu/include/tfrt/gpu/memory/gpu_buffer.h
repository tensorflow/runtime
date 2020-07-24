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

//===- gpu_buffer.h - Types for holding GPU memory --------------*- C++ -*-===//
//
// This file declares classes that can be used to hold CUDA buffers
// and tensors.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_GPU_MEMORY_GPU_BUFFER_H_
#define TFRT_GPU_MEMORY_GPU_BUFFER_H_

#include <cstdint>

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {

class GpuAllocator;

// GpuBuffer represents a range of GPU memory.
// GpuBuffer is the type that is produced by the allocator. Besides the
// raw pointer, it also includes the size and the allocator pointer.
// We track the size because passing the size to GpuAllocator::Deallocate
// enables significantly more efficient algorithms. GpuBuffer includes
// GpuAllocator* so that the destructor can easily deallocate the memory.
//
// GpuBuffers own their memory.
// GpuBuffers are thread-safe.
// GpuBuffers are neither copyable nor movable.
class GpuBuffer : public ReferenceCounted<GpuBuffer> {
 public:
  // Creates a buffer with base `pointer`, holding `size` bytes, that will be
  // deallocated using `allocator` when destroyed.
  // `pointer` must have been obtained from the `allocator`.
  // Prefer using GpuAllocator::Allocate instead of creating buffers manually.
  GpuBuffer(gpu::stream::Pointer<void> pointer, size_t size,
            GpuAllocator* allocator);

  using Deallocator = llvm::unique_function<void(GpuBuffer* buffer)>;
  // Create a GpuBuffer by taking ownership of an externally allocated GPU
  // buffer. `deallocator` is called with `pointer` and `size` as arguments when
  // we destroy this buffer.
  GpuBuffer(gpu::stream::Pointer<void> pointer, size_t size,
            Deallocator deallocator);

  ~GpuBuffer();

  const gpu::stream::Pointer<void>& pointer() const { return pointer_; }

  // Returns the number of `bytes` held by this buffer.
  size_t size() const { return size_; }

  bool IsValid() const { return pointer_ != nullptr; }

 private:
  // Pointer value of 0 means that the buffer is not pointing to valid memory.
  gpu::stream::Pointer<void> pointer_;

  // Size of this buffer in bytes, i.e. number of bytes in the GPU memory that
  // this buffer represents.
  size_t size_ : sizeof(size_t) * 8 - 1;

  // Discriminator for the union.
  bool has_allocator_ : 1;
  // TODO(zhangqiaorjc): Use variant instead of union.
  union {
    // The allocator that allocated this buffer.
    GpuAllocator* allocator_;

    // The deallocator function that can be used to deallocate the externally
    // allocated buffer.
    Deallocator deallocator_;
  };
};

template <typename T>
T* GetRawPointer(const GpuBuffer& buffer) {
  return static_cast<T*>(buffer.pointer().raw());
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const GpuBuffer& buffer);

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_MEMORY_GPU_BUFFER_H_
