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

// GPU memory allocator
//
// This file defines the abstract interface for a GPU memory allocator
#ifndef TFRT_GPU_MEMORY_GPU_ALLOCATOR_H_
#define TFRT_GPU_MEMORY_GPU_ALLOCATOR_H_

#include "llvm/Support/Error.h"
#include "tfrt/gpu/memory/gpu_buffer.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {

// GpuAllocator implementations are expected to be thread-safe.
class GpuCrtAllocator {
 public:
  // Buffers returned by subclasses must be aligned at least to `kAlignment`.
  // NOTE: Kernels should not assume that all buffers passed in will be aligned
  // at least to `kAlignment`. Compiler can allocate a large buffer and pass
  // less aligned parts of it to kernels.
  static const size_t kAlignment = 256;

  virtual ~GpuCrtAllocator() = default;

  // Allocates a buffer of at least `size` bytes.
  // Allocated buffer is associated with `stream`.
  // `stream` is referred to as the "primary stream" for this buffer.
  // Usage on other streams is permitted, but users must
  // synchronize streams appropriately and call RecordUsage() to let
  // the allocator know.
  // When a buffer is used on a non-primary stream, it cannot be reused
  // (i.e. returned from Allocate() again) until the use on a non-primary stream
  // has finished. This can result in extra synchronization and some overheads,
  // especially when operating close to memory capacity.
  virtual llvm::Expected<RCReference<GpuCrtBuffer>> AllocateBuffer(
      size_t size, wrapper::Stream stream) {
    return MakeStringError("This method has not been implemented.");
  }

  // Lets the allocator know that the space identified by `buffer` will
  // not be used in the future. Users are permitted to call this method
  // before already scheduled computation completes.
  virtual void Deallocate(const GpuCrtBuffer& buffer) = 0;

  // Users must call this method if the `buffer` is used on a `stream` that
  // is different from the primary stream, i.e. the stream that was passed
  // to `Allocate()` to create `buffer`.
  // The call can be made anytime between the buffer allocation and
  // deallocation.
  // TODO(iga): Consider adding RecordLastUsage variant of this method. If the
  // user can notify us right after the last usage, we can create an event at
  // that point and synchronize to it (instead of top-of-the-stream during
  // Deallocate()) to know when it is safe to reuse the buffer.
  virtual llvm::Error RecordUsage(const GpuCrtBuffer& buffer,
                                  wrapper::Stream stream) = 0;
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_MEMORY_GPU_ALLOCATOR_H_
