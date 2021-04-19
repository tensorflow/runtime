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

// GPU caching allocator
//
// This file defines the interface for a caching GPU memory allocator.
#ifndef TFRT_GPU_MEMORY_CACHING_GPU_ALLOCATOR_H_
#define TFRT_GPU_MEMORY_CACHING_GPU_ALLOCATOR_H_

#include <map>

#include "llvm/Support/Error.h"
#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {
class GpuContext;

// CachingGpuAllocator is thread-safe.
class CachingGpuAllocator : public GpuCrtAllocator {
 public:
  explicit CachingGpuAllocator(AsyncValueRef<GpuContext> context);
  ~CachingGpuAllocator() override;

  llvm::Expected<RCReference<gpu::GpuCrtBuffer>> Allocate(
      size_t size, wrapper::Stream stream) override;

  void Deallocate(const gpu::GpuCrtBuffer& buffer) override;

  llvm::Error RecordUsage(const gpu::GpuCrtBuffer& buffer,
                          wrapper::Stream stream) override;

 private:
  std::map<wrapper::Pointer<void>, wrapper::Context> allocations_;
  AsyncValueRef<GpuContext> context_;
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_MEMORY_CACHING_GPU_ALLOCATOR_H_
