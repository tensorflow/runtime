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
// This file defines the GPU CUDA sub allocator.
#ifndef TFRT_GPU_MEMORY_SUB_ALLOCATOR_H_
#define TFRT_GPU_MEMORY_SUB_ALLOCATOR_H_

#include <map>

#include "llvm/Support/Error.h"
#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {

// SubAllocator provides the actual device allocation logic.
class SubAllocator {
 public:
  explicit SubAllocator(wrapper::Platform platform) : platform_(platform) {}

  wrapper::Platform platform() const { return platform_; }

  llvm::Expected<wrapper::Pointer<void>> Allocate(size_t size,
                                                  wrapper::Stream stream);
  void Deallocate(wrapper::Pointer<void> pointer);

 private:
  uintptr_t next_addr_ = 1;

  wrapper::Platform platform_;
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_MEMORY_SUB_ALLOCATOR_H_
