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

//===- host_allocator.cc - Host Memory Allocator Abstraction --------------===//
//
// This file implements the generic interface for host memory allocators.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/host_allocator.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "llvm/Support/MathExtras.h"

namespace {
void* tfrt_aligned_alloc(size_t alignment, size_t size) {
#if defined(__ANDROID__) || defined(OS_ANDROID)
  return memalign(alignment, size);
#else  // !__ANDROID__ && !OS_ANDROID
  void* ptr = nullptr;
  // posix_memalign requires that the requested alignment be at least
  // sizeof(void*). In this case, fall back on malloc which should return memory
  // aligned to at least the size of a pointer.
  const int required_alignment = sizeof(void*);
  if (alignment < required_alignment) return malloc(size);
  if (posix_memalign(&ptr, alignment, size) != 0)
    return nullptr;
  else
    return ptr;
#endif
}
}  // namespace

namespace tfrt {

class MallocAllocator : public HostAllocator {
  // Allocate the specified number of bytes with the specified alignment.
  void* AllocateBytes(size_t size, size_t alignment) override {
    if (alignment <= 8) return malloc(size);

    // aligned_alloc requires the size to be a multiple of the alignment.
    size = llvm::alignTo(size, alignment, /*skew=*/0);
    return tfrt_aligned_alloc(alignment, size);
  }

  // Deallocate the specified pointer that has the specified size.
  void DeallocateBytes(void* ptr, size_t size) override { free(ptr); }
};

void HostAllocator::VtableAnchor() {}

std::unique_ptr<HostAllocator> CreateMallocAllocator() {
  return std::make_unique<MallocAllocator>();
}

}  // namespace tfrt
