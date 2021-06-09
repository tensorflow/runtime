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

#include "tfrt/host_context/host_allocator.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "llvm/Support/MathExtras.h"
#include "tfrt/support/alloc.h"

namespace tfrt {

class MallocAllocator : public HostAllocator {
  // Allocate the specified number of bytes with the specified alignment.
  void* AllocateBytes(size_t size, size_t alignment) override {
    return AlignedAlloc(alignment, size);
  }

  // Deallocate the specified pointer that has the specified size.
  void DeallocateBytes(void* ptr, size_t size) override { AlignedFree(ptr); }
};

void HostAllocator::VtableAnchor() {}

std::unique_ptr<HostAllocator> CreateMallocAllocator() {
  return std::make_unique<MallocAllocator>();
}

}  // namespace tfrt
