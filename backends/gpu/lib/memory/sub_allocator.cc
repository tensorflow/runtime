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

//===- sub_allocator.cc - SubAllocator -------------------------*- C++ -*--===//
//
// This file implements the C++ interface to CUDA caching allocator.
#include "tfrt/gpu/memory/sub_allocator.h"

#include <llvm/Support/Errc.h>

#include <cstdint>

#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {

llvm::Expected<wrapper::Pointer<void>> SubAllocator::Allocate(
    size_t size, wrapper::Stream stream) {
  size_t mask = GpuCrtAllocator::kAlignment - 1;
  if (GpuCrtAllocator::kAlignment & mask) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Alignment must be power of two.");
  }
  uintptr_t address = (next_addr_ + mask) & ~mask;
  next_addr_ = address + size;
  return wrapper::Pointer<void>(reinterpret_cast<void*>(address), platform_);
}

void SubAllocator::Deallocate(wrapper::Pointer<void> pointer) {
  // TODO(apryakhin): Fill up with the deallocation logic and
  // consider providing a dedicated stream.
}

}  // namespace gpu
}  // namespace tfrt
