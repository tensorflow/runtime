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

//===- gpu_buffer.cc - Types for holding GPU memory -----------------------===//
//
// This file implements classes that can be used to hold CUDA buffers
// and tensors.

#include "tfrt/gpu/memory/gpu_buffer.h"

#include "llvm/Support/Format.h"
#include "tfrt/gpu/memory/gpu_allocator.h"

namespace tfrt {
namespace gpu {

GpuBuffer::GpuBuffer(gpu::stream::Pointer<void> pointer, size_t size,
                     GpuAllocator* allocator)
    : pointer_(pointer),
      size_(size),
      has_allocator_(true),
      allocator_(allocator) {}

GpuBuffer::GpuBuffer(gpu::stream::Pointer<void> pointer, size_t size,
                     Deallocator deallocator)
    : pointer_(pointer),
      size_(size),
      has_allocator_(false),
      deallocator_(std::move(deallocator)) {}

GpuBuffer::~GpuBuffer() {
  if (has_allocator_) {
    if (allocator_ != nullptr) {
      allocator_->Deallocate(*this);
    }
  } else {
    deallocator_(this);
    deallocator_.~Deallocator();
  }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const GpuBuffer& buffer) {
  os << "GpuBuffer<pointer=" << buffer.pointer() << ", size=" << buffer.size()
     << ">";
  return os;
}

}  // namespace gpu
}  // namespace tfrt
