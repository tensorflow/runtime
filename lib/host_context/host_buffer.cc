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

//===- host_buffer.cc - HostBuffer implementation -------------------------===//
//
// This file implements the HostBuffer type.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/host_buffer.h"

#include <cstdint>

#include "llvm/Support/Format.h"
#include "tfrt/host_context/host_allocator.h"

namespace tfrt {

RCReference<HostBuffer> HostBuffer::CreateUninitialized(
    size_t size, size_t alignment, HostAllocator *allocator) {
  assert(alignof(std::max_align_t) >= alignment && "Invalid alignment");
  auto *buf =
      allocator->AllocateBytes(sizeof(HostBuffer) + size, alignof(HostBuffer));
  if (!buf) return {};

  return TakeRef(new (buf) HostBuffer(size, allocator));
}

RCReference<HostBuffer> HostBuffer::CreateFromExternal(
    void *ptr, size_t size, Deallocator deallocator) {
  // Not allocated via HostAllocator as HostBuffer::CreateUninitialized.
  return TakeRef(new HostBuffer(ptr, size, std::move(deallocator)));
}

RCReference<HostBuffer> HostBuffer::CreateFromExternal(
    RCReference<HostBuffer> parent_buffer, size_t offset, size_t size) {
  if (!parent_buffer) return {};
  assert(parent_buffer->size() >= offset + size &&
         "Invalid `offset` and `size` for given buffer.");

  auto ptr = static_cast<char *>(parent_buffer->data()) + offset;
  return TakeRef(new HostBuffer(ptr, size, std::move(parent_buffer)));
}

HostBuffer::~HostBuffer() {
  switch (mode_) {
    case Mode::kInlined:
      break;
    case Mode::kOutOfLine:
      out_of_line_.deallocator(out_of_line_.ptr, size_);
      out_of_line_.deallocator.~Deallocator();
      break;
    case Mode::kSliced:
      sliced_.parent_buffer.~RCReference<HostBuffer>();
      break;
  }
}

void HostBuffer::Destroy() {
  switch (mode_) {
    case Mode::kInlined:
      this->~HostBuffer();
      inlined_.allocator->DeallocateBytes(this, sizeof(HostBuffer) + size_);
      break;
    case Mode::kOutOfLine:
    case Mode::kSliced:
      delete this;
      break;
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const HostBuffer &buffer) {
  os << "HostBuffer<pointer="
     << llvm::format_hex(reinterpret_cast<uintptr_t>(buffer.data()), 1)
     << ", size=" << buffer.size() << ">";
  return os;
}

}  // namespace tfrt
