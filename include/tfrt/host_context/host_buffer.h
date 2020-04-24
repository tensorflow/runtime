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

//===- host_buffer.h - Reference counted host buffer ------------*- C++ -*-===//
//
// This file declares HostBuffer.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_HOST_BUFFER_H_
#define TFRT_HOST_CONTEXT_HOST_BUFFER_H_

#include "llvm/Support/raw_ostream.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
class HostAllocator;

// HostBuffer is a reference counted chunk of untyped memory on the host, whose
// memory is managed by HostContext.
class HostBuffer : public ReferenceCounted<HostBuffer> {
 public:
  // Create an uninitialized HostBuffer of the specified size and alignment.
  // This returns a null RCReference on allocation failure.
  // `allocator` will be used to allocate the memory and to deallocate it
  // when the returned buffer is destroyed.
  static RCReference<HostBuffer> CreateUninitialized(size_t size,
                                                     size_t alignment,
                                                     HostAllocator *allocator);
  void *data() { return &data_[0]; }
  const void *data() const { return &data_[0]; }
  size_t size() const { return size_; }

 private:
  // For access to Destroy().
  friend class ReferenceCounted<HostBuffer>;

  HostBuffer(size_t size, HostAllocator *allocator)
      : size_(size), allocator_(allocator) {}

  void Destroy();

  size_t size_;
  HostAllocator *allocator_;

  // The data is allocated in the flexible memory array.
  alignas(alignof(std::max_align_t)) char data_[];
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const HostBuffer &buffer);

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_HOST_BUFFER_H_
