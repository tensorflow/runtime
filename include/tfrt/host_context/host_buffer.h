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

#include "llvm/ADT/FunctionExtras.h"
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

  using Deallocator = llvm::unique_function<void(void *ptr, size_t size)>;
  // Create a HostBuffer by taking ownership of an externally allocated buffer.
  // `deallocator` is called with `ptr` and `size` as arguments when we destroy
  // this buffer.
  static RCReference<HostBuffer> CreateFromExternal(void *ptr, size_t size,
                                                    Deallocator deallocator);

  void *data() {
    if (is_inlined_) return &inlined_.data[0];
    return out_of_line_.ptr;
  }

  const void *data() const {
    if (is_inlined_) return &inlined_.data[0];
    return out_of_line_.ptr;
  }
  size_t size() const { return size_; }

 private:
  // For access to Destroy().
  friend class ReferenceCounted<HostBuffer>;

  HostBuffer(size_t size, HostAllocator *allocator)
      : size_(size), is_inlined_(true), inlined_{.allocator = allocator} {}

  HostBuffer(void *ptr, size_t size, Deallocator deallocator)
      : size_(size),
        is_inlined_(false),
        out_of_line_{.ptr = ptr, .deallocator = std::move(deallocator)} {}

  ~HostBuffer();

  void Destroy();

  size_t size_ : 63;
  bool is_inlined_ : 1;
  // TODO(zhangqiaorjc): Use variant instead of union.
  union {
    struct {
      HostAllocator *allocator;
      // The data is allocated in the flexible memory array.
      alignas(alignof(std::max_align_t)) char data[];
    } inlined_;

    struct {
      void *ptr;
      Deallocator deallocator;
    } out_of_line_;
  };
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const HostBuffer &buffer);

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_HOST_BUFFER_H_
