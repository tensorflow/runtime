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

// This file implements a subclass of the MallocAllocator which can be used to
// test graceful failure handling.

#include "tfrt/host_context/host_allocator.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/mutex.h"

namespace tfrt {

class FixedSizeAllocator : public HostAllocator {
 public:
  FixedSizeAllocator(size_t capacity, std::unique_ptr<HostAllocator> allocator)
      : capacity_(capacity), allocator_(std::move(allocator)) {}

 protected:
  // Allocate the specified number of bytes with the specified alignment.
  void* AllocateBytes(size_t size, size_t alignment) override {
    if (!CheckAllocation(size)) {
      return nullptr;
    }

    return allocator_->AllocateBytes(size, alignment);
  }

  // Deallocate the specified pointer that has the specified size.
  void DeallocateBytes(void* ptr, size_t size) override {
    if (size > exclude_threshold_) {
      mutex_lock lock(capacity_lock_);
      current_size_ -= size;
      TFRT_LOG(INFO) << "Freeing " << size << " bytes. "
                     << (capacity_ - current_size_) << " available.";
    }

    allocator_->DeallocateBytes(ptr, size);
  }

 private:
  mutex capacity_lock_;
  size_t current_size_{0};
  size_t capacity_;
  size_t exclude_threshold_{32};
  std::unique_ptr<HostAllocator> allocator_;

  bool CheckAllocation(size_t size) {
    if (size <= exclude_threshold_) {
      return true;
    }

    // For simplicity, we don't adjust size for alignment.
    mutex_lock lock(capacity_lock_);
    if (size + current_size_ <= capacity_) {
      current_size_ += size;
      TFRT_LOG(INFO) << "Allocating " << size << " bytes. "
                     << (capacity_ - current_size_) << " remaining.";
      return true;
    } else {
      TFRT_LOG(ERROR) << "Attempted to allocate " << size << " bytes. However "
                      << "current capacity is only "
                      << capacity_ - current_size_ << ".";
      return false;
    }
  }
};

std::unique_ptr<HostAllocator> CreateFixedSizeAllocator(size_t capacity) {
  return std::make_unique<FixedSizeAllocator>(capacity,
                                              CreateMallocAllocator());
}

}  // namespace tfrt
