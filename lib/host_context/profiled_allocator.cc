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

//===- profiled_allocator.cc - Profiled Memory Allocator ------------------===//
//
// This file implements a profiling host memory allocator that does a memory
// leak check and prints allocation statistics when destroyed.

#include "tfrt/host_context/profiled_allocator.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "tfrt/host_context/host_allocator.h"

namespace tfrt {

namespace {

template <typename T>
void AtomicUpdateMax(T const& value, std::atomic<T>* max_value) noexcept {
  T prev_max_value = *max_value;
  // Note that compare_exchange_weak updates `prev_max_value` on failure.
  while (prev_max_value < value &&
         !max_value->compare_exchange_weak(prev_max_value, value)) {
  }
}

}  // namespace

class ProfiledAllocator : public HostAllocator {
 public:
  explicit ProfiledAllocator(std::unique_ptr<HostAllocator> allocator)
      : allocator_(std::move(allocator)) {}

  ~ProfiledAllocator() override {
    if (print_profile_) {
      PrintStats();
    }
  }

  void* AllocateBytes(size_t size, size_t alignment) override {
    ++curr_num_allocations_;
    ++cum_num_allocations_;
    curr_num_bytes_allocated_.fetch_add(size);
    AtomicUpdateMax<int64_t>(curr_num_allocations_, &max_num_allocations_);
    AtomicUpdateMax<int64_t>(curr_num_bytes_allocated_,
                             &max_num_bytes_allocated_);

    return allocator_->AllocateBytes(size, alignment);
  }

  void DeallocateBytes(void* ptr, size_t size) override {
    --curr_num_allocations_;
    curr_num_bytes_allocated_.fetch_sub(size);

    allocator_->DeallocateBytes(ptr, size);
  }

 protected:
  void PrintStats() const {
    printf("HostAllocator profile:\n");
    printf("Current number of allocations = %" PRId64 "\n",
           curr_num_allocations_.load());
    printf("Max number of allocations = %" PRId64 "\n",
           max_num_allocations_.load());
    printf("Total number of allocations = %" PRId64 "\n",
           cum_num_allocations_.load());
    printf("Current number of bytes allocated = %" PRId64 "\n",
           curr_num_bytes_allocated_.load());
    printf("Max number of bytes allocated = %" PRId64 "\n",
           max_num_bytes_allocated_.load());
    fflush(stdout);
  }

  bool print_profile_ = true;
  std::atomic<int64_t> curr_num_allocations_{0};
  std::atomic<int64_t> max_num_allocations_{0};
  std::atomic<int64_t> cum_num_allocations_{0};
  std::atomic<int64_t> curr_num_bytes_allocated_{0};
  std::atomic<int64_t> max_num_bytes_allocated_{0};

 private:
  std::unique_ptr<HostAllocator> allocator_;
};

class LeakCheckAllocator : public ProfiledAllocator {
 public:
  explicit LeakCheckAllocator(std::unique_ptr<HostAllocator> allocator)
      : ProfiledAllocator(std::move(allocator)) {
    print_profile_ = false;
  }

  // Cause process to exit(1) when memory leak is detected.
  ~LeakCheckAllocator() override {
    if (curr_num_bytes_allocated_.load() != 0) {
      PrintStats();
      printf("Memory leak detected: %" PRId64 " alive allocations, %" PRId64
             " alive bytes\n",
             curr_num_allocations_.load(), curr_num_bytes_allocated_.load());
      fflush(stdout);
      exit(1);
    }
  }
};

std::unique_ptr<HostAllocator> CreateProfiledAllocator(
    std::unique_ptr<HostAllocator> allocator) {
  return std::make_unique<ProfiledAllocator>(std::move(allocator));
}

std::unique_ptr<HostAllocator> CreateLeakCheckAllocator(
    std::unique_ptr<HostAllocator> allocator) {
  return std::make_unique<LeakCheckAllocator>(std::move(allocator));
}
}  // namespace tfrt
