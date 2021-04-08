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

// Unit tests for TFRT HostAllocator and HostArray classes.

#include "tfrt/host_context/host_allocator.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace tfrt {
namespace {

class HostAllocatorTest : public ::testing::Test {
 protected:
  HostAllocatorTest() : allocator_(CreateMallocAllocator()) {}
  std::unique_ptr<HostAllocator> allocator_;
};

constexpr size_t kTestAllocateSize = 1024;
TEST_F(HostAllocatorTest, AllocateDeallocateBytesWithAlignment) {
  auto is_aligned = [](void* ptr, size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
  };

  for (size_t alignment : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
    void* buffer = allocator_->AllocateBytes(kTestAllocateSize, alignment);
    ASSERT_NE(nullptr, buffer);
    EXPECT_TRUE(is_aligned(buffer, alignment));
    memset(buffer, 0, kTestAllocateSize);
    allocator_->DeallocateBytes(buffer, kTestAllocateSize);
  }
}

constexpr size_t kTestAllocateEntryCount = 100;
TEST_F(HostAllocatorTest, AllocateDeallocate) {
  uint64_t* entries = allocator_->Allocate<uint64_t>(kTestAllocateEntryCount);
  for (int idx = 0; idx < kTestAllocateEntryCount; ++idx) entries[idx] = idx;
  allocator_->Deallocate<uint64_t>(entries, kTestAllocateEntryCount);
}

// Tests for HostArray class.
constexpr size_t kTestArraySize = 16;
class HostArrayTest : public ::testing::Test {
 protected:
  HostArrayTest() {
    int idx = 0;
    for (auto& str : host_array_.mutable_array()) {
      new (&str) std::string(std::to_string(idx++));
    }
  }

  std::unique_ptr<HostAllocator> allocator_{CreateMallocAllocator()};
  HostArray<std::string> host_array_{kTestArraySize, allocator_.get()};
};

TEST_F(HostArrayTest, GetSize) {
  EXPECT_EQ(kTestArraySize, host_array_.size());
}

TEST_F(HostArrayTest, ArrayReference) {
  for (int idx = 0; idx < kTestArraySize; ++idx)
    EXPECT_EQ(std::to_string(idx), host_array_[idx]);
}

TEST_F(HostArrayTest, GetArray) {
  ArrayRef<std::string> array_ref = host_array_.array();
  EXPECT_EQ(kTestArraySize, array_ref.size());
  EXPECT_EQ("10", array_ref[10]);
}

TEST_F(HostArrayTest, GetMutableArray) {
  MutableArrayRef<std::string> array_ref = host_array_.mutable_array();
  array_ref[0] = "new";
}

TEST_F(HostArrayTest, Move) {
  HostArray<std::string> moved_array(std::move(host_array_));
  EXPECT_EQ(kTestArraySize, moved_array.size());
  EXPECT_EQ(0, host_array_.size());
}

}  // namespace
}  // namespace tfrt
