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

// Unit test for TFRT HostBuffer.

#include "tfrt/host_context/host_buffer.h"

#include "gtest/gtest.h"
#include "tfrt/host_context/host_allocator.h"

namespace tfrt {
namespace {

class TestAllocator : public HostAllocator {
 public:
  explicit TestAllocator(std::unique_ptr<HostAllocator> allocator)
      : allocator_(std::move(allocator)) {}

  void* AllocateBytes(size_t size, size_t alignment) override {
    bytes_allocated_ += size;

    return allocator_->AllocateBytes(size, alignment);
  }

  void DeallocateBytes(void* ptr, size_t size) override {
    bytes_allocated_ -= size;

    allocator_->DeallocateBytes(ptr, size);
  }

  bool HasMemory() const { return bytes_allocated_ != 0; }

 private:
  std::unique_ptr<HostAllocator> allocator_;
  int64_t bytes_allocated_ = 0;
};

class HostBufferTest : public ::testing::Test {
 protected:
  HostBufferTest() : malloc_allocator_{CreateMallocAllocator()} {}

  static bool IsAligned(void* ptr, size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
  }

  std::unique_ptr<HostAllocator> malloc_allocator_;
};

TEST_F(HostBufferTest, Alignment) {
  for (auto alignment : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
    auto host_buffer =
        HostBuffer::CreateUninitialized(10, alignment, malloc_allocator_.get());
    EXPECT_TRUE(IsAligned(host_buffer->data(), alignment));
  }
}

TEST_F(HostBufferTest, MemoryLeak) {
  TestAllocator allocator(CreateMallocAllocator());

  for (auto alignment : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
    auto host_buffer =
        HostBuffer::CreateUninitialized(10, alignment, &allocator);

    host_buffer.reset();
    EXPECT_FALSE(allocator.HasMemory());
  }
}

const size_t kTestAllocSize = 1024 * 1024;
const size_t kTestAlignment = 64;
TEST_F(HostBufferTest, CreateUninitialized) {
  RCReference<HostBuffer> host_buffer = HostBuffer::CreateUninitialized(
      kTestAllocSize, kTestAlignment, malloc_allocator_.get());

  memset(host_buffer->data(), 0, kTestAllocSize);

  EXPECT_EQ(kTestAllocSize, host_buffer->size());
  EXPECT_TRUE(IsAligned(host_buffer->data(), kTestAlignment));
  EXPECT_TRUE(host_buffer->IsExclusiveDataOwner());
}

TEST_F(HostBufferTest, CreateFromExternal) {
  RCReference<HostBuffer> parent_buffer = HostBuffer::CreateUninitialized(
      kTestAllocSize, kTestAlignment, malloc_allocator_.get());

  RCReference<HostBuffer> child_buffer = HostBuffer::CreateFromExternal(
      std::move(parent_buffer), kTestAllocSize / 2, kTestAllocSize / 2);

  EXPECT_EQ(kTestAllocSize / 2, child_buffer->size());
  EXPECT_TRUE(child_buffer->IsExclusiveDataOwner());
}

TEST_F(HostBufferTest, CastAs) {
  std::array<int, 3> data{1, 2, 3};
  RCReference<HostBuffer> host_buffer = HostBuffer::CreateFromExternal(
      data.data(), data.size() * sizeof(int), [](void*, size_t) {});

  MutableArrayRef<int> mutable_array_ref = host_buffer->CastAs<int>();
  EXPECT_EQ(mutable_array_ref.size(), 3);
  EXPECT_EQ(mutable_array_ref[0], 1);

  const HostBuffer& const_hb = *host_buffer;
  ArrayRef<int> array_ref = const_hb.CastAs<int>();
  EXPECT_EQ(array_ref.size(), 3);
  EXPECT_EQ(array_ref[0], 1);
}

}  // namespace
}  // namespace tfrt
