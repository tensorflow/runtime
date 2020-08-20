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

//===- host_buffer_test.cc --------------------------------------*- C++ -*-===//
//
// Unit test for TFRT HostBuffer.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/host_buffer.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/host_context/host_allocator.h"

namespace {

TEST(HostBufferTest, Alignment) {
  auto is_aligned = [](void* ptr, size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
  };

  auto allocator = tfrt::CreateMallocAllocator();

  for (auto alignment : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
    auto host_buffer =
        tfrt::HostBuffer::CreateUninitialized(10, alignment, allocator.get());
    EXPECT_TRUE(is_aligned(host_buffer->data(), alignment));
  }
}
}  // namespace
