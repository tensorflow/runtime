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

// Unit test for AlignedBuffer

#include "tfrt/support/aligned_buffer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(AlignedBufferTest, AlignedTo64Bytes) {
  AlignedBuffer<64> bef;
  bef.push_back(0xCC);
  auto addr = reinterpret_cast<uintptr_t>(bef.data());
  ASSERT_EQ(addr % 64, 0);
  ASSERT_EQ(bef.size(), 1);
  ASSERT_EQ(bef[0], 0xCC);
}

}  // namespace
}  // namespace tfrt
