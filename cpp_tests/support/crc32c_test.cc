// Copyright 2021 The TensorFlow Runtime Authors
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

#include "tfrt/support/crc32c.h"

#include "gtest/gtest.h"

namespace tfrt {
namespace crc32c {

extern bool CanAccelerate();
extern uint32_t AcceleratedExtend(uint32_t crc, const char* buf, size_t size);
extern uint32_t RegularExtend(uint32_t crc, const char* buf, size_t size);

namespace {

constexpr char kTestBuffer[] = "Copyright 2021 The TensorFlow Runtime Authors";

TEST(Crc32cTest, Extend) {
  uint32_t crc1 = Extend(0, kTestBuffer, sizeof(kTestBuffer));
  uint32_t crc2 = Extend(0, kTestBuffer, sizeof(kTestBuffer) / 2);
  uint32_t crc3 = Extend(crc2, kTestBuffer + sizeof(kTestBuffer) / 2,
                         sizeof(kTestBuffer) - sizeof(kTestBuffer) / 2);
  EXPECT_EQ(crc1, crc3);
}

TEST(Crc32cTest, RegularExtend) {
  uint32_t crc1 = RegularExtend(0, kTestBuffer, sizeof(kTestBuffer));
  uint32_t crc2 = RegularExtend(0, kTestBuffer, sizeof(kTestBuffer) / 2);
  uint32_t crc3 = RegularExtend(crc2, kTestBuffer + sizeof(kTestBuffer) / 2,
                                sizeof(kTestBuffer) - sizeof(kTestBuffer) / 2);
  EXPECT_EQ(crc1, crc3);
}

TEST(Crc32cTest, AcceleratedExtend) {
  if (CanAccelerate()) {
    for (int buffer_size = 1; buffer_size <= sizeof(kTestBuffer);
         ++buffer_size) {
      uint32_t crc1 = AcceleratedExtend(0, kTestBuffer, sizeof(kTestBuffer));
      uint32_t crc2 = RegularExtend(0, kTestBuffer, sizeof(kTestBuffer));
      EXPECT_EQ(crc1, crc2);
    }
  }
}

}  // namespace
}  // namespace crc32c
}  // namespace tfrt
