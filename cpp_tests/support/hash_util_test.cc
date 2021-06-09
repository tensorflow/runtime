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

#include "tfrt/support/hash_util.h"

#include "gtest/gtest.h"

namespace tfrt {
namespace {

constexpr char kTestBuffer[] = "Copyright 2021 The TensorFlow Runtime Authors";

TEST(HashUtilTest, Hash32SameSeed) {
  for (int buffer_size = 1; buffer_size < sizeof(kTestBuffer); ++buffer_size) {
    EXPECT_EQ(Hash32(kTestBuffer, buffer_size, buffer_size),
              Hash32(kTestBuffer, buffer_size, buffer_size));
  }
}

TEST(HashUtilTest, Hash32DifferentSeeds) {
  for (int buffer_size = 1; buffer_size < sizeof(kTestBuffer); ++buffer_size) {
    EXPECT_NE(Hash32(kTestBuffer, buffer_size, buffer_size),
              Hash32(kTestBuffer, buffer_size, buffer_size + 1));
  }
}

TEST(HashUtilTest, Hash32CheckAlgorithm) {
  EXPECT_EQ(Hash32(kTestBuffer, sizeof(kTestBuffer), 0), 0xff2938e5);
}

TEST(HashUtilTest, Hash64SameSeed) {
  for (int buffer_size = 1; buffer_size < sizeof(kTestBuffer); ++buffer_size) {
    EXPECT_EQ(Hash64(kTestBuffer, buffer_size, buffer_size),
              Hash64(kTestBuffer, buffer_size, buffer_size));
  }
}

TEST(HashUtilTest, Hash64DifferentSeeds) {
  for (int buffer_size = 1; buffer_size < sizeof(kTestBuffer); ++buffer_size) {
    EXPECT_NE(Hash64(kTestBuffer, buffer_size, buffer_size),
              Hash64(kTestBuffer, buffer_size, buffer_size + 1));
  }
}

TEST(HashUtilTest, Hash64CheckAlgorithm) {
  EXPECT_EQ(Hash64(kTestBuffer, sizeof(kTestBuffer), 0), 0x32e09401ac25876d);
}

}  // namespace
}  // namespace tfrt
