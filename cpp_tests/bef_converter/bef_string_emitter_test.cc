/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

#include "../../lib/bef_converter/mlir_to_bef/bef_string_emitter.h"

#include "gtest/gtest.h"

namespace tfrt {
namespace {

static constexpr char kTestStr1[] = "Apple";
TEST(BefStringEmitterTest, EmitSingleString) {
  BefStringEmitter emitter;
  size_t offset = emitter.EmitString(kTestStr1);

  auto buffer = emitter.TakeResult();
  EXPECT_EQ(strcmp(reinterpret_cast<char *>(buffer.data() + offset), kTestStr1),
            0);
}

static constexpr char kTestStr2[] = "Banana";
TEST(BefStringEmitterTest, EmitTwoStrings) {
  BefStringEmitter emitter;
  size_t first_offset = emitter.EmitString(kTestStr1);
  size_t second_offset = emitter.EmitString(kTestStr2);

  auto buffer = emitter.TakeResult();
  EXPECT_EQ(
      strcmp(reinterpret_cast<char *>(buffer.data() + first_offset), kTestStr1),
      0);
  EXPECT_EQ(strcmp(reinterpret_cast<char *>(buffer.data() + second_offset),
                   kTestStr2),
            0);
}

TEST(BefStringEmitterTest, EmitDuplicateString) {
  BefStringEmitter emitter;
  size_t first_offset = emitter.EmitString(kTestStr1);
  size_t second_offset = emitter.EmitString(kTestStr1);
  EXPECT_EQ(first_offset, second_offset);
}

}  // namespace
}  // namespace tfrt
