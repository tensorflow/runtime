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

#include "tfrt/dtype/dtype.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tfrt {
namespace {

const char* GetName(DType::Kind kind) { return DType(kind).GetName(); }
size_t GetHostSize(DType::Kind kind) { return DType(kind).GetHostSize(); }
size_t GetHostAlignment(DType::Kind kind) {
  return DType(kind).GetHostAlignment();
}

TEST(DType, QueryTraits) {
  EXPECT_STREQ(GetName(DType::F16), "f16");
  EXPECT_STREQ(GetName(DType::Resource), "resource");
  EXPECT_STREQ(GetName(DType::Variant), "variant");

  EXPECT_EQ(GetHostSize(DType::I8), 1);
  EXPECT_EQ(GetHostSize(DType::F16), 2);
  EXPECT_EQ(GetHostSize(DType::BF16), 2);
  EXPECT_EQ(GetHostSize(DType::Complex64), 8);
  EXPECT_EQ(GetHostSize(DType::Complex128), 16);
  EXPECT_EQ(GetHostSize(DType::String), -1);
  EXPECT_EQ(GetHostSize(DType::Resource), -1);
  EXPECT_EQ(GetHostSize(DType::Variant), -1);

  EXPECT_EQ(GetHostAlignment(DType::I8), 1);
  EXPECT_EQ(GetHostAlignment(DType::Complex64), 4);
  EXPECT_EQ(GetHostAlignment(DType::Complex128), 8);
  EXPECT_EQ(GetHostAlignment(DType::F16), 2);
  EXPECT_EQ(GetHostAlignment(DType::BF16), 2);
  EXPECT_EQ(GetHostAlignment(DType::String), -1);
  EXPECT_EQ(GetHostAlignment(DType::Resource), -1);
  EXPECT_EQ(GetHostAlignment(DType::Variant), -1);
}

}  // namespace
}  // namespace tfrt
