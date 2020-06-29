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

//===- value_test.cc --------------------------------------------*- C++ -*-===//
//
// Unit test for TFRT Value.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/value.h"

#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(ValueTest, InPlace) {
  Value v0;
  ASSERT_FALSE(v0.HasValue());

  // Testing move of an empty Value.
  Value v0_moved{std::move(v0)};
  ASSERT_FALSE(v0_moved.HasValue());

  Value v1{2};
  static_assert(Value::IsInPlace<int>());

  ASSERT_TRUE(v1.HasValue());
  ASSERT_EQ(v1.get<int>(), 2);
  ASSERT_TRUE(v1.IsType<int>());
  ASSERT_FALSE(v1.IsType<float>());

  v1.emplace<float>(1.0);

  ASSERT_EQ(v1.get<float>(), 1.0);
  ASSERT_TRUE(v1.IsType<float>());
  ASSERT_FALSE(v1.IsType<int>());

  v1.set(3);

  ASSERT_EQ(v1.get<int>(), 3);
  ASSERT_TRUE(v1.IsType<int>());
  ASSERT_FALSE(v1.IsType<float>());

  Value v2{1.0};

  v2 = std::move(v1);

  ASSERT_EQ(v2.get<int>(), 3);
  ASSERT_TRUE(v2.IsType<int>());
  ASSERT_FALSE(v2.IsType<float>());
  ASSERT_FALSE(v1.HasValue());  // NOLINT Disable use after move warning

  v2.reset();
  ASSERT_FALSE(v2.HasValue());
}

struct BigType1 {
  explicit BigType1(int v) : v(v) {}

  int v = 0;
  // A large array to make BigType1 stored out of place in Value.
  char array[128]{};

  bool operator==(int a) const { return v == a; }
};

struct BigType2 {
  explicit BigType2(int v) : v(v) {}

  // A large array to make BigType1 stored out of place in Value.
  char array[128]{};
  int v = 0;

  bool operator==(int a) const { return v == a; }
};
static_assert(!Value::IsInPlace<BigType1>());
static_assert(!Value::IsInPlace<BigType2>());

TEST(ValueTest, OutOfPlace) {
  Value v1{0};
  v1.set(BigType1{2});

  ASSERT_EQ(v1.get<BigType1>(), 2);
  ASSERT_TRUE(v1.IsType<BigType1>());
  ASSERT_FALSE(v1.IsType<float>());

  v1.emplace<BigType2>(1);

  ASSERT_EQ(v1.get<BigType2>(), 1);
  ASSERT_TRUE(v1.IsType<BigType2>());
  ASSERT_FALSE(v1.IsType<int>());

  v1.set(BigType1{3});

  ASSERT_EQ(v1.get<BigType1>(), 3);
  ASSERT_TRUE(v1.IsType<BigType1>());
  ASSERT_FALSE(v1.IsType<float>());

  Value v2{1.0};

  v2 = std::move(v1);

  ASSERT_EQ(v2.get<BigType1>(), 3);
  ASSERT_TRUE(v2.IsType<BigType1>());
  ASSERT_FALSE(v2.IsType<float>());
  ASSERT_FALSE(v1.HasValue());  // NOLINT Disable use after move warning

  v2.reset();
  ASSERT_FALSE(v2.HasValue());
}

}  // namespace
}  // namespace tfrt
