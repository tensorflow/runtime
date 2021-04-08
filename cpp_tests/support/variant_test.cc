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

// Tests related to variant.
#include "tfrt/support/variant.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tfrt {
namespace {

struct T1 {
  int v = 0;
};

struct T2 {
  int64_t v = 0;
};

TEST(VariantTest, Basic) {
  auto check = [](const Variant<T1, T2>& variant, auto&& t) {
    using Type = std::decay_t<decltype(t)>;
    ASSERT_TRUE(variant.is<Type>());
    ASSERT_EQ(variant.get<Type>().v, t.v);

    visit(
        [&](auto& e) {
          using T = std::decay_t<decltype(e)>;
          ASSERT_TRUE((std::is_same<Type, T>::value));
          ASSERT_EQ(e.v, t.v);
        },
        variant);
  };

  Variant<T1, T2> v1;
  check(v1, T1{0});

  v1 = T2{2};
  check(v1, T2{2});

  v1.emplace<T1>(T1{1});
  check(v1, T1{1});

  auto v2 = v1;
  check(v2, T1{1});
}

TEST(VariantTest, DuplicatedType) {
  Variant<T1, T1> v1;

  ASSERT_TRUE(v1.is<T1>());
  ASSERT_EQ(v1.get<T1>().v, 0);

  v1 = T1{1};
  ASSERT_TRUE(v1.is<T1>());
  ASSERT_EQ(v1.get<T1>().v, 1);

  visit([&](auto& e) { ASSERT_EQ(e.v, 1); }, v1);
}

struct MoveOnlyT {
  MoveOnlyT() = default;
  MoveOnlyT(const MoveOnlyT&) = delete;
  MoveOnlyT(MoveOnlyT&&) = default;
  explicit MoveOnlyT(int v) : v(v) {}

  int v = 0;
};

TEST(VariantTest, MoveOnlyType) {
  Variant<T1, MoveOnlyT> v1;

  ASSERT_TRUE(v1.is<T1>());
  ASSERT_EQ(v1.get<T1>().v, 0);

  v1 = T1{1};
  ASSERT_TRUE(v1.is<T1>());
  ASSERT_EQ(v1.get<T1>().v, 1);

  visit([&](auto& e) { ASSERT_EQ(e.v, 1); }, v1);

  v1 = MoveOnlyT{1};
  ASSERT_TRUE(v1.is<MoveOnlyT>());
}

}  // namespace
}  // namespace tfrt
