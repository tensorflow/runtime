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

// Tests related to MapByType
#include "tfrt/support/map_by_type.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tfrt {
namespace {

struct T1 {
  int v = 0;
};

struct T2 {
  int v = 1;
  int* destruct_count;
  explicit T2(int* count) : destruct_count(count) {}
  ~T2() { (*destruct_count)++; }
};

struct MapTag {};

TEST(MapByTypeTest, Basic) {
  MapByType<MapTag> map;
  EXPECT_FALSE(map.contains<int>());

  auto& i = map.insert(2);
  EXPECT_TRUE(map.contains<int>());
  EXPECT_TRUE(map.contains<const int>());
  EXPECT_EQ(map.get<int>(), 2);
  EXPECT_EQ(map.get<const int>(), 2);

  i = 3;
  EXPECT_EQ(map.get<int>(), 3);
  EXPECT_EQ(map.get<const int>(), 3);

  EXPECT_EQ(*map.getIfExists<int>(), 3);

  EXPECT_EQ(map.getIfExists<float>(), nullptr);

  EXPECT_FALSE(map.contains<T1>());
  auto& t1 = map.emplace<T1>(T1{2});
  EXPECT_EQ(t1.v, 2);
  EXPECT_TRUE(map.contains<T1>());
  EXPECT_EQ(map.get<T1>().v, 2);

  struct A {
    int v = 3;
  };

  EXPECT_FALSE(map.contains<A>());
  A t2_lvalue;
  auto& t2 = map.insert(t2_lvalue);
  EXPECT_EQ(t2.v, 3);
  EXPECT_TRUE(map.contains<A>());
  EXPECT_EQ(map.get<A>().v, 3);
}

TEST(MapByTypeTest, DestructorCount) {
  int destruct_count = 0;
  {
    MapByType<MapTag> map2;
    {
      auto t2 = std::make_unique<T2>(&destruct_count);
      auto& t2_ref = map2.emplace<std::unique_ptr<T2>>(std::move(t2));
      EXPECT_EQ(t2_ref->v, 1);
    }
    {
      auto first_use = map2.get<std::unique_ptr<T2>>().get();
      EXPECT_EQ(first_use->v, 1);
    }
    {
      auto second_use = map2.get<std::unique_ptr<T2>>().get();
      EXPECT_EQ(second_use->v, 1);
    }
    EXPECT_EQ(destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 1);
}

}  // namespace
}  // namespace tfrt
