/*
 * Copyright 2022 The TensorFlow Runtime Authors
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
#include "tfrt/bef/bef.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(BefTest, VectorOfTrivial) {
  tfrt::bef::Buffer buffer;
  tfrt::bef::Allocator alloc(&buffer);

  auto ctor = bef::New<bef::Vector<uint32_t>>(&alloc, /*size=*/4);

  for (int i = 0; i < 4; ++i) {
    ctor.ConstructAt(i, i);
  }

  bef::Vector<uint32_t> view(buffer.Get(ctor.address()));

  ASSERT_EQ(view.size(), 4);
  EXPECT_EQ(view[0], 0);
  EXPECT_EQ(view[1], 1);
  EXPECT_EQ(view[2], 2);
  EXPECT_EQ(view[3], 3);

  EXPECT_THAT(view, testing::ElementsAreArray({0, 1, 2, 3}));

  bef::Vector<uint32_t> empty;
  ASSERT_TRUE(empty.empty());
}

TEST(BefTest, VectorOfVector) {
  tfrt::bef::Buffer buffer;
  tfrt::bef::Allocator alloc(&buffer);

  using T = bef::Vector<uint32_t>;
  using V = bef::Vector<T>;

  auto vctor = bef::New<V>(&alloc, 3);

  {
    auto tctor = vctor.ConstructAt(0, 2);
    tctor.ConstructAt(0, 0);
    tctor.ConstructAt(1, 1);
  }

  {
    auto tctor = vctor.ConstructAt(1, 1);
    tctor.ConstructAt(0, 2);
  }

  vctor.ConstructAt(2, 0);

  V v(buffer.Get(vctor.address()));

  auto t0 = v[0];
  ASSERT_EQ(t0.size(), 2);
  EXPECT_EQ(t0[0], 0);
  EXPECT_EQ(t0[1], 1);
  EXPECT_THAT(t0, testing::ElementsAreArray({0, 1}));

  auto t1 = v[1];
  ASSERT_EQ(t1.size(), 1);
  EXPECT_EQ(t1[0], 2);
  EXPECT_THAT(t1, testing::ElementsAreArray({2}));

  auto t2 = v[2];
  ASSERT_EQ(t2.size(), 0);

  bef::Vector<bef::Vector<uint32_t>> empty;
  ASSERT_TRUE(empty.empty());
}

}  // namespace
}  // namespace tfrt
