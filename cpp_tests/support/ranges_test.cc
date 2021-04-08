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

// Tests related to ranges.
#include "tfrt/support/ranges.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/support/ranges_util.h"

namespace tfrt {
namespace {

TEST(RangeTest, CountedView) {
  int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<int> values;
  for (auto& i : views::Counted(a, 4)) values.push_back(i);
  EXPECT_THAT(values, testing::ElementsAre(1, 2, 3, 4));

  EXPECT_THAT(AsVector(views::Counted(a, 3)), testing::ElementsAre(1, 2, 3));
  EXPECT_THAT(AsSmallVector<4>(views::Counted(a, 3)),
              testing::ElementsAre(1, 2, 3));

  const auto il = {1, 2, 3, 4, 5};
  EXPECT_THAT(AsVector(views::Counted(il.begin() + 1, 3)),
              testing::ElementsAre(2, 3, 4));
  EXPECT_THAT(AsSmallVector<4>(views::Counted(il.begin() + 1, 3)),
              testing::ElementsAre(2, 3, 4));
}

}  // namespace
}  // namespace tfrt
