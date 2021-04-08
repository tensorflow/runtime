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

// Unit test for the functions defined in bef_encoding.h

#include "tfrt/support/bef_encoding.h"

#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(BefEncodingTest, IsValidAlignment) {
  EXPECT_FALSE(IsValidAlignment(0));
  EXPECT_TRUE(IsValidAlignment(1));
  EXPECT_TRUE(IsValidAlignment(2));
  EXPECT_FALSE(IsValidAlignment(3));
  EXPECT_TRUE(IsValidAlignment(4));
}

constexpr unsigned kTestMaxAlignment = 64;
TEST(BefEncodingTest, CalculateAlignmentPaddingSizeWithPrefix) {
  for (unsigned prefix_size = 1; prefix_size <= kTestMaxAlignment;
       ++prefix_size) {
    for (unsigned alignment = 1; alignment <= kTestMaxAlignment;
         alignment *= 2) {
      for (size_t offset = 0; offset <= kTestMaxAlignment; ++offset) {
        const unsigned padding =
            CalculateAlignmentPaddingSize(offset, prefix_size, alignment);
        auto test_offset = offset + padding;
        test_offset += prefix_size;
        EXPECT_EQ(test_offset % alignment, 0);
      }
    }
  }
}

TEST(BefEncodingTest, CalculateAlignmentPaddingSizeForTwoAlignedPrefix) {
  for (unsigned prefix1 = 1; prefix1 <= kTestMaxAlignment; prefix1 *= 2) {
    for (unsigned prefix2 = 1; prefix2 <= kTestMaxAlignment; prefix2 *= 2) {
      // Requirement: prefix1 <= prefix2.
      if (prefix1 > prefix2) continue;
      for (unsigned alignment = 1; alignment <= kTestMaxAlignment;
           alignment *= 2) {
        for (size_t offset = 0; offset <= kTestMaxAlignment; ++offset) {
          const unsigned padding =
              (alignment >= prefix2)
                  ? CalculateAlignmentPaddingSize(offset, prefix1 + prefix2,
                                                  alignment)
                  : CalculateAlignmentPaddingSize(offset, prefix1, prefix2);
          auto test_offset = offset + padding;

          ASSERT_EQ(test_offset % prefix1, 0);
          test_offset += prefix1;

          ASSERT_EQ(test_offset % prefix2, 0);
          test_offset += prefix2;

          ASSERT_EQ(test_offset % alignment, 0);
        }
      }
    }
  }
}

}  // namespace
}  // namespace tfrt
