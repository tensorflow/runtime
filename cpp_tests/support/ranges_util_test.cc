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

#include "tfrt/support/ranges_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace {

struct IntValue : ReferenceCounted<IntValue> {
  explicit IntValue(int v) : v{v} {}
  int v;
};

TEST(RangesUtilTest, CopyRefToVector) {
  auto int_refs = {MakeRef<IntValue>(1), MakeRef<IntValue>(2)};

  EXPECT_TRUE(llvm::equal(int_refs, CopyRefToSmallVector<3>(int_refs)));
  EXPECT_TRUE(llvm::equal(int_refs, CopyRefToVector(int_refs)));
}

}  // namespace
}  // namespace tfrt
