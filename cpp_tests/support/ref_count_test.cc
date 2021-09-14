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

// Unit test for ReferenceCounted and RCReference classes

#include "tfrt/support/ref_count.h"

#include "gtest/gtest.h"

namespace tfrt {
namespace {

class WrappedInt32 : public ReferenceCounted<WrappedInt32> {
 public:
  explicit WrappedInt32(int32_t value) : value_(value) {}
  ~WrappedInt32() {}

  int32_t value() const { return value_; }

 private:
  int32_t value_;
};

TEST(RefCountTest, ReferenceCountedIsUnique) {
  WrappedInt32* wi = new WrappedInt32(123);
  EXPECT_TRUE(wi->IsUnique());
  wi->DropRef();
}

TEST(RefCountTest, ReferenceCountedIsNotUnique) {
  WrappedInt32* wi = new WrappedInt32(123);
  wi->AddRef();
  EXPECT_FALSE(wi->IsUnique());
  wi->DropRef();
  wi->DropRef();
}

TEST(RefCountTest, RCReferenceBasic) {
  RCReference<WrappedInt32> rwi = MakeRef<WrappedInt32>(123);
  EXPECT_EQ(123, rwi->value());
  EXPECT_TRUE(rwi->IsUnique());
}

TEST(RefCountTest, RCReferenceReset) {
  RCReference<WrappedInt32> rwi = MakeRef<WrappedInt32>(123);
  rwi.reset();
  EXPECT_FALSE(rwi);
}

TEST(RefCountTest, RCReferenceRelease) {
  RCReference<WrappedInt32> rwi = MakeRef<WrappedInt32>(123);
  WrappedInt32* wi = rwi.release();
  EXPECT_FALSE(rwi);
  EXPECT_EQ(123, wi->value());
  wi->DropRef();
}

TEST(RefCountTest, RCReferenceMove) {
  RCReference<WrappedInt32> rwi = MakeRef<WrappedInt32>(123);
  RCReference<WrappedInt32> rwi_moved(std::move(rwi));
  EXPECT_TRUE(rwi_moved->IsUnique());
  EXPECT_EQ(123, rwi_moved->value());
}

TEST(RefCountTest, RCReferenceCopy) {
  RCReference<WrappedInt32> rwi = MakeRef<WrappedInt32>(123);
  RCReference<WrappedInt32> rwi_copied = rwi;

  EXPECT_FALSE(rwi->IsUnique());
  EXPECT_FALSE(rwi_copied->IsUnique());

  EXPECT_EQ(123, rwi->value());
  EXPECT_EQ(123, rwi_copied->value());

  RCReference<WrappedInt32> rwi_other = MakeRef<WrappedInt32>(456);
  rwi = rwi_other;

  EXPECT_FALSE(rwi->IsUnique());
  EXPECT_TRUE(rwi_copied->IsUnique());

  EXPECT_EQ(456, rwi->value());
  EXPECT_EQ(123, rwi_copied->value());
}

TEST(RefCountTest, RCReferenceCopyRef) {
  RCReference<WrappedInt32> rwi = MakeRef<WrappedInt32>(123);
  RCReference<WrappedInt32> rwi_copied = rwi.CopyRef();

  EXPECT_FALSE(rwi->IsUnique());
  EXPECT_FALSE(rwi_copied->IsUnique());

  EXPECT_EQ(123, rwi->value());
  EXPECT_EQ(123, rwi_copied->value());
}

TEST(RefCountTest, RCReferenceSwap) {
  RCReference<WrappedInt32> rwi_a = MakeRef<WrappedInt32>(123);
  RCReference<WrappedInt32> rwi_b = MakeRef<WrappedInt32>(456);

  rwi_a.swap(rwi_b);

  EXPECT_EQ(456, rwi_a->value());
  EXPECT_EQ(123, rwi_b->value());
}

TEST(RefCountTest, RCReferenceGlobalSwap) {
  RCReference<WrappedInt32> rwi_a = MakeRef<WrappedInt32>(123);
  RCReference<WrappedInt32> rwi_b = MakeRef<WrappedInt32>(456);

  swap(rwi_a, rwi_b);

  EXPECT_EQ(456, rwi_a->value());
  EXPECT_EQ(123, rwi_b->value());
}

}  // namespace
}  // namespace tfrt
