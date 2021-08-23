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

// This file contains unit tests for TFRT AsyncValueRef class.

#include "tfrt/host_context/async_value_ref.h"

#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace {

class WrappedInt32 {
 public:
  explicit WrappedInt32(int32_t value) : value_(value) {}
  int32_t value() const { return value_; }

 private:
  int32_t value_;
};

constexpr int32_t kTestValue = 42;

class AsyncValueRefTest : public ::testing::Test {
 protected:
  AsyncValueRefTest() { host_context_ = CreateHostContext(); }
  std::unique_ptr<HostContext> host_context_;
};

TEST_F(AsyncValueRefTest, ValueCheck) {
  auto wrapped_int_value =
      MakeAvailableAsyncValueRef<WrappedInt32>(host_context_.get(), kTestValue);
  EXPECT_EQ(wrapped_int_value.get().value(), kTestValue);
  EXPECT_EQ(wrapped_int_value->value(), kTestValue);
  EXPECT_EQ((*wrapped_int_value).value(), kTestValue);
}

TEST_F(AsyncValueRefTest, ValueCheckFromRCReference) {
  auto wrapped_int_value =
      MakeAvailableAsyncValueRef<WrappedInt32>(host_context_.get(), kTestValue);
  RCReference<AsyncValue> generic_value = std::move(wrapped_int_value);
  EXPECT_EQ(generic_value->get<WrappedInt32>().value(), kTestValue);
}

TEST_F(AsyncValueRefTest, ValueCheckFromAliasedRCReference) {
  auto wrapped_int_value =
      MakeAvailableAsyncValueRef<WrappedInt32>(host_context_.get(), kTestValue);
  RCReference<AsyncValue> generic_value = std::move(wrapped_int_value);
  AsyncValueRef<WrappedInt32> aliased_int_value(std::move(generic_value));
  EXPECT_EQ(aliased_int_value.get().value(), kTestValue);
  EXPECT_EQ(aliased_int_value->value(), kTestValue);
  EXPECT_EQ((*aliased_int_value).value(), kTestValue);
}

TEST_F(AsyncValueRefTest, ConstructedToError) {
  auto value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), kTestValue);

  EXPECT_FALSE(value.IsConcrete());
  EXPECT_FALSE(value.IsAvailable());

  value.AndThen([] {});
  value.SetError(DecodedDiagnostic("test error"));

  EXPECT_TRUE(value.IsAvailable());
  EXPECT_FALSE(value.IsConcrete());
  EXPECT_TRUE(value.IsError());
}

TEST_F(AsyncValueRefTest, ConstructedToConcrete) {
  auto value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), kTestValue);

  EXPECT_FALSE(value.IsConcrete());
  EXPECT_FALSE(value.IsAvailable());

  value.AndThen([] {});
  value.SetStateConcrete();

  EXPECT_TRUE(value.IsAvailable());
  EXPECT_TRUE(value.IsConcrete());
  EXPECT_FALSE(value.IsError());

  EXPECT_EQ(kTestValue, value.get());
}

TEST_F(AsyncValueRefTest, UnconstructedEmplace) {
  auto value = MakeUnconstructedAsyncValueRef<int32_t>(host_context_.get());

  EXPECT_FALSE(value.IsConcrete());
  EXPECT_FALSE(value.IsAvailable());

  value.AndThen([] {});

  value.emplace(kTestValue);
  EXPECT_TRUE(value.IsAvailable());
  EXPECT_TRUE(value.IsConcrete());

  EXPECT_EQ(kTestValue, value.get());
}

TEST_F(AsyncValueRefTest, CopyRef) {
  auto value =
      MakeAvailableAsyncValueRef<int32_t>(host_context_.get(), kTestValue);

  EXPECT_TRUE(value.IsConcrete());

  EXPECT_TRUE(value.IsUnique());
  auto copied_value = value.CopyRef();
  EXPECT_FALSE(value.IsUnique());

  EXPECT_EQ(value.GetAsyncValue(), copied_value.GetAsyncValue());
}

TEST_F(AsyncValueRefTest, AndThenError) {
  auto value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), kTestValue);

  DecodedDiagnostic diag{"test_error"};
  value.AndThen([&](Error error) { EXPECT_EQ(StrCat(error), StrCat(diag)); });

  value.SetError(diag);
}

TEST_F(AsyncValueRefTest, AndThenNoError) {
  auto value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), kTestValue);

  value.AndThen([](Error error) { EXPECT_FALSE(!!error); });

  value.SetStateConcrete();
}

TEST_F(AsyncValueRefTest, AndThenExpectedError) {
  auto value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), kTestValue);

  DecodedDiagnostic diag{"test_error"};
  value.AndThen([&](Expected<int32_t*> v) {
    EXPECT_FALSE(!!v);
    EXPECT_EQ(StrCat(v.takeError()), StrCat(diag));
  });

  value.SetError(diag);
}

TEST_F(AsyncValueRefTest, PtrAndThenExpectedError) {
  auto value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), kTestValue);

  DecodedDiagnostic diag{"test_error"};
  value.AsPtr().AndThen([&](Expected<int32_t*> v) {
    EXPECT_FALSE(!!v);
    EXPECT_EQ(StrCat(v.takeError()), StrCat(diag));
  });

  value.SetError(diag);
}

TEST_F(AsyncValueRefTest, AndThenExpectedNoError) {
  auto value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), kTestValue);

  value.AndThen([](Expected<int32_t*> v) {
    EXPECT_TRUE(!!v);
    EXPECT_EQ(**v, kTestValue);
  });

  value.SetStateConcrete();
}

TEST_F(AsyncValueRefTest, PtrAndThenExpectedNoError) {
  auto value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), kTestValue);

  value.AsPtr().AndThen([](Expected<int32_t*> v) {
    EXPECT_TRUE(!!v);
    EXPECT_EQ(**v, kTestValue);
  });

  value.SetStateConcrete();
}

TEST_F(AsyncValueRefTest, AsExpectedError) {
  auto value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), kTestValue);

  DecodedDiagnostic diag{"test_error"};
  value.SetError(diag);
  Expected<int32_t*> expected = value.AsExpected();
  EXPECT_FALSE(!!expected);
  EXPECT_EQ(StrCat(expected.takeError()), StrCat(diag));
}

TEST_F(AsyncValueRefTest, AsExpectedNoError) {
  auto value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), kTestValue);
  value.SetStateConcrete();

  Expected<int32_t*> expected = value.AsExpected();
  EXPECT_TRUE(!!expected);
  EXPECT_EQ(**expected, kTestValue);
}

}  // namespace
}  // namespace tfrt
