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

// This file contains unit tests for TFRT AsyncValuePtr class.

#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace {

class AsyncValuePtrTest : public ::testing::Test {
 protected:
  AsyncValuePtrTest() { host_context_ = CreateHostContext(); }
  std::unique_ptr<HostContext> host_context_;
};

TEST_F(AsyncValuePtrTest, Construct) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  EXPECT_EQ(ptr.get(), 42);
}

TEST_F(AsyncValuePtrTest, Emplace) {
  AsyncValueRef<int32_t> ref = MakeUnconstructedAsyncValueRef<int32_t>();
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  EXPECT_FALSE(ptr.IsConcrete());
  EXPECT_FALSE(ptr.IsAvailable());

  ptr.emplace(42);
  EXPECT_EQ(ptr.get(), 42);
}

TEST_F(AsyncValuePtrTest, SetError) {
  AsyncValueRef<int32_t> ref = MakeUnconstructedAsyncValueRef<int32_t>();
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  EXPECT_FALSE(ptr.IsConcrete());
  EXPECT_FALSE(ptr.IsAvailable());

  ptr.SetError("test error");

  EXPECT_TRUE(ptr.IsAvailable());
  EXPECT_TRUE(ptr.IsError());
}

TEST_F(AsyncValuePtrTest, AndThen) {
  AsyncValueRef<int32_t> ref = MakeUnconstructedAsyncValueRef<int32_t>();
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  EXPECT_FALSE(ptr.IsConcrete());
  EXPECT_FALSE(ptr.IsAvailable());

  bool executed = false;
  ptr.AndThen([&]() { executed = true; });

  ptr.emplace(42);
  EXPECT_TRUE(executed);
}

}  // namespace
}  // namespace tfrt
