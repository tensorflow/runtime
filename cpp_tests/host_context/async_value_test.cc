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

// This file contains unit tests for TFRT AsyncValue class.

#include "tfrt/host_context/async_value.h"

#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace {

class AsyncValueTest : public ::testing::Test {
 protected:
  AsyncValueTest() { host_context_ = CreateHostContext(); }

  std::unique_ptr<HostContext> host_context_;
};

TEST_F(AsyncValueTest, ConstructedToError) {
  AsyncValue* value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), 123).release();
  bool callback_triggered = false;

  EXPECT_TRUE(value->IsConstructed());
  EXPECT_FALSE(value->IsConcrete());
  EXPECT_FALSE(value->IsAvailable());

  value->AndThen([&] { callback_triggered = true; });
  EXPECT_FALSE(callback_triggered);
  value->SetError(DecodedDiagnostic("test error"));
  EXPECT_TRUE(callback_triggered);

  EXPECT_TRUE(value->IsAvailable());
  EXPECT_FALSE(value->IsConcrete());
  EXPECT_TRUE(value->IsError());
  value->DropRef();
}

TEST_F(AsyncValueTest, ConstructedToConcrete) {
  AsyncValue* value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), 123).release();

  EXPECT_TRUE(value->IsConstructed());
  EXPECT_FALSE(value->IsConcrete());
  EXPECT_FALSE(value->IsAvailable());

  value->AndThen([] {});
  value->SetStateConcrete();

  EXPECT_TRUE(value->IsAvailable());
  EXPECT_TRUE(value->IsConcrete());
  EXPECT_FALSE(value->IsError());

  EXPECT_EQ(123, value->get<int32_t>());
  value->DropRef();
}

TEST_F(AsyncValueTest, UnconstructedEmplace) {
  AsyncValue* value =
      MakeUnconstructedAsyncValueRef<int32_t>(host_context_.get()).release();

  EXPECT_FALSE(value->IsConstructed());
  EXPECT_FALSE(value->IsConcrete());
  EXPECT_FALSE(value->IsAvailable());

  value->AndThen([] {});

  value->emplace<int32_t>(123);
  EXPECT_FALSE(value->IsConstructed());
  EXPECT_TRUE(value->IsAvailable());
  EXPECT_TRUE(value->IsConcrete());

  EXPECT_EQ(123, value->get<int32_t>());

  value->DropRef();
}

TEST_F(AsyncValueTest, AddAndDropRef) {
  AsyncValue* value =
      MakeConstructedAsyncValueRef<int32_t>(host_context_.get(), 123).release();

  value->AndThen([] {});
  value->SetStateConcrete();

  EXPECT_TRUE(value->IsConcrete());

  EXPECT_TRUE(value->IsUnique());
  value->AddRef();
  EXPECT_FALSE(value->IsUnique());

  EXPECT_EQ(123, value->get<int32_t>());
  value->DropRef();
  value->DropRef();
}

}  // namespace
}  // namespace tfrt
