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

//===- async_value_ref_test.cc --------------------------------------------===//
//
// This file contains unit tests for tfrt::AsyncValueRef.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/async_value.h"

#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace {

TEST(AsyncValueRef, Conversions) {
  std::unique_ptr<HostContext> host = CreateHostContext();

  class WrappedInt32 {
   public:
    explicit WrappedInt32(int32_t value) : value_(value) {}
    int32_t value() const { return value_; }

   private:
    int32_t value_;
  };

  AsyncValueRef<WrappedInt32> wrapped_int_value =
      MakeAvailableAsyncValueRef<WrappedInt32>(host.get(), 42);
  EXPECT_EQ(wrapped_int_value.get().value(), 42);
  EXPECT_EQ(wrapped_int_value->value(), 42);
  EXPECT_EQ((*wrapped_int_value).value(), 42);

  RCReference<AsyncValue> generic_value = std::move(wrapped_int_value);
  EXPECT_EQ(generic_value->get<WrappedInt32>().value(), 42);

  AsyncValueRef<WrappedInt32> aliased_int_value(std::move(generic_value));
  EXPECT_EQ(aliased_int_value.get().value(), 42);
  EXPECT_EQ(aliased_int_value->value(), 42);
  EXPECT_EQ((*aliased_int_value).value(), 42);
}

TEST(AsyncValue, ConstructedToError) {
  std::unique_ptr<HostContext> host = CreateHostContext();
  AsyncValue* value =
      MakeConstructedAsyncValueRef<int32_t>(host.get(), 0).release();
  value->AndThen([] {});
  value->SetError(DecodedDiagnostic("test error"));
  value->DropRef();
}

}  // namespace
}  // namespace tfrt
