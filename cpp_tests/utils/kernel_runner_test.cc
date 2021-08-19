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

#include "tfrt/utils/kernel_runner.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(KernelRunnerTest, Basic) {
  auto sum =
      KernelRunner("tfrt_test.sum").SetArgs(1, 2, 3).RunAndGetResult<int>();

  EXPECT_EQ(sum, 6);
}

TEST(KernelRunnerTest, ArrayAttribute) {
  auto sum_i32 = KernelRunner("tfrt_test.sum_array_attr.i32")
                     .AddStringAttribute("11")
                     .AddArrayAttribute<int>({1, 2, 3})
                     .AddRequestContextData<int>(2)
                     .AddResource<int>("val", 4)
                     .RunAndGetResult<int>();

  EXPECT_EQ(sum_i32, 23);

  auto sum_i64 = KernelRunner("tfrt_test.sum_array_attr.i64")
                     .AddStringAttribute("10")
                     .AddArrayAttribute<int64_t>({1, 2, 3})
                     .AddRequestContextData<int64_t>(2)
                     .AddResource<int64_t>("val", 4)
                     .RunAndGetResult<int64_t>();

  EXPECT_EQ(sum_i64, 22);
}

TEST(KernelRunnerTest, RepeatedRuns) {
  {
    KernelRunner runner1("tfrt_test.sum");
    runner1.SetArgs(1, 2, 3);
    EXPECT_EQ(runner1.RunAndGetResult<int>(), 6);
    EXPECT_EQ(runner1.RunAndGetResult<int>(), 6);
    EXPECT_EQ(runner1.RunAndGetResult<int>(), 6);
  }

  {
    KernelRunner runner2{"tfrt_test.sum_array_attr.i64"};

    runner2.AddStringAttribute("10")
        .AddArrayAttribute<int64_t>({1, 2, 3})
        .AddResource<int64_t>("val", 4)
        .AddRequestContextData<int64_t>(2);

    EXPECT_EQ(runner2.RunAndGetResult<int64_t>(), 22);
    EXPECT_EQ(runner2.RunAndGetResult<int64_t>(), 22);
    EXPECT_EQ(runner2.RunAndGetResult<int64_t>(), 22);
  }
}

TEST(KernelRunnerTest, StringAttribute) {
  auto str = KernelRunner("tfrt_test.get_string")
                 .AddStringAttribute("hello")
                 .RunAndGetResult<std::string>();

  EXPECT_EQ(str, "hello");
}

TEST(KernelRunnerTest, AsyncKernel) {
  auto sum = KernelRunner("tfrt_test.async_add.i32")
                 .SetArgs(1, 2)
                 .RunAndGetResult<int>();

  EXPECT_EQ(sum, 3);
}

TEST(KernelRunnerTest, RequestContext) {
  KernelRunner runner("tfrt_test.get_request_context_data.i32");
  runner.AddRequestContextData<int>(3);
  auto value = runner.RunAndGetResult<int>();

  EXPECT_EQ(value, 3);
}

}  // namespace
}  // namespace tfrt
