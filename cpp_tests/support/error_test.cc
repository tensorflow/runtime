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

//===- error_test.cc --------------------------------------------*- C++ -*-===//
//
// Tests related to error handling.
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/Error.h"

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
StackTrace CreateStackTrace0() { return CreateStackTrace(/*skip_count=*/1); }
StackTrace CreateStackTrace1() { return CreateStackTrace0(); }
StackTrace CreateStackTrace2() { return CreateStackTrace1(); }

namespace {
llvm::Error SuccessError() { return llvm::Error::success(); }
llvm::Error FailError() { return MakeStringError(); }
llvm::Expected<int> ValueExpected() { return 42; }
llvm::Expected<int> FailExpected() { return FailError(); }

TEST(Test, Macros) {
  EXPECT_TRUE(IsSuccess(SuccessError()));
  EXPECT_FALSE(IsSuccess(FailError()));

  // This works as well, but the above is clearer.
  EXPECT_FALSE(SuccessError());
  // No viable conversion from 'const llvm::Error' to 'bool':
  // EXPECT_TRUE(FailError());
  EXPECT_FALSE(!FailError());

  EXPECT_FALSE(IsSuccess([] {
    if (auto error = SuccessError()) return error;
    if (auto error = FailError()) return error;
    return SuccessError();
  }()));

  EXPECT_FALSE(IsSuccess([] {
    TFRT_ASSIGN_OR_RETURN(auto x, ValueExpected());
    TFRT_ASSIGN_OR_RETURN(auto y, FailExpected());
    (void)x;
    (void)y;
    return SuccessError();
  }()));

  TFRT_ASSERT_AND_ASSIGN(auto x, ValueExpected());
  (void)x;
}

static ::testing::AssertionResult Contains(llvm::StringRef string,
                                           llvm::StringRef substr) {
  if (string.contains(substr)) {
    return ::testing::AssertionSuccess()
           << llvm::formatv("'{0}' contains '{1}'", string, substr).str();
  }
  return ::testing::AssertionFailure()
         << llvm::formatv("'{1}' not found in '{0}'", string, substr).str();
}

TEST(Test, StackTrace) {
  auto stack_trace = CreateStackTrace2();
  if (!stack_trace) GTEST_SKIP() << "Stack traces unavailable";
  std::string buffer;
  llvm::raw_string_ostream(buffer) << stack_trace;
  // TODO(csigg): MSAN adds __interceptor_backtrace, breaking the check below.
  GTEST_SKIP() << "Fails in MSAN builds";
  EXPECT_FALSE(Contains(buffer, "tfrt::CreateStackTrace0()"));
  // TODO(csigg): figure out how to prevent functions from being inlined.
  GTEST_SKIP() << "Fails in optimized builds";
  EXPECT_TRUE(Contains(buffer, "tfrt::CreateStackTrace1()"));
  EXPECT_TRUE(Contains(buffer, "tfrt::CreateStackTrace2()"));
  // File and line info requires `--strip=never` or `-c dbg`.
  // In case that info is missing, we also don't print the " @ ".
  if (llvm::StringRef(buffer).contains(" @ "))
    EXPECT_TRUE(Contains(buffer, __FILE__));
}

void BM_CaptureStackTrace(benchmark::State& state) {
  bool fail = true;
  for (auto _ : state) {
    fail = fail && FailError();
  }
  EXPECT_TRUE(fail);
}
BENCHMARK(BM_CaptureStackTrace);

void BM_PrintStackTrace(benchmark::State& state) {
  auto error = FailError();
  llvm::raw_null_ostream os;
  for (auto _ : state) {
    os << error;
  }
}
BENCHMARK(BM_PrintStackTrace);

}  // namespace
}  // namespace tfrt
