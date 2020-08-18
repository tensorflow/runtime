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

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#if __has_attribute(noinline)
#define NOINLINE __attribute__((noinline))
#define HAS_ATTRIBUTE_NOINLINE 1
#else
#define NOINLINE
#endif

namespace tfrt {

NOINLINE StackTrace CreateStackTrace0() {
  int skip_count = 1;  // Do not include this function in the stack trace.
#if __has_feature(address_sanitizer) || __has_feature(memory_sanitizer) || \
    __has_feature(thread_sanitizer)
  ++skip_count;  // Skip __interceptor_backtrace added by ASAN/MSAN/TSAN.
#endif
  auto ret = CreateStackTrace(skip_count);
  DoNotOptimize(ret.get());
  return ret;
}

NOINLINE StackTrace CreateStackTrace1() {
  auto ret = CreateStackTrace0();
  DoNotOptimize(ret.get());
  return ret;
}
NOINLINE StackTrace CreateStackTrace2() {
  auto ret = CreateStackTrace1();
  DoNotOptimize(ret);
  return ret;
}

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
  EXPECT_FALSE(Contains(buffer, "tfrt::CreateStackTrace0()"));
#ifndef HAS_ATTRIBUTE_NOINLINE
  GTEST_SKIP() << "Couldn't specify 'noinline' attribute.";
#endif
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
