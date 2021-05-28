/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file defines utilities related to error handling in tests.
#ifndef TFRT_CPP_TESTS_ERROR_UTIL_H_
#define TFRT_CPP_TESTS_ERROR_UTIL_H_

#include <functional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/raw_os_ostream.h"
#include "tfrt/support/error_util.h"

// Helper macro to get value from llvm::Expected.
//
// The result of 'expr' should be a llvm::Expected<T>. If it has a value, it
// is assigned to 'lhs'. Otherwise the test fails fatally.
#define TFRT_ASSERT_AND_ASSIGN(lhs, expr) \
  TFRT_ASSERT_AND_ASSIGN_IMPL(TFRT_CONCAT(_expected_, __COUNTER__), lhs, expr)
#define TFRT_ASSERT_AND_ASSIGN_IMPL(expected, lhs, expr)  \
  auto expected = expr;                                   \
  ASSERT_THAT(expected.takeError(), ::tfrt::IsSuccess()); \
  lhs = std::move(*expected)

namespace llvm {
// Google Test outputs to std::ostream. Provide ADL'able overload.
inline std::ostream& operator<<(std::ostream& os, const Error& error) {
  raw_os_ostream(os) << error;
  return os;
}
}  // namespace llvm

namespace tfrt {
// Not a GMock matcher, see MakePredicateFormatterFromMatcher() comment.
enum class ErrorMatcher {
  expect_success,
  expect_failure,
};
inline ErrorMatcher IsFailure() { return ErrorMatcher::expect_failure; }
inline ErrorMatcher IsSuccess() { return ErrorMatcher::expect_success; }

class ErrorPredicateFormatter {
 public:
  explicit ErrorPredicateFormatter(ErrorMatcher matcher)
      : expect_failure_(matcher == IsFailure()) {}

  ::testing::AssertionResult operator()(const char* value_text,
                                        llvm::Error error) const {
    bool is_failure = static_cast<bool>(error);

    if (expect_failure_ == is_failure) {
      consumeError(std::move(error));
      return ::testing::AssertionSuccess();
    }

    std::string message;
    llvm::raw_string_ostream(message)
        << "Value of: " << value_text << "\n"
        << "Expected: " << (expect_failure_ ? "failure" : "success") << "\n"
        << "  Actual: " << toString(std::move(error));

    return ::testing::AssertionFailure() << message;
  }

 private:
  bool expect_failure_;
};
}  // namespace tfrt

namespace testing {
namespace internal {
// Overload GMock's (internal) predicate formatter factory.
//
// This is necessary because GMock does not support move-only types. The
// alternative would be to explicitly convert (verbosely, at every call site)
// the llvm::Error to a copyable result and message, and predicate on that.
inline tfrt::ErrorPredicateFormatter MakePredicateFormatterFromMatcher(
    tfrt::ErrorMatcher matcher) {
  return tfrt::ErrorPredicateFormatter(matcher);
}
}  // namespace internal
}  // namespace testing

#endif  // TFRT_CPP_TESTS_ERROR_UTIL_H_
