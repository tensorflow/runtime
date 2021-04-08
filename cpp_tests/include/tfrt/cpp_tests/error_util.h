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

#include "gtest/gtest.h"
#include "llvm/Support/raw_os_ostream.h"
#include "tfrt/support/error_util.h"

// Helper macro to get value from llvm::Expected.
//
// The result of 'expr' should be a llvm::Expected<T>. If it has a value, it
// is assigned to 'lhs'. Otherwise the test fails fatally.
#define TFRT_ASSERT_AND_ASSIGN(lhs, expr) \
  TFRT_ASSERT_AND_ASSIGN_IMPL(TFRT_CONCAT(_expected_, __COUNTER__), lhs, expr)
#define TFRT_ASSERT_AND_ASSIGN_IMPL(expected, lhs, expr) \
  auto expected = expr;                                  \
  ASSERT_FALSE(!expected) << expected.takeError();       \
  lhs = std::move(*expected)

// Same as above, but uses TFRT_LOG(FATAL) instead of ASSERT_FALSE.
// The macro above should be preferred. This can be used in constructors, or
// functions that return a value. ASSERT_FALSE expands to a return statement
// on error.
#define TFRT_ASSIGN_OR_DIE(lhs, expr) \
  TFRT_ASSIGN_OR_DIE_IMPL(TFRT_CONCAT(_expected_, __COUNTER__), lhs, expr)
#define TFRT_ASSIGN_OR_DIE_IMPL(expected, lhs, expr)      \
  auto expected = expr;                                   \
  if (!expected) TFRT_LOG(FATAL) << expected.takeError(); \
  lhs = std::move(*expected)

namespace llvm {

// Google Test outputs to std::ostream. Provide ADL'able overload.
inline std::ostream& operator<<(std::ostream& os, const Error& error) {
  raw_os_ostream(os) << error;
  return os;
}

// Make llvm::Error testable with EXPECT_TRUE and ASSERT_TRUE.
//
// Usage: EXPECT_TRUE(IsSuccess(SomeFunctionReturningLlvmError()));
inline ::testing::AssertionResult IsSuccess(Error&& error) {
  if (error) return ::testing::AssertionFailure() << error;
  return ::testing::AssertionSuccess();
}

// Check that logging 'error' produces 'message'.
inline ::testing::AssertionResult IsErrorString(Error&& error,
                                                StringRef message) {
  std::string buffer;
  raw_string_ostream(buffer) << error;
  if (buffer != message) return ::testing::AssertionFailure() << buffer;
  return ::testing::AssertionSuccess();
}

}  // namespace llvm

#endif  // TFRT_CPP_TESTS_ERROR_UTIL_H_
