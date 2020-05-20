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

//===- tracing_benchmark.cc -------------------------------------*- C++ -*-===//
//
// Benchmark measuring tracing overhead.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tracing/tracing.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/Error.h"

namespace tfrt {
namespace tracing {
namespace {

using ::testing::DefaultValue;
using ::testing::InSequence;
using ::testing::IsNull;
using ::testing::Matcher;
using ::testing::StrEq;
using ::testing::TypedEq;

class MockTracingSink : TracingSink {
 public:
  MockTracingSink() { RegisterTracingSink(this); }

  MOCK_METHOD(Error, RequestTracing, (bool enable), (override));
  MOCK_METHOD(void, RecordTracingEvent, (const char*, string_view), (override));
  MOCK_METHOD(void, RecordTracingEvent, (const char*, const char*), (override));
  MOCK_METHOD(void, RecordTracingEvent, (const char*, std::string&&),
              (override));

  MOCK_METHOD(void, PushTracingScope, (const char*, string_view), (override));
  MOCK_METHOD(void, PushTracingScope, (const char*, const char*), (override));
  MOCK_METHOD(void, PushTracingScope, (const char*, std::string&&), (override));
  MOCK_METHOD(void, PopTracingScope, (), (override));
};

auto kSetDefaultErrorFactory = [] {
  return DefaultValue<llvm::Error>::SetFactory(
             []() -> llvm::Error { return llvm::Error::success(); }),
         0;
}();

TEST(TracingTest, Request) {
  MockTracingSink sink;
  EXPECT_FALSE(IsTracingEnabled());
  RequestTracing(false);
  EXPECT_FALSE(IsTracingEnabled());
#ifdef TFRT_DISABLE_TRACING
  GTEST_SKIP() << "Tracing is disabled";
#endif
  EXPECT_CALL(sink, RequestTracing(true));
  RequestTracing(true);
  EXPECT_TRUE(IsTracingEnabled());
  EXPECT_CALL(sink, RequestTracing(false));
  RequestTracing(false);
  EXPECT_FALSE(IsTracingEnabled());
}

TEST(TracingTest, Events) {
#ifdef TFRT_DISABLE_TRACING
  GTEST_SKIP() << "Tracing is disabled";
#endif
  InSequence seq;
  MockTracingSink sink;

  EXPECT_CALL(sink,
              RecordTracingEvent(IsNull(), TypedEq<const char*>("event0")))
      .Times(0);  // Should not call before tracing is enabled.
  RecordTracingEvent("event0");

  EXPECT_CALL(sink, RequestTracing(true));
  RequestTracing(true);

  EXPECT_CALL(sink,
              RecordTracingEvent(IsNull(), TypedEq<const char*>("event1")));
  RecordTracingEvent("event1");

  EXPECT_CALL(sink,
              RecordTracingEvent(IsNull(), TypedEq<string_view>("event2")));
  RecordTracingEvent(string_view("event2"));

  // TypedEq<std::string&&> doesn't compile before gtest v1.10.
  EXPECT_CALL(sink, RecordTracingEvent(
                        IsNull(), Matcher<std::string&&>(StrEq("event3"))));
  RecordTracingEvent(std::string("event3"));

  EXPECT_CALL(sink, RecordTracingEvent(TypedEq<const char*>("category"),
                                       TypedEq<const char*>("event4")));
  RecordTracingEvent("category", "event4");

  EXPECT_CALL(sink, RequestTracing(false));
  RequestTracing(false);

  EXPECT_CALL(sink,
              RecordTracingEvent(IsNull(), TypedEq<const char*>("event5")))
      .Times(0);  // Should not call after tracing is disabled.
  RecordTracingEvent("event5");
}

TEST(TracingTest, Scopes) {
#ifdef TFRT_DISABLE_TRACING
  GTEST_SKIP() << "Tracing is disabled";
#endif
  InSequence seq;
  MockTracingSink sink;

  EXPECT_CALL(sink, PushTracingScope(IsNull(), TypedEq<const char*>("scope0")))
      .Times(0);           // Should not call before tracing is enabled.
  TracingScope("scope0");  // NOLINT(bugprone-unused-raii)

  EXPECT_CALL(sink, RequestTracing(true));
  RequestTracing(true);

  EXPECT_CALL(sink, PushTracingScope(IsNull(), TypedEq<const char*>("scope1")));
  EXPECT_CALL(sink, PopTracingScope());
  TracingScope("scope1");  // NOLINT(bugprone-unused-raii)

  EXPECT_CALL(sink, PushTracingScope(IsNull(), TypedEq<string_view>("scope2")));
  EXPECT_CALL(sink, PopTracingScope());
  TracingScope(string_view("scope2"));  // NOLINT(bugprone-unused-raii)

  // TypedEq<std::string&&> doesn't compile before gtest v1.10.
  EXPECT_CALL(sink, PushTracingScope(IsNull(),
                                     Matcher<std::string&&>(StrEq("scope3"))));
  EXPECT_CALL(sink, PopTracingScope());
  TracingScope(std::string("scope3"));  // NOLINT(bugprone-unused-raii)

  EXPECT_CALL(sink, PushTracingScope(TypedEq<const char*>("category"),
                                     TypedEq<const char*>("scope4")));
  EXPECT_CALL(sink, PopTracingScope());
  TracingScope("category", "scope4");  // NOLINT(bugprone-unused-raii)

  EXPECT_CALL(sink, RequestTracing(false));
  RequestTracing(false);

  EXPECT_CALL(sink, PushTracingScope(IsNull(), TypedEq<const char*>("scope5")))
      .Times(0);           // Should not call after tracing is disabled.
  TracingScope("scope5");  // NOLINT(bugprone-unused-raii)
}

}  // namespace
}  // namespace tracing
}  // namespace tfrt
