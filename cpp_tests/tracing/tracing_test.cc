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
  MOCK_METHOD(void, RecordTracingEvent, (string_view), (override));
  MOCK_METHOD(void, RecordTracingEvent, (const char*), (override));
  MOCK_METHOD(void, RecordTracingEvent, (std::string &&), (override));

  MOCK_METHOD(void, PushTracingScope, (string_view), (override));
  MOCK_METHOD(void, PushTracingScope, (const char*), (override));
  MOCK_METHOD(void, PushTracingScope, (std::string &&), (override));
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
              RecordTracingEvent(TypedEq<const char*>("event0")))
      .Times(0);  // Should not call before tracing is enabled.
  RecordTracingEvent(TracingLevel::Default, "event0");

  EXPECT_CALL(sink, RequestTracing(true));
  RequestTracing(true);

  EXPECT_CALL(sink, RecordTracingEvent(TypedEq<const char*>("event1")));
  RecordTracingEvent(TracingLevel::Default, "event1");

  EXPECT_CALL(sink, RecordTracingEvent(TypedEq<string_view>("event2")));
  RecordTracingEvent(TracingLevel::Default, string_view("event2"));

  // TypedEq<std::string&&> doesn't compile before gtest v1.10.
  EXPECT_CALL(sink,
              RecordTracingEvent(Matcher<std::string&&>(StrEq("event3"))));
  RecordTracingEvent(TracingLevel::Default, std::string("event3"));

  SetTracingLevel(TracingLevel::Verbose);

  EXPECT_CALL(sink,
              RecordTracingEvent(TypedEq<const char*>("event4")))
      .Times(0);  // Should not call after tracing is disabled.
  RecordTracingEvent(TracingLevel::Debug, "event4");

  EXPECT_CALL(sink, RecordTracingEvent(TypedEq<const char*>("event5")));
  RecordTracingEvent(TracingLevel::Verbose, "event5");

  EXPECT_CALL(sink, RecordTracingEvent(TypedEq<const char*>("event6")));
  RecordTracingEvent(TracingLevel::Default, "event6");

  EXPECT_CALL(sink, RequestTracing(false));
  RequestTracing(false);

  EXPECT_CALL(sink,
              RecordTracingEvent(TypedEq<const char*>("event7")))
      .Times(0);  // Should not call after tracing is disabled.
  RecordTracingEvent(TracingLevel::Default, "event7");
}

TEST(TracingTest, Scopes) {
#ifdef TFRT_DISABLE_TRACING
  GTEST_SKIP() << "Tracing is disabled";
#endif
  InSequence seq;
  MockTracingSink sink;

  EXPECT_CALL(sink, PushTracingScope(TypedEq<const char*>("scope0")))
      .Times(0);  // Should not call before tracing is enabled.
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Default, "scope0");

  EXPECT_CALL(sink, RequestTracing(true));
  RequestTracing(true);

  EXPECT_CALL(sink, PushTracingScope(TypedEq<const char*>("scope1")));
  EXPECT_CALL(sink, PopTracingScope());
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Default, "scope1");

  EXPECT_CALL(sink, PushTracingScope(TypedEq<string_view>("scope2")));
  EXPECT_CALL(sink, PopTracingScope());
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Default, string_view("scope2"));

  // TypedEq<std::string&&> doesn't compile before gtest v1.10.
  EXPECT_CALL(sink, PushTracingScope(Matcher<std::string&&>(StrEq("scope3"))));
  EXPECT_CALL(sink, PopTracingScope());
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Default, std::string("scope3"));

  SetTracingLevel(TracingLevel::Verbose);

  EXPECT_CALL(sink, PushTracingScope(TypedEq<const char*>("scope4"))).Times(0);
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Debug, "scope4");

  EXPECT_CALL(sink, PushTracingScope(TypedEq<const char*>("scope5")));
  EXPECT_CALL(sink, PopTracingScope());
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Verbose, "scope5");

  EXPECT_CALL(sink, PushTracingScope(TypedEq<const char*>("scope6")));
  EXPECT_CALL(sink, PopTracingScope());
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Default, "scope6");

  EXPECT_CALL(sink, RequestTracing(false));
  RequestTracing(false);

  EXPECT_CALL(sink, PushTracingScope(TypedEq<const char*>("scope7")))
      .Times(0);  // Should not call after tracing is disabled.
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Default, "scope7");
}

}  // namespace
}  // namespace tracing
}  // namespace tfrt
