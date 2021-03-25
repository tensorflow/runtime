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

MATCHER_P(FunctionReturns, value, "") { return arg() == value; }

class MockTracingSink : TracingSink {
 public:
  MockTracingSink() { RegisterTracingSink(this); }

  MOCK_METHOD(Error, RequestTracing, (bool enable), (override));
  MOCK_METHOD(void, RecordTracingEvent, (TracingSink::NameGenerator),
              (override));
  MOCK_METHOD(void, PushTracingScope, (TracingSink::NameGenerator), (override));
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

  SetTracingLevel(TracingLevel::Default);

  EXPECT_CALL(sink, RecordTracingEvent(FunctionReturns("event0")))
      .Times(0);  // Should not call before tracing is enabled.
  RecordTracingEvent(TracingLevel::Default, [] { return "event0"; });

  EXPECT_CALL(sink, RequestTracing(true));
  RequestTracing(true);

  EXPECT_CALL(sink, RecordTracingEvent(FunctionReturns("event1")));
  RecordTracingEvent(TracingLevel::Default, [] { return "event1"; });

  EXPECT_CALL(sink, RecordTracingEvent(FunctionReturns("event2")))
      .Times(0);  // Should not call after tracing is disabled.
  RecordTracingEvent(TracingLevel::Debug, [] { return "event2"; });

  SetTracingLevel(TracingLevel::Debug);

  EXPECT_CALL(sink, RecordTracingEvent(FunctionReturns("event3")));
  RecordTracingEvent(TracingLevel::Debug, [] { return "event3"; });

  EXPECT_CALL(sink, RequestTracing(false));
  RequestTracing(false);

  EXPECT_CALL(sink, RecordTracingEvent(FunctionReturns("event4")))
      .Times(0);  // Should not call after tracing is disabled.
  RecordTracingEvent(TracingLevel::Debug, [] { return "event4"; });
}

TEST(TracingTest, Scopes) {
#ifdef TFRT_DISABLE_TRACING
  GTEST_SKIP() << "Tracing is disabled";
#endif
  InSequence seq;
  MockTracingSink sink;

  SetTracingLevel(TracingLevel::Default);

  EXPECT_CALL(sink, PushTracingScope(FunctionReturns("scope0")))
      .Times(0);  // Should not call before tracing is enabled.
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Default, [] { return "scope0"; });

  EXPECT_CALL(sink, RequestTracing(true));
  RequestTracing(true);

  EXPECT_CALL(sink, PushTracingScope(FunctionReturns("scope1")));
  EXPECT_CALL(sink, PopTracingScope());
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Default, [] { return "scope1"; });

  EXPECT_CALL(sink, PushTracingScope(FunctionReturns("scope2"))).Times(0);
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Debug, [] { return "scope2"; });

  SetTracingLevel(TracingLevel::Debug);

  EXPECT_CALL(sink, PushTracingScope(FunctionReturns("scope3")));
  EXPECT_CALL(sink, PopTracingScope());
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Debug, [] { return "scope3"; });

  EXPECT_CALL(sink, RequestTracing(false));
  RequestTracing(false);

  EXPECT_CALL(sink, PushTracingScope(FunctionReturns("scope4"))).Times(0);
  EXPECT_CALL(sink, PopTracingScope()).Times(0);
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Debug, [] { return "scope4"; });
}

}  // namespace
}  // namespace tracing
}  // namespace tfrt
