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

// Benchmark measuring tracing overhead.

#include "tfrt/tracing/tracing.h"

#include <limits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace tracing {
namespace {

using ::testing::ByMove;
using ::testing::DefaultValue;
using ::testing::InSequence;
using ::testing::Matcher;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

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

#define TFRT_SKIP_IF(cond) \
  if (cond) GTEST_SKIP() << #cond;

TEST(TracingTest, Request) {
  TFRT_SKIP_IF(internal::kMaxTracingLevel < TracingLevel::Default)

  InSequence seq;
  StrictMock<MockTracingSink> sink;
  EXPECT_FALSE(IsTracingEnabled(TracingLevel::Default));

  // Sink returns error, tracing should not get enabled.
  EXPECT_CALL(sink, RequestTracing(true))
      .WillOnce(Return(ByMove(MakeStringError("fail"))));
  RequestTracing(true);
  EXPECT_FALSE(IsTracingEnabled(TracingLevel::Default));

  // Sink return success, tracing should get enabled.
  EXPECT_CALL(sink, RequestTracing(true)).Times(1);
  RequestTracing(true);
  EXPECT_TRUE(IsTracingEnabled(TracingLevel::Default));

  // Check that re-enabling is handled with counter without calling sink.
  RequestTracing(true);
  RequestTracing(false);

  // Sink returns error, tracing should not get disabled.
  EXPECT_CALL(sink, RequestTracing(false))
      .WillOnce(Return(ByMove(MakeStringError("fail"))));
  RequestTracing(false);
  EXPECT_TRUE(IsTracingEnabled(TracingLevel::Default));

  // Sink return success, tracing should get disabled.
  EXPECT_CALL(sink, RequestTracing(false)).Times(1);
  RequestTracing(false);
  EXPECT_FALSE(IsTracingEnabled(TracingLevel::Default));
}

// This test should pass independent of the value of TFRT_MAX_TRACING_LEVEL.
TEST(TracingTest, Levels) {
  NiceMock<MockTracingSink> sink;
  for (auto level : {TracingLevel::None, TracingLevel::Default,
                     TracingLevel::Debug, TracingLevel::Verbose}) {
    SetTracingLevel(level);
    RequestTracing(true);
    EXPECT_EQ(GetCurrentTracingLevel(),
              std::min(level, internal::kMaxTracingLevel));
    EXPECT_EQ(IsTracingEnabled(level), level <= internal::kMaxTracingLevel);
    RequestTracing(false);
    EXPECT_EQ(GetCurrentTracingLevel(), TracingLevel::None);
    EXPECT_EQ(IsTracingEnabled(level), level == TracingLevel::None);
  }

  SetTracingLevel(internal::kMaxTracingLevel);  // Reset level.
}

TEST(TracingTest, Events) {
  TFRT_SKIP_IF(internal::kMaxTracingLevel < TracingLevel::Default);
  TFRT_SKIP_IF(internal::kMaxTracingLevel >= TracingLevel::Debug);

  InSequence seq;
  NiceMock<MockTracingSink> sink;

  // Should not call before tracing is enabled.
  EXPECT_CALL(sink, RecordTracingEvent(FunctionReturns("event0"))).Times(0);
  RecordTracingEvent(TracingLevel::Default, [] { return "event0"; });

  RequestTracing(true);

  EXPECT_CALL(sink, RecordTracingEvent(FunctionReturns("event1"))).Times(1);
  RecordTracingEvent(TracingLevel::Default, [] { return "event1"; });
  EXPECT_CALL(sink, RecordTracingEvent(FunctionReturns("event2"))).Times(0);
  RecordTracingEvent(TracingLevel::Debug, [] { return "event2"; });

  RequestTracing(false);

  // Should not call after tracing is disabled.
  EXPECT_CALL(sink, RecordTracingEvent(FunctionReturns("event3"))).Times(0);
  RecordTracingEvent(TracingLevel::Default, [] { return "event3"; });
}

TEST(TracingTest, Scopes) {
  TFRT_SKIP_IF(internal::kMaxTracingLevel < TracingLevel::Default);
  TFRT_SKIP_IF(internal::kMaxTracingLevel >= TracingLevel::Debug);

  InSequence seq;
  NiceMock<MockTracingSink> sink;

  // Should not call before tracing is enabled.
  EXPECT_CALL(sink, PushTracingScope(FunctionReturns("scope0"))).Times(0);
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Default, [] { return "scope0"; });

  RequestTracing(true);

  EXPECT_CALL(sink, PushTracingScope(FunctionReturns("scope1"))).Times(1);
  EXPECT_CALL(sink, PopTracingScope()).Times(1);
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Default, [] { return "scope1"; });

  EXPECT_CALL(sink, PushTracingScope(FunctionReturns("scope2"))).Times(0);
  EXPECT_CALL(sink, PopTracingScope()).Times(0);
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Debug, [] { return "scope2"; });

  RequestTracing(false);

  // Should not call after tracing is disabled.
  EXPECT_CALL(sink, PushTracingScope(FunctionReturns("scope3"))).Times(0);
  EXPECT_CALL(sink, PopTracingScope()).Times(0);
  // NOLINTNEXTLINE(bugprone-unused-raii)
  TracingScope(TracingLevel::Default, [] { return "scope3"; });
}

}  // namespace
}  // namespace tracing
}  // namespace tfrt
