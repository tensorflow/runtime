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

#include <chrono>
#include <future>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {
namespace tracing {
namespace {

class BenchmarkTracingSink : TracingSink {
 public:
  BenchmarkTracingSink() {
    RegisterTracingSink(this);
#ifndef TFRT_BM_DISABLE_TRACING_REQUEST
    tfrt::tracing::RequestTracing(true);
    EXPECT_TRUE(IsTracingEnabled());
#endif
  }
  ~BenchmarkTracingSink() override {
    tfrt::tracing::RequestTracing(false);
    EXPECT_FALSE(IsTracingEnabled());
    EXPECT_EQ(num_completed_, num_scopes_ + num_ranges_);
  }
  Error RequestTracing(bool enable) override { return Error::success(); }

 public:
  void RecordTracingEvent(NameGenerator gen_name) override { ++num_events_; }
  void PushTracingScope(NameGenerator gen_name) override { ++num_scopes_; }
  void PopTracingScope() override { ++num_completed_; }

 private:
  uint64_t num_events_ = 0;
  uint64_t num_scopes_ = 0;
  uint64_t num_ranges_ = 0;
  uint64_t num_completed_ = 0;
};

void BM_EmptyLoop(benchmark::State& state) {
  BenchmarkTracingSink sink;
  uint64_t dummy = 0;
  for (auto _ : state) {
    dummy++;
  }
}
BENCHMARK(BM_EmptyLoop);

void BM_TracingEvents(benchmark::State& state) {
  BenchmarkTracingSink sink;
  for (auto _ : state) {
    RecordTracingEvent(TracingLevel::Default, [] { return "event"; });
  }
}
BENCHMARK(BM_TracingEvents);

void BM_StrCatTracingEvents(benchmark::State& state) {
  BenchmarkTracingSink sink;
  for (auto _ : state) {
    RecordTracingEvent(TracingLevel::Default,
                       [] { return StrCat("event", ""); });
  }
}
BENCHMARK(BM_StrCatTracingEvents);

void BM_TracingScopes(benchmark::State& state) {
  BenchmarkTracingSink sink;
  for (auto _ : state) {
    TracingScope(TracingLevel::Default, [] { return "scope"; });
  }
}
BENCHMARK(BM_TracingScopes);

void BM_StrCatTracingScopes(benchmark::State& state) {
  BenchmarkTracingSink sink;
  for (auto _ : state) {
    TracingScope(TracingLevel::Default, [&] { return StrCat("scope", ""); });
  }
}
BENCHMARK(BM_StrCatTracingScopes);

void BM_InactiveTracingEvents(benchmark::State& state) {
  BenchmarkTracingSink sink;
  tfrt::tracing::SetTracingLevel(tfrt::tracing::TracingLevel::Default);
  for (auto _ : state) {
    TFRT_TRACE_EVENT(Debug, "event");
  }
}
BENCHMARK(BM_InactiveTracingEvents);

void BM_InactiveTracingScopes(benchmark::State& state) {
  BenchmarkTracingSink sink;
  tfrt::tracing::SetTracingLevel(tfrt::tracing::TracingLevel::Default);
  for (auto _ : state) {
    TFRT_TRACE_SCOPE(Debug, "scope");
  }
}
BENCHMARK(BM_InactiveTracingScopes);
}  // namespace
}  // namespace tracing
}  // namespace tfrt
