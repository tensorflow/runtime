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

#include <chrono>
#include <future>
#include <string>

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
    tfrt::tracing::RequestTracing(true);
  }
  ~BenchmarkTracingSink() override { tfrt::tracing::RequestTracing(false); }
  Error RequestTracing(bool enable) override { return Error::success(); }

 public:
  void RecordTracingEvent(NameGenerator gen_name) override { gen_name(); }
  void PushTracingScope(NameGenerator gen_name) override { gen_name(); }
  void PopTracingScope() override {}

 private:
};

template <typename F>
static void BenchmarkImpl(benchmark::State& state, F func) {
  BenchmarkTracingSink sink;
  for (auto _ : state) {
    func();
  }
}

static void BM_EmptyLoop(benchmark::State& state) {
  BenchmarkImpl(state, [] {});
}
BENCHMARK(BM_EmptyLoop);

static std::string StrName() { return "name"; }
static std::string StrCatName() { return StrCat("name"); }

template <TracingLevel level, std::string (*gen_name)()>
static void RecordEvent() {
  RecordTracingEvent(level, gen_name);
}

static void BM_DebugTracingEvents(benchmark::State& state) {
  BenchmarkImpl(state, RecordEvent<TracingLevel::Debug, StrName>);
}
static void BM_VerboseTracingEvents(benchmark::State& state) {
  BenchmarkImpl(state, RecordEvent<TracingLevel::Verbose, StrName>);
}
static void BM_DefaultTracingEvents(benchmark::State& state) {
  BenchmarkImpl(state, RecordEvent<TracingLevel::Default, StrName>);
}
static void BM_StrCatTracingEvents(benchmark::State& state) {
  BenchmarkImpl(state, RecordEvent<TracingLevel::Default, StrCatName>);
}
BENCHMARK(BM_DebugTracingEvents);
BENCHMARK(BM_VerboseTracingEvents);
BENCHMARK(BM_DefaultTracingEvents);
BENCHMARK(BM_StrCatTracingEvents);

template <TracingLevel level, std::string (*gen_name)()>
static void RecordScope() {
  TracingScope(level, gen_name);
}

static void BM_DebugTracingScopes(benchmark::State& state) {
  BenchmarkImpl(state, RecordScope<TracingLevel::Debug, StrName>);
}
static void BM_VerboseTracingScopes(benchmark::State& state) {
  BenchmarkImpl(state, RecordScope<TracingLevel::Verbose, StrName>);
}
static void BM_DefaultTracingScopes(benchmark::State& state) {
  BenchmarkImpl(state, RecordScope<TracingLevel::Default, StrName>);
}
static void BM_StrCatTracingScopes(benchmark::State& state) {
  BenchmarkImpl(state, RecordScope<TracingLevel::Default, StrCatName>);
}
BENCHMARK(BM_DebugTracingScopes);
BENCHMARK(BM_VerboseTracingScopes);
BENCHMARK(BM_DefaultTracingScopes);
BENCHMARK(BM_StrCatTracingScopes);

}  // namespace
}  // namespace tracing
}  // namespace tfrt
