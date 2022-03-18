// Copyright 2022 The TensorFlow Runtime Authors
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

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "tfrt/support/error_util.h"
#include "tfrt/tracing/tracing.h"

// This file implements a tracing sink which produces activities that can be
// loaded in chrome://tracing. If run as part of a test, a trace.json file
// is written as undeclared test output. Otherwise it is written to stdout.
//
// Usage: replace simple_tracing_sink dependency of bef_executor target with
// chrome_tracing_sink and run with --enable_tracing.

namespace tfrt {
namespace tracing {

class ChromeTracingSink : public TracingSink {
  using Clock = std::chrono::high_resolution_clock;
  using Start = std::pair<std::string, Clock::time_point>;

  struct Entry {
    std::string name;
    Clock::time_point begin, end;
    std::thread::id tid;
    std::unique_ptr<Entry> next;
  };

  class Duration {
   public:
    explicit Duration(const Clock::duration& duration)
        : ns_(std::chrono::duration_cast<std::chrono::nanoseconds>(duration)) {}

    friend std::ostream& operator<<(std::ostream& os,
                                    const Duration& duration) {
      static const double us_per_ns = 1e-3;
      std::array<char, 32> array;
      snprintf(array.data(), array.size(), "%.3f",
               duration.ns_.count() * us_per_ns);
      return os << array.data();
    }

   private:
    std::chrono::nanoseconds ns_;
  };

 public:
  ~ChromeTracingSink() override { delete head_.exchange(nullptr); }

  Error RequestTracing(bool enable) override {
    if (enable) return Error::success();
    std::unique_ptr<Entry> head(head_.exchange(nullptr));
    std::ofstream ofs;
    if (const char* dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR"))
      ofs.open(dir + std::string("/trace.json"));
    std::ostream& os = ofs ? ofs : std::cout;
    os << "{\n  \"traceEvents\": [\n";
    for (; head; head = std::move(head->next)) {
      os << R"(    {"ph": "X", "name": ")" << head->name;
      os << R"(", "pid": 0, "tid": )" << head->tid;
      os << R"(, "ts": )" << Duration(head->begin - start_);
      os << R"(, "dur": )" << Duration(head->end - head->begin) << "},\n";
    }
    os << "    {}\n  ],\n  \"displayTimeUnit\": \"ns\"\n}\n";
    return Error::success();
  }

  void RecordTracingEvent(TracingSink::NameGenerator name_gen) override {
    auto now = Clock::now();
    auto entry = new Entry{name_gen(), now, now, std::this_thread::get_id()};
    entry->next.reset(head_.exchange(entry));
  }

  void PushTracingScope(TracingSink::NameGenerator name_gen) override {
    stack_.emplace_back(name_gen(), Clock::now());
  }

  void PopTracingScope() override {
    auto now = Clock::now();
    auto entry = new Entry{std::move(stack_.back().first), stack_.back().second,
                           now, std::this_thread::get_id()};
    entry->next.reset(head_.exchange(entry));
    stack_.pop_back();
  }

  const Clock::time_point start_ = Clock::now();
  static thread_local std::vector<Start> stack_;
  std::atomic<Entry*> head_ = {nullptr};
};

thread_local std::vector<ChromeTracingSink::Start> ChromeTracingSink::stack_;

static const bool kRegisterTracingSink = []() {
  RegisterTracingSink(new ChromeTracingSink);
  return true;
}();

}  // namespace tracing
}  // namespace tfrt
