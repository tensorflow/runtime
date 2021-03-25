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

//===- simple_tracing_sink.cc - A Simple implementation of Tracing Sink ---===//
//
// This file implements a simple tracing sink which prints activities to stderr.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tracing/simple_tracing_sink/simple_tracing_sink.h"

#include <chrono>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "tfrt/support/logging.h"

namespace tfrt {
namespace tracing {

namespace {

const auto kProcessStart = std::chrono::steady_clock::now();

class TracingStorage {
  using Time = std::chrono::steady_clock::time_point;

  struct Activity {
    std::string name;
    Time start;
    Time end;
  };

  static const int kMaxEntries = 100;

 public:
  ~TracingStorage() {
    using microseconds = std::chrono::microseconds;
    using nanoseconds = std::chrono::nanoseconds;

    for (const auto& activity : activities_) {
      auto start_us = std::chrono::duration_cast<microseconds>(activity.start -
                                                               kProcessStart);

      if (activity.start == activity.end) {
        TFRT_LOG(INFO) << "::: [" << activity.name << "]: " << start_us.count()
                       << " us";
        continue;
      }

      auto end_us = std::chrono::duration_cast<microseconds>(activity.end -
                                                             kProcessStart);
      auto duration_ns = std::chrono::duration_cast<nanoseconds>(
          activity.end - activity.start);

      TFRT_LOG(INFO) << "::: [" << activity.name << "]: " << start_us.count()
                     << " us -- " << end_us.count() << " ("
                     << duration_ns.count() << " ns)";
    }
    TFRT_LOG(INFO) << "Total activities collected: " << activities_.size();
  }

  void RecordEvent(std::string&& name) {
    if (activities_.size() >= kMaxEntries) return;
    auto now = Now();
    activities_.push_back(Activity{std::move(name), now, now});
  }

  void PushScope(std::string&& name) {
    if (activities_.size() >= kMaxEntries) return;
    stack_.emplace_back(std::move(name), Now());
  }

  void PopScope() {
    if (activities_.size() >= kMaxEntries) return;
    if (stack_.empty()) return;
    activities_.push_back({std::move(std::get<std::string>(stack_.back())),
                           std::get<Time>(stack_.back()), Now()});
    stack_.pop_back();
  }

 private:
  static Time Now() { return std::chrono::steady_clock::now(); }

  llvm::SmallVector<std::tuple<std::string, Time>, 16> stack_;
  llvm::SmallVector<Activity, kMaxEntries> activities_;
};
}  // namespace

static TracingStorage& GetTracingStorage() {
  static thread_local TracingStorage tracing_storage;
  return tracing_storage;
}

Error SimpleTracingSink::RequestTracing(bool enable) {
  return Error::success();
}

void SimpleTracingSink::RecordTracingEvent(
    TracingSink::NameGenerator gen_name) {
  GetTracingStorage().PushScope(gen_name());
}
void SimpleTracingSink::PushTracingScope(TracingSink::NameGenerator gen_name) {
  GetTracingStorage().PushScope(gen_name());
}
void SimpleTracingSink::PopTracingScope() { GetTracingStorage().PopScope(); }

}  // namespace tracing
}  // namespace tfrt
