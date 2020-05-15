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

#ifndef NO_TFRT_TRACING
#include <vector>

#include "tfrt/support/logging.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {
namespace internal {
namespace tracing {

static auto process_start = CurrentSteadyClockTime();
constexpr int kMAX_ACTIVITY_PER_THREAD = 100;

class ActivityStorage {
 public:
  ~ActivityStorage() {
    for (const auto& activity : activities_) {
      auto start_time = activity.start_time - process_start;

      if (activity.end_time.hasValue()) {
        auto end_time = activity.end_time.getValue() - process_start;
        auto duration = end_time - start_time;
        TFRT_LOG_INFO
            << "::: [" << activity.title << "]: "
            << std::chrono::duration_cast<std::chrono::microseconds>(start_time)
                   .count()
            << " us -- "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_time)
                   .count()
            << " ("
            << std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
                   .count()
            << " ns)";
      } else {
        TFRT_LOG_INFO << "::: [" << activity.title << "]: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                             start_time)
                             .count()
                      << " us";
      }
    }
    TFRT_LOG_INFO << "Total activities collected: " << counter_;
  }
  void AddTracingActivity(TracingActivity activity) {
    if (counter_++ < kMAX_ACTIVITY_PER_THREAD) {
      activities_.push_back(std::move(activity));
    } else {
      TFRT_TRACE_OFF();
    }
  }

 private:
  std::vector<TracingActivity> activities_;
  size_t counter_;
};

void SimpleRecordActivity(TracingActivity& activity) {
  static thread_local ActivityStorage storage;
  storage.AddTracingActivity(std::move(activity));
}

}  // namespace tracing
}  // namespace internal
}  // namespace tfrt
#endif  // NO_TFRT_TRACING
