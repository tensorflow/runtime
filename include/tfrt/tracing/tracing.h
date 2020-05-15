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

//===- tracing.h - Tracing API ----------------------------------*- C++ -*-===//
//
// This file declares the tracing library.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_TRACING_TRACING_H_
#define TFRT_TRACING_TRACING_H_

#ifdef NO_TFRT_TRACING

// There are four developer-facing tracing macros:
// TFRT_TRACE_{""|KERNEL}_{EVENT|SCOPE}.
//
// The `TFRT_TRACE_KERNEL_*` behaves like `TFRT_TRACE_*` at the moment. The main
// difference is, logically, `*KERNEL*` variation marks only the parts when a
// BEF kernel or function is being executed.
//
// `SCOPE` marks an activity with start and end, while `EVENT` marks a single
// time point. The recommendation is to use `*_SCOPE` when the tracing activity
// is long enough (~100ns) and `*_EVENT` otherwise.

#define TFRT_TRACE_EVENT(id)
#define TFRT_TRACE_SCOPE(id)
#define TFRT_TRACE_KERNEL_EVENT(id)
#define TFRT_TRACE_KERNEL_SCOPE(id)
#define TFRT_TRACE_REGISTER_SINK(name, func)
#define TFRT_TRACE_ON()
#define TFRT_TRACE_OFF()

#else  // NO_TFRT_TRACING is not set.

#include <atomic>
#include <chrono>
#include <string>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace internal {
namespace tracing {
using SteadyTimePoint = std::chrono::steady_clock::time_point;
SteadyTimePoint CurrentSteadyClockTime();

struct TracingActivity {
  std::string title;
  SteadyTimePoint start_time;
  // EVENT does not have an end_time value.
  Optional<SteadyTimePoint> end_time;
  TracingActivity(const TracingActivity&) = delete;
  TracingActivity& operator=(const TracingActivity&) = delete;
  TracingActivity(TracingActivity&&) = default;
  TracingActivity& operator=(TracingActivity&&) = default;
};

struct TracingScope {
 public:
  explicit TracingScope(tfrt::string_view title);
  ~TracingScope();
  TracingScope(TracingScope&&) noexcept;
  TracingScope(const TracingScope&) = delete;
  TracingScope& operator=(const TracingScope&) = delete;

 private:
  TracingActivity tracing_activity_;
};

using TracingSinkFunction = void(TracingActivity&);

class TracingApi {
 public:
  static bool IsTracingOn() { return is_tracing_on_; }
  static void RecordActivity(TracingActivity& activity) {
    if (record_activity_function_ != nullptr) {
      record_activity_function_(activity);
    }
  }
  static void TurnTracingOn();
  static void TurnTracingOff();
  static void RegisterTracingSink(const std::string&, TracingSinkFunction);

 private:
  static std::atomic<bool> is_tracing_on_;
  static TracingSinkFunction* record_activity_function_;
};

}  // namespace tracing
}  // namespace internal

#define TFRT_TRACE_ON ::tfrt::internal::tracing::TracingApi::TurnTracingOn
#define TFRT_TRACE_OFF ::tfrt::internal::tracing::TracingApi::TurnTracingOff

#define __TFRT_TRACE_UNIQUE_NAME(base_name) \
  __TFRT_TRACE_NAME_MERGE(base_name, __COUNTER__)
#define __TFRT_TRACE_NAME_MERGE(name1, name2) \
  __TFRT_TRACE_NAME_MERGE_(name1, name2)
#define __TFRT_TRACE_NAME_MERGE_(name1, name2) name1##name2

#define TFRT_TRACE_KERNEL_EVENT(id)                                            \
  if (::tfrt::internal::tracing::TracingApi::IsTracingOn()) {                  \
    ::tfrt::internal::tracing::TracingActivity __tracing_activity{             \
        id, ::tfrt::internal::tracing::CurrentSteadyClockTime(), {}};          \
    ::tfrt::internal::tracing::TracingApi::RecordActivity(__tracing_activity); \
  }

#define TFRT_TRACE_KERNEL_SCOPE(id)                                            \
  ::tfrt::Optional<::tfrt::internal::tracing::TracingScope>                    \
      __TFRT_TRACE_UNIQUE_NAME(__tracing_point){                               \
          [&]() -> ::tfrt::Optional<::tfrt::internal::tracing::TracingScope> { \
            if (::tfrt::internal::tracing::TracingApi::IsTracingOn()) {        \
              return ::tfrt::internal::tracing::TracingScope(id);              \
            } else {                                                           \
              return llvm::None;                                               \
            }                                                                  \
          }()};

#define TFRT_TRACE_EVENT(id) TFRT_TRACE_KERNEL_EVENT(id)

#define TFRT_TRACE_SCOPE(id) TFRT_TRACE_KERNEL_SCOPE(id)

#define TFRT_TRACE_REGISTER_SINK(NAME, FUNC)                                \
  static const bool __TFRT_TRACE_UNIQUE_NAME(__tracing_sink) = []() {       \
    ::tfrt::internal::tracing::TracingApi::RegisterTracingSink(NAME, FUNC); \
    return true;                                                            \
  }()

}  // namespace tfrt

#endif  // NO_TFRT_TRACING
#endif  // TFRT_TRACING_TRACING_H_
