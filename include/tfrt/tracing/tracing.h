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

#include <atomic>
#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace tracing {

class TracingSink {
 public:
  using NameGenerator = llvm::function_ref<std::string()>;

  virtual ~TracingSink();

  // This function is called before trace recording is enabled and after trace
  // recording has been disabled. Pending tracing scopes will still be popped
  // even after tracing has been disabled. If the function returns an error,
  // trace recording will not be enabled.
  virtual Error RequestTracing(bool enable) = 0;

  // Records an instant event for the calling thread.
  virtual void RecordTracingEvent(NameGenerator gen_name) = 0;

  // Pushes a tracing scope to the calling thread's stack.
  virtual void PushTracingScope(NameGenerator gen_name) = 0;
  // Ends the tracing scope from top of the calling thread's stack.
  // May be called after trace recording has been disabled.
  virtual void PopTracingScope() = 0;

  // The following functions forward to the above. Derived classes can override
  // them as an optimization if their sinks consume the corresponding type.
};

// When choosing a level, use
// Default for activities related to ops and kernels execution.
// Verbose for extra information which is cheap to generate.
// Debug for extra information which is expensive to generate
enum class TracingLevel { Default = 0, Verbose = 1, Debug = 2 };

namespace internal {
// The one and only registered tracing sink.
extern TracingSink* kTracingSink;
// Counter whether tracing is currently enabled. If positive, tracing events and
// scopes should be sent to the sink.
extern std::atomic<int> kIsTracingEnabled;

// Stores the current tracing level. All activities which lower level will be
// discarded.
extern std::atomic<TracingLevel> kTracingLevel;
}  // namespace internal

// Registers the tracing sink. Only one sink can be registered at any time.
// Tracing needs to be disabled during registration.
void RegisterTracingSink(TracingSink* tracing_sink);

#ifndef TFRT_DISABLE_TRACING
// Returns whether tracing is currently enabled.
inline bool IsTracingEnabled() {
  return internal::kIsTracingEnabled.load(std::memory_order_acquire) > 0;
}

inline bool IsAboveTracingLevel(TracingLevel level) {
  auto current_level = internal::kTracingLevel.load(std::memory_order_acquire);
  return static_cast<int>(current_level) >= static_cast<int>(level);
}

#else  // TFRT_DISABLE_TRACING
// Always return false because tracing is disabled at compile time.
constexpr inline bool IsTracingEnabled() { return false; }
constexpr inline bool IsAboveTracingLevel(TracingLevel) { return false; }

#endif

// Requests the tracing sink to enable or disable tracing.
void RequestTracing(bool enable);
void SetTracingLevel(TracingLevel level);

// RAII class to request tracing for the duration of the instance.
class TracingRequester {
  // No copy or assignment.
  TracingRequester(const TracingRequester&) = delete;
  void operator=(const TracingRequester&) = delete;

 public:
  TracingRequester() { RequestTracing(true); }
  ~TracingRequester() { RequestTracing(false); }
};

// Functions to add a tracing event.
inline void RecordTracingEvent(TracingLevel level,
                               TracingSink::NameGenerator gen_name) {
  if (IsTracingEnabled() && IsAboveTracingLevel(level)) {
    internal::kTracingSink->RecordTracingEvent(gen_name);
  }
}

// RAII class that pushes/pops a tracing scope.
class TracingScope {
  // No copy or assignment.
  TracingScope(const TracingScope&) = delete;
  TracingScope& operator=(const TracingScope&) = delete;

 public:
  TracingScope(TracingLevel level, TracingSink::NameGenerator get_name)
      : enabled_(IsTracingEnabled() && IsAboveTracingLevel(level)) {
    if (enabled_) internal::kTracingSink->PushTracingScope(get_name);
  }

  ~TracingScope() {
    if (enabled_) internal::kTracingSink->PopTracingScope();
  }

 private:
  const bool enabled_;
};

}  // namespace tracing
}  // namespace tfrt

// There are four developer-facing tracing macros:
// TFRT_TRACE_{""|KERNEL}_{EVENT|SCOPE}.
//
// The `TFRT_TRACE_KERNEL_*` use 'kernel' category and are intended for parts
// where a BEF kernel or function is being executed. The other macros use no
// category.
//
// `SCOPE` marks an activity with start and end, while `EVENT` marks a single
// time point. The recommendation is to use `*_SCOPE` when the tracing activity
// is long enough (~100ns) and `*_EVENT` otherwise.

#ifndef TFRT_DISABLE_TRACING
#define __TFRT_TRACE_GET_LEVEL(level) tfrt::tracing::TracingLevel::level
#define TFRT_TRACE_SCOPE(level, message)                                     \
  ::tfrt::tracing::TracingScope tracing_scope(__TFRT_TRACE_GET_LEVEL(level), \
                                              [&] { return message; })
#define TFRT_TRACE_EVENT(level, message)                             \
  ::tfrt::tracing::RecordTracingEvent(__TFRT_TRACE_GET_LEVEL(level), \
                                      [&] { return message; })

#else  // TFRT_DISABLE_TRACING
// Note: the above macro definitions would generate the same code as these stubs
// because IsTracingEnabled() always returns false and all code is eliminated.
#define TFRT_TRACE_SCOPE(level, message)
#define TFRT_TRACE_EVENT(level, message)
#endif  // TFRT_DISABLE_TRACING

#endif  // TFRT_TRACING_TRACING_H_
