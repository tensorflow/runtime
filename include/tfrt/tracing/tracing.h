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

// Tracing API
//
// This file declares the tracing library.

#ifndef TFRT_TRACING_TRACING_H_
#define TFRT_TRACING_TRACING_H_

#include <atomic>
#include <cstdint>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "tfrt/support/forward_decls.h"

#define TFRT_GET_TRACE_LEVEL(level) ::tfrt::tracing::TracingLevel::level

namespace tfrt {
namespace tracing {

class TracingSink {
 public:
  using NameGenerator = llvm::function_ref<std::string()>;

  virtual ~TracingSink();

  // This function is called before trace recording is enabled and after trace
  // recording has been disabled. Pending tracing scopes will still be popped
  // even after tracing has been disabled. If the function returns an error,
  // the tracing state is not changed.
  virtual Error RequestTracing(bool enable) = 0;

  // Records an instant event for the calling thread.
  virtual void RecordTracingEvent(NameGenerator gen_name) = 0;

  // Pushes a tracing scope to the calling thread's stack.
  virtual void PushTracingScope(NameGenerator gen_name) = 0;
  // Ends the tracing scope from top of the calling thread's stack.
  // May be called after trace recording has been disabled.
  virtual void PopTracingScope() = 0;
};

// Enum specifying the verbosity of tracing activities.
//
// A tracing levels is disabled if its value is larger than
// TFRT_MAX_TRACING_LEVEL. Tracing activities with disabled constexpr levels
// incur zero runtime overhead.
//
// A tracing level is inactive if its value is larger than
// GetCurrentTracingLevel(). Tracing activities with inactive levels incur a
// small runtime overhead (comperable to an inactive VLOG).
enum class TracingLevel {
  // Tracing level to pass to SetTracingLevel() to deactivate all activities.
  None = 0,
  // Enabled and active by default. Use this level for activities that are
  // useful for everyone. Generating the activity name should be cheap.
  Default = 1,
  // Enabled and inactive by default. Use this level for activities that are
  // not directly useful for an end user, but you might want to see without
  // having to rebuild the binary. Generating the activity name should be
  // cheap enough so that timing information is not skewed.
  Verbose = 2,
  // Disabled and inactive by default. Use this level for activities that are
  // too detailed to be generally useful, but you still want to see (after
  // rebuilding your binary). Generating the activity name can be expensive.
  Debug = 3,
};
raw_ostream& operator<<(raw_ostream& os, TracingLevel level);
Expected<TracingLevel> ParseTracingLevel(llvm::StringRef name);

namespace internal {
// Compile-time tracing level above which all activities are disabled.
constexpr auto kMaxTracingLevel = TFRT_GET_TRACE_LEVEL(TFRT_MAX_TRACING_LEVEL);
// The one and only registered tracing sink.
extern TracingSink* kTracingSink;
// Counter whether tracing is currently enabled. If positive, tracing events and
// scopes should be sent to the sink.
extern int kTracingEnabled;
// Tracing level when tracing is enabled.
extern TracingLevel kTracingLevel;

// The current tracing level. All activities with lower level will be discarded.
extern std::atomic<TracingLevel> kCurrentTracingLevel;
}  // namespace internal

// Registers the tracing sink. Only one sink can be registered at any time.
// Tracing needs to be disabled during registration.
void RegisterTracingSink(TracingSink* tracing_sink);

// Returns the current tracing level.
inline TracingLevel GetCurrentTracingLevel() {
  return internal::kCurrentTracingLevel.load(std::memory_order_acquire);
}

// Returns whether tracing is enabled for the given level.
inline bool IsTracingEnabled(TracingLevel level) {
  // Evaluated at compile time, eliminates all activities if
  // TFRT_MAX_TRACING_LEVEL is set to None. Change to 'if constexpr' in C++17.
  if (internal::kMaxTracingLevel <= TracingLevel::None) return false;
  // Likely evaluated at compile time, eliminates activities if level is a
  // compile time constant above TFRT_MAX_TRACING_LEVEL.
  // Works for gcc >= v10.3 and all versions of clang and msvc.
  if (internal::kMaxTracingLevel < level) return false;
  return level <= GetCurrentTracingLevel();
}

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
  if (IsTracingEnabled(level)) {
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
      : enabled_(IsTracingEnabled(level)) {
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

// `TFRT_TRACE_SCOPE` marks an activity with start and end, while
// `TFRT_TRACE_EVENT` marks a single time point. The recommendation is to use
// scope when the tracing activity is long enough (~100ns) and event otherwise.
#define TFRT_TRACE_SCOPE(level, message)                                   \
  ::tfrt::tracing::TracingScope tracing_scope(TFRT_GET_TRACE_LEVEL(level), \
                                              [&] { return message; })
#define TFRT_TRACE_EVENT(level, message)                           \
  ::tfrt::tracing::RecordTracingEvent(TFRT_GET_TRACE_LEVEL(level), \
                                      [&] { return message; })

#endif  // TFRT_TRACING_TRACING_H_
