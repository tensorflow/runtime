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

//===- tracing.cc - Tracing API -------------------------------------------===//
//
// This file implements the tracing library.
#include "tfrt/tracing/tracing.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <mutex>
#include <utility>

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/logging.h"

namespace tfrt {
namespace tracing {

TracingSink::~TracingSink() = default;

raw_ostream& operator<<(raw_ostream& os, TracingLevel level) {
  switch (level) {
    case TracingLevel::None:
      return os << "none";
    case TracingLevel::Default:
      return os << "default";
    case TracingLevel::Verbose:
      return os << "verbose";
    case TracingLevel::Debug:
      return os << "debug";
    default:
      return os << "TracingLevel(" << static_cast<int>(level) << ")";
  }
}

Expected<TracingLevel> ParseTracingLevel(llvm::StringRef name) {
  return llvm::StringSwitch<Expected<TracingLevel>>(name)
      .Case("none", TracingLevel::None)
      .Case("default", TracingLevel::Default)
      .Case("verbose", TracingLevel::Verbose)
      .Case("debug", TracingLevel::Debug)
      .Default(MakeStringError("Unknown TracingLevel: ", name));
}

TracingSink* internal::kTracingSink = nullptr;
int internal::kTracingEnabled(0);
TracingLevel internal::kTracingLevel =
    std::min(TracingLevel::Default, internal::kMaxTracingLevel);
std::atomic<TracingLevel> internal::kCurrentTracingLevel(TracingLevel::None);

static std::mutex& GetTracingMutex() {
  static auto mutex = new std::mutex;
  return *mutex;
}

void RegisterTracingSink(TracingSink* tracing_sink) {
  std::lock_guard<std::mutex> lock(GetTracingMutex());
  assert(tracing_sink);
  assert(internal::kTracingEnabled == 0);
  internal::kTracingSink = tracing_sink;
}

void RequestTracing(bool enable) {
  std::lock_guard<std::mutex> lock(GetTracingMutex());
  if (internal::kTracingSink == nullptr) {
    TFRT_LOG(WARNING) << "No tfrt::TracingSink registered";
    return;
  }
  if (enable) {
    if (internal::kTracingEnabled++ > 0) return;  // Already enabled.
  } else {
    if (internal::kTracingEnabled == 0) return;   // Already disabled.
    if (--internal::kTracingEnabled > 0) return;  // Still enabled.
  }
  if (auto error = internal::kTracingSink->RequestTracing(enable)) {
    internal::kTracingEnabled = !enable;
    TFRT_LOG(WARNING) << std::move(error);
    return;
  }
  auto level = enable ? internal::kTracingLevel : TracingLevel::None;
  internal::kCurrentTracingLevel.store(level);
}

void SetTracingLevel(TracingLevel level) {
  if (level > internal::kMaxTracingLevel) {
    TFRT_LOG(WARNING) << "Tracing level '" << level
                      << "' clamped to TFRT_MAX_TRACING_LEVEL ('"
                      << internal::kMaxTracingLevel << "')";
    level = internal::kMaxTracingLevel;
  }
  std::lock_guard<std::mutex> lock(GetTracingMutex());
  internal::kTracingLevel = level;
  if (internal::kTracingEnabled) {
    internal::kCurrentTracingLevel.store(level);
  }
}

}  // namespace tracing
}  // namespace tfrt
