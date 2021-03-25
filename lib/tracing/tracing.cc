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
//
//===----------------------------------------------------------------------===//
#include "tfrt/tracing/tracing.h"

#include <cassert>
#include <mutex>

#include "llvm/Support/Error.h"

namespace tfrt {
namespace tracing {
TracingSink::~TracingSink() = default;

TracingSink* internal::kTracingSink = nullptr;
std::atomic<int> internal::kIsTracingEnabled(0);
std::atomic<TracingLevel> internal::kTracingLevel(TracingLevel::Default);

static std::mutex& GetTracingMutex() {
  static auto mutex = new std::mutex;
  return *mutex;
}

void RegisterTracingSink(TracingSink* tracing_sink) {
  std::unique_lock<std::mutex> lock(GetTracingMutex());
  assert(tracing_sink);
  assert(internal::kIsTracingEnabled.load(std::memory_order_acquire) == 0);
  internal::kTracingSink = tracing_sink;
}

void RequestTracing(bool enable) {
  std::unique_lock<std::mutex> lock(GetTracingMutex());
  if (internal::kTracingSink == nullptr) return;
  auto value = internal::kIsTracingEnabled.load(std::memory_order_acquire);
  if (enable) {
    if (value++ > 0) return;
  } else {
    if (value == 0 || --value > 0) return;
  }
  internal::kIsTracingEnabled.store(value, std::memory_order_release);
  // Don't log error to avoid binary size bloat.
  consumeError(internal::kTracingSink->RequestTracing(enable));
}

void SetTracingLevel(TracingLevel level) {
  internal::kTracingLevel.store(level);
}

}  // namespace tracing
}  // namespace tfrt
