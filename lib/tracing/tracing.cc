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

void TracingSink::RecordTracingEvent(const char* category, const char* name) {
  RecordTracingEvent(category, string_view(name));
}

void TracingSink::RecordTracingEvent(const char* category, std::string&& name) {
  RecordTracingEvent(category, string_view(name));
}

void TracingSink::PushTracingScope(const char* category, const char* name) {
  PushTracingScope(category, string_view(name));
}

void TracingSink::PushTracingScope(const char* category, std::string&& name) {
  PushTracingScope(category, string_view(name));
}

TracingSink* internal::kTracingSink = nullptr;
std::atomic<int> internal::kIsTracingEnabled(0);

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
  auto previous_value = internal::kIsTracingEnabled.fetch_add(
      enable ? 1 : -1, std::memory_order_release);
  if (previous_value != (enable ? 0 : 1)) return;
  // Don't log error to avoid binary size bloat.
  consumeError(internal::kTracingSink->RequestTracing(enable));
}

}  // namespace tracing
}  // namespace tfrt
