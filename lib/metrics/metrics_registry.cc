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

// This file defines functions to register the MetricsRegistry instance.

#include "tfrt/metrics/metrics_registry.h"

#include <cassert>
#include <mutex>

namespace tfrt {
namespace metrics {

MetricsRegistry* internal::kMetricsRegistry = nullptr;

static std::mutex& GetMetricsMutex() {
  static auto mutex = new std::mutex;
  return *mutex;
}

void RegisterMetricsRegistry(MetricsRegistry* metrics_registry) {
  std::lock_guard<std::mutex> lock(GetMetricsMutex());
  assert(metrics_registry);
  assert(internal::kMetricsRegistry == nullptr);
  internal::kMetricsRegistry = metrics_registry;
}

}  // namespace metrics
}  // namespace tfrt
