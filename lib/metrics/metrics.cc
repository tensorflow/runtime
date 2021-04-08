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

// This file defines global functions to create metrics.

#include "tfrt/metrics/metrics.h"

#include "tfrt/metrics/metrics_registry.h"

namespace tfrt {
namespace metrics {

// A dummy implementation of the Gauge metric interface.
template <typename T>
class DummyGauge : public Gauge<T> {
 public:
  DummyGauge() {}

  void Set(T value) override {}
};

// A dummy implementation of the Histogram metric interface.
class DummyHistogram : public Histogram {
 public:
  DummyHistogram() {}

  void Record(double value) override {}
};

template <>
Gauge<std::string>* NewGauge(std::string name) {
  if (internal::kMetricsRegistry != nullptr)
    return internal::kMetricsRegistry->NewStringGauge(name);
  return new DummyGauge<std::string>();
}

Histogram* NewHistogram(std::string name, const Buckets& buckets) {
  if (internal::kMetricsRegistry != nullptr)
    return internal::kMetricsRegistry->NewHistogram(name, buckets);
  return new DummyHistogram();
}

}  // namespace metrics
}  // namespace tfrt
