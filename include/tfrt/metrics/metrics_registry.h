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

// This file declares the MetricsRegistry interface.

#ifndef TFRT_METRICS_METRICS_REGISTRY_H_
#define TFRT_METRICS_METRICS_REGISTRY_H_

#include <string>

#include "gauge.h"
#include "histogram.h"

namespace tfrt {
namespace metrics {

class MetricsRegistry {
 public:
  virtual ~MetricsRegistry() {}

  virtual Gauge<std::string>* NewStringGauge(std::string name) = 0;

  virtual Histogram* NewHistogram(std::string name, const Buckets& buckets) = 0;
};

namespace internal {
// The global metric registry to be used for all TFRT metrics.
extern MetricsRegistry* kMetricsRegistry;
}  // namespace internal

// Registers the metrics registry. Only one registry can be registered at any
// time.
void RegisterMetricsRegistry(MetricsRegistry* metrics_registry);

}  // namespace metrics
}  // namespace tfrt

#endif  // TFRT_METRICS_METRICS_REGISTRY_H_
