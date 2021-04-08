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

// This file provides methods to define a few common TFRT metrics.

#ifndef TFRT_METRICS_COMMON_METRICS_H_
#define TFRT_METRICS_COMMON_METRICS_H_

#include "tfrt/metrics/metrics.h"

namespace tfrt {
namespace metrics {

inline void AddTFRTVersionMetric() {
  static auto* version_metric = [] {
    auto* m = metrics::NewGauge<std::string>("/tensorflow/runtime/version");
    m->Set("TFRT_V0");
    return m;
  }();
  (void)version_metric;
}

}  // namespace metrics
}  // namespace tfrt

#endif  // TFRT_METRICS_COMMON_METRICS_H_
