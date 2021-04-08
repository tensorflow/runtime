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

// This file declares global functions to create metrics.

#ifndef TFRT_METRICS_METRICS_H_
#define TFRT_METRICS_METRICS_H_

#include <string>

#include "gauge.h"
#include "histogram.h"

namespace tfrt {
namespace metrics {

//===----------------------------------------------------------------------===//
// Methods to create Gauge metrics
//===----------------------------------------------------------------------===//

template <typename T>
Gauge<T>* NewGauge(std::string name);

template <>
Gauge<std::string>* NewGauge(std::string name);

//===----------------------------------------------------------------------===//
// Methods to create Histogram metrics
//===----------------------------------------------------------------------===//

Histogram* NewHistogram(std::string name, const Buckets& buckets);

}  // namespace metrics
}  // namespace tfrt

#endif  // TFRT_METRICS_METRICS_H_
