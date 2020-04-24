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

//===- metrics_api.h --------------------------------------------*- C++ -*-===//
//
// This file declares the APIs to create/update metrics.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_METRICS_METRICS_API_H_
#define TFRT_METRICS_METRICS_API_H_

#include <string>
#include <vector>

namespace tfrt {
namespace metrics {

template <typename T>
class Gauge {
 public:
  virtual ~Gauge() {}

  virtual void SetValue(T value) = 0;
};

template <typename T>
Gauge<T>* NewGauge(std::string name);

}  // namespace metrics
}  // namespace tfrt

#endif  // TFRT_METRICS_METRICS_API_H_
