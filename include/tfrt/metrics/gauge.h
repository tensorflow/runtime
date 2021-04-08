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

// This file declares the Gauge metric interface.

#ifndef TFRT_METRICS_GAUGE_H_
#define TFRT_METRICS_GAUGE_H_

namespace tfrt {
namespace metrics {

// The Gauge metric interface.
template <typename T>
class Gauge {
 public:
  virtual ~Gauge() {}

  virtual void Set(T value) = 0;
};

}  // namespace metrics
}  // namespace tfrt

#endif  // TFRT_METRICS_GAUGE_H_
