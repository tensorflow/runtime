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

// This file declares the Histogram metric interface.

#ifndef TFRT_METRICS_HISTOGRAM_H_
#define TFRT_METRICS_HISTOGRAM_H_

#include <cassert>
#include <vector>

namespace tfrt {
namespace metrics {

// Bucket ranges used by the Histogram metric.
class Buckets {
 public:
  // Returns a Buckets whose lower bounds (except for the underflow bucket) are
  // given by `bounds`. The underflow bucket will have an upper bound of
  // `bounds.front()` and the overflow bucket will have a lower bound of
  // `bounds.back()`.
  //
  // REQUIRES: |bounds| contains a non-empty sequence of monotonically
  // increasing finite numbers.
  static Buckets Explicit(std::vector<double> bounds) {
    assert(!bounds.empty());
    return Buckets(std::move(bounds));
  }

  const std::vector<double>& explicit_bounds() const { return bounds_; }

 private:
  explicit Buckets(std::vector<double> bounds) : bounds_(std::move(bounds)) {}

  const std::vector<double> bounds_;
};

// The Histogram metric interface.
class Histogram {
 public:
  virtual ~Histogram() {}

  virtual void Record(double value) = 0;
};

}  // namespace metrics
}  // namespace tfrt

#endif  // TFRT_METRICS_HISTOGRAM_H_
