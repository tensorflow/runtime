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

//===- metrics_api_dummy.cc -----------------------------------------------===//
//
// This file provides a dummy implementation of all metrics APIs if
// ENABLE_TFRT_METRICS is not defined.
//
//===----------------------------------------------------------------------===//

#include "tfrt/metrics/metrics_api.h"

namespace tfrt {
namespace metrics {

// If ENABLE_TFRT_METRICS is not defined, provide a dummy implementation of the
// metrics APIs defined in metrics_api.h. Otherwise, we assume users will
// provide their own implementation of all metrics APIs as a library during TFRT
// compilation.

#if !defined(ENABLE_TFRT_METRICS)
template <typename T>
class DummyGauge : public Gauge<T> {
 public:
  DummyGauge() {}

  void SetValue(T value) override {}
};

template <typename T>
Gauge<T>* NewGauge(std::string name) {
  return new DummyGauge<T>();
}

template Gauge<std::string>* NewGauge<std::string>(std::string name);
#endif

}  // namespace metrics
}  // namespace tfrt
