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

//===- simple_tracing_sink.h - Xprof Tracing Consumer -----------*- C++ -*-===//
//
// This file declares a simple tracing consumer.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_TRACING_SIMPLE_TRACING_SINK_H_
#define TFRT_TRACING_SIMPLE_TRACING_SINK_H_

#include "tfrt/tracing/tracing.h"

namespace tfrt {
namespace tracing {

class SimpleTracingSink : public TracingSink {
 public:
  Error RequestTracing(bool enable) override;
  void RecordTracingEvent(NameGenerator gen_name) override;
  void PushTracingScope(NameGenerator gen_name) override;
  void PopTracingScope() override;
};
}  // namespace tracing
}  // namespace tfrt

#endif  // TFRT_TRACING_SIMPLE_TRACING_SINK_H_
