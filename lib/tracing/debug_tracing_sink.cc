// Copyright 2021 The TensorFlow Runtime Authors
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

//===- debug_tracing_sink.cc - An implementation of Debug Tracing Sink ----===//
//
// This file implements a debug tracing sink which prints activities to stdout.
//
//===----------------------------------------------------------------------===//

#include <string>

#include "llvm/Support/Error.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {
namespace tracing {

class DebugTracingSink : public TracingSink {
 public:
  Error RequestTracing(bool enable) override { return Error::success(); }

  void RecordTracingEvent(NameGenerator gen_name) override {
    os_ << "Event:" << gen_name() << "\n";
  }

  void PushTracingScope(NameGenerator gen_name) override {
    os_ << "Scope:" << gen_name() << "\n";
  }

  void PopTracingScope() override { os_ << "End Scope\n"; }

 private:
  llvm::raw_ostream& os_ = llvm::outs();
};

static const bool kRegisterTracingSink = [] {
  RegisterTracingSink(new DebugTracingSink);
  return true;
}();
}  // namespace tracing
}  // namespace tfrt
