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

#include "tfrt/tracing/nvtx_tracing_sink.h"

#include <iostream>

#include "tfrt/support/error_util.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {
namespace tracing {

void SetNvtxInjectionFunc(NvtxInitializeInjectionNvtxFunc_t func) {
  InitializeInjectionNvtx2_fnptr = func;
}

static nvtxEventAttributes_t GetEventAttributes(const char* name) {
  nvtxEventAttributes_t attrs = {NVTX_VERSION, NVTX_EVENT_ATTRIB_STRUCT_SIZE};
  attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attrs.message.ascii = name;
  return attrs;
}

static nvtxDomainHandle_t GetDomain() {
  static auto domain = nvtxDomainCreate("tfrt::tracing::NvtxTracingSink");
  return domain;
}

class NvtxTracingSink : public TracingSink {
  Error RequestTracing(bool enable) override {
    nvtxInitialize(/*reserved=*/nullptr);
    if (nvtxGlobals_v3.nvtxDomainMarkEx_impl_fnptr == nullptr)
      return MakeStringError("No NVTX backend injected.");
    return Error::success();
  }

  void RecordTracingEvent(TracingSink::NameGenerator name_gen) override {
    auto name = name_gen();
    auto attrs = GetEventAttributes(name.c_str());
    nvtxDomainMarkEx(GetDomain(), &attrs);
  }

  void PushTracingScope(TracingSink::NameGenerator name_gen) override {
    auto name = name_gen();
    auto attrs = GetEventAttributes(name.c_str());
    nvtxDomainRangePushEx(GetDomain(), &attrs);
  }

  void PopTracingScope() override { nvtxDomainRangePop(GetDomain()); }
};

static const bool kRegisterTracingSink = []() {
  RegisterTracingSink(new NvtxTracingSink);
  return true;
}();

}  // namespace tracing
}  // namespace tfrt
