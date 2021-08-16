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

#include "tfrt/utils/kernel_runner.h"

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "llvm/Support/MathExtras.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/resource_context.h"

namespace tfrt {
namespace {

std::unique_ptr<HostContext> CreateDefaultHostContext() {
  auto host = std::make_unique<HostContext>(
      [&](const DecodedDiagnostic& diag) {
        llvm::errs() << "Diagnostic: " << diag << "\n";
      },
      CreateMallocAllocator(), CreateMultiThreadedWorkQueue(4, 4));

  RegisterStaticKernels(host->GetMutableRegistry());

  return host;
}

}  // namespace

KernelRunner::KernelRunner(string_view name, HostContext* host)
    : default_host_context_{host ? nullptr : CreateDefaultHostContext()},
      host_{host ? host : default_host_context_.get()},
      kernel_fn_{[this, name] {
        const KernelRegistry& reg = host_->GetKernelRegistry();
        KernelImplementation impl = reg.GetKernel(name);
        if (auto* kernel_fn = impl.get_if<AsyncKernelImplementation>()) {
          return *kernel_fn;
        } else {
          llvm::errs() << "Kernel not found: " << name << "\n";
          abort();
        }
      }()},
      req_ctx_builder_{host_, &resource_ctx_} {}

KernelRunner& KernelRunner::AddStringAttribute(string_view str) {
  attr_offsets_.emplace_back(bef_attr_encoder_.EncodeStringAttr(str));
  return *this;
}

void KernelRunner::Run(size_t num_results) {
  // First clear the previous results if any.
  results_.clear();

  if (!req_ctx_) {
    Expected<RCReference<RequestContext>> req_ctx =
        std::move(req_ctx_builder_).build();
    assert(req_ctx);
    req_ctx_ = std::move(*req_ctx);
  }

  KernelFrameBuilder frame{ExecutionContext{req_ctx_.CopyRef()}};

  for (auto& arg : arguments_) {
    frame.AddArg(arg.CopyRef());
  }

  frame.SetAttributeSection(bef_attr_encoder_.result());
  frame.SetAttributes(attr_offsets_);
  frame.SetNumResults(num_results);
  kernel_fn_(&frame);

  for (auto& result : frame.GetResults()) {
    results_.emplace_back(std::move(result));
  }

  Await(host_, results_);
}

}  // namespace tfrt
