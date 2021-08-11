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
namespace detail {

void* BefBufferAllocator::allocate(size_t size, size_t align) {
  assert(align <= GetRequiredBefAlignment());

  // Find the next index with the required alignment.
  uintptr_t aligned_start = llvm::alignTo(data_.size(), align);

  size_t new_size = aligned_start + size;
  data_.resize(new_size);

  offsets_.emplace_back(aligned_start);
  return data_.data() + aligned_start;
}

}  // namespace detail

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

// Binary format of string attributes in BEF.
struct BEFStringAttribute {
  AttrSizeT size;
  char str[1];
};

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
  void* attr_addr = allocator_.allocate(sizeof(BEFStringAttribute) + str.size(),
                                        alignof(BEFStringAttribute));
  auto* attr = new (attr_addr) BEFStringAttribute();
  attr->size = str.size();
  memcpy(attr->str, str.data(), str.size());
  return *this;
}

void KernelRunner::Run(size_t num_results) {
  Expected<RCReference<RequestContext>> req_ctx =
      std::move(req_ctx_builder_).build();
  assert(req_ctx);

  KernelFrameBuilder frame{ExecutionContext{std::move(*req_ctx)}};

  for (auto& arg : arguments_) {
    frame.AddArg(std::move(arg));
  }

  frame.SetAttributeSection(allocator_.data());
  frame.SetAttributes(allocator_.offsets());
  frame.SetNumResults(num_results);
  kernel_fn_(&frame);

  for (auto& result : frame.GetResults()) {
    results_.emplace_back(std::move(result));
  }

  Await(host_, results_);
}

}  // namespace tfrt
