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
#include <utility>

#include "llvm/Support/MathExtras.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/resource_context.h"
#include "tfrt/host_context/sync_kernel_frame.h"
#include "tfrt/support/variant.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

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
      kernel_fn_{[this, name]() -> KernelImplementation {
        const KernelRegistry& reg = host_->GetKernelRegistry();
        KernelImplementation impl = reg.GetKernel(name);
        if (impl.is<Monostate>()) {
          llvm::errs() << "Kernel not found: " << name << "\n";
          abort();
        }
        return impl;
      }()},
      req_ctx_builder_{host_, &resource_ctx_} {}

KernelRunner& KernelRunner::AddStringAttribute(string_view str) {
  attr_offsets_.emplace_back(bef_attr_encoder_.EncodeStringAttr(str));
  return *this;
}

void KernelRunner::Run(size_t num_results) {
  // First clear the previous results if any.
  results_.clear();
  sync_results_.clear();
  if (!req_ctx_) {
    Expected<RCReference<RequestContext>> req_ctx =
        std::move(req_ctx_builder_).build();
    assert(req_ctx);
    req_ctx_ = std::move(*req_ctx);
  }
  if (is_sync_kernel()) {
    RunSyncInternal(num_results);
  } else {
    RunAsyncInternal(num_results);
  }
}

KernelRunner& KernelRunner::AddDenseAttribute(const DenseHostTensor& dht) {
  attr_offsets_.emplace_back(
      SerializeDenseHostTensorToDenseAttr(dht, &bef_attr_encoder_));
  return *this;
}

void KernelRunner::RunAsyncInternal(size_t num_results) {
  KernelFrameBuilder frame{ExecutionContext{req_ctx_}};

  for (auto& arg : arguments_) {
    frame.AddArg(arg);
  }

  frame.SetAttributeSection(bef_attr_encoder_.result());
  frame.SetAttributes(attr_offsets_);
  frame.SetNumResults(num_results);
  kernel_fn_.get<AsyncKernelImplementation>()(&frame);

  for (auto& result : frame.GetResults()) {
    results_.emplace_back(std::move(result));
  }

  Await(host_, results_);
}

void KernelRunner::RunSyncInternal(size_t num_results) {
  sync_results_.resize(num_results);

  llvm::SmallVector<Value*, 16> registers;
  llvm::SmallVector<uint32_t, 16> argument_indices;
  argument_indices.reserve(arguments_.size());
  llvm::SmallVector<uint32_t, 16> result_indices;
  result_indices.reserve(num_results);
  llvm::SmallVector<const void*, 16> attributes;

  // Set up args
  for (auto& arg : sync_arguments_) {
    registers.emplace_back(&arg);
    argument_indices.emplace_back(registers.size() - 1);
  }

  // Set up results
  for (int i = 0; i < num_results; ++i) {
    registers.emplace_back(&sync_results_[i]);
    result_indices.emplace_back(registers.size() - 1);
  }

  // Set up attributes
  for (uint32_t attr_offset : attr_offsets_) {
    attributes.emplace_back(&bef_attr_encoder_.result()[attr_offset]);
  }

  ExecutionContext exec_ctx(req_ctx_);
  SyncKernelFrameBuilder frame{registers, exec_ctx};
  frame.SetArguments(argument_indices);
  frame.SetResults(result_indices);
  frame.SetAttributes(attributes);
  kernel_fn_.get<SyncKernelImplementation>()(&frame);
}

}  // namespace tfrt
