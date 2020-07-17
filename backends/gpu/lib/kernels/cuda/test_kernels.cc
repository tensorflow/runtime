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

//===- test_kernels.cc - CUDA runtime interface ---------------------------===//
//
// This file defines the C++ functions that implement the test CUDA kernels
//
//===----------------------------------------------------------------------===//
#include "test_kernels.h"

#include <unordered_map>

#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/stream/cuda_wrapper.h"
#include "tfrt/gpu/stream/hash_utils.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/mutex.h"

namespace tfrt {
namespace cuda {

// Convert 'error' to string and report to 'out'.
static void ReportError(KernelErrorHandler out, llvm::Error error) {
  llvm::handleAllErrors(std::move(error), [&](const llvm::ErrorInfoBase& info) {
    out.ReportError(__FILE__, ':', __LINE__, ' ', info.message());
  });
}

// cuda_test.context.get returns the primary CUDA context for the given device.
static void TestContextGet(Argument<gpu::stream::Device> device,
                           Argument<Chain> in_chain,
                           Result<gpu::stream::Context> out_context,
                           Result<Chain> out_chain,
                           KernelErrorHandler handler) {
  static auto test_contexts =
      new std::unordered_map<gpu::stream::Device, gpu::stream::OwningContext>();
  static mutex* mu = new mutex();

  mutex_lock lock(*mu);
  auto it = test_contexts->find(*device);
  if (it == test_contexts->end()) {
    auto context = gpu::stream::DevicePrimaryCtxRetain(*device);
    if (!context) return ReportError(handler, context.takeError());
    auto res = test_contexts->insert({*device, std::move(*context)});
    assert(res.second && "Insertion must have succeeded");
    out_context.Emplace(res.first->second.get());
  } else {
    out_context.Emplace(it->second.get());
  }
  out_chain.Set(in_chain);
}

void RegisterTestCudaKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("cuda_test.context.get", TFRT_KERNEL(TestContextGet));
}

}  // namespace cuda
}  // namespace tfrt
