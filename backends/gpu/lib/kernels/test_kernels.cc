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

// This file implements the tfrt_gpu_test kernels.
#include "test_kernels.h"

#include <unordered_map>

#include "kernels_detail.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/support/mutex.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {
namespace gpu {

// Convert 'error' to string and report to 'out'.
static void ReportError(KernelErrorHandler out, llvm::Error error) {
  llvm::handleAllErrors(std::move(error), [&](const llvm::ErrorInfoBase& info) {
    out.ReportError(__FILE__, ':', __LINE__, ' ', info.message());
  });
}

// Convenience function that copies a host tensor to the device and returns a
// buffer pointing to the newly allocated memory. The intended purpose of this
// function is to make writing unit tests simpler
static Expected<GpuBuffer> TestCpyTensorHtoD(Argument<GpuAllocator> allocator,
                                             const GpuStream& stream,
                                             const DenseHostTensor& src) {
  size_t size_bytes = src.DataSizeInBytes();
  auto buffer =
      GpuBuffer::Allocate(allocator.ValueRef(), size_bytes, stream.get());
  if (!buffer) return buffer.takeError();
  auto current = wrapper::CtxSetCurrent(stream.context());
  if (!current) return current.takeError();
  if (auto error = wrapper::Memcpy(
          *current, buffer->pointer(),
          wrapper::Pointer<const void>(src.data(), current->platform()),
          size_bytes))
    return std::move(error);
  return buffer;
}

void RegisterGpuTestKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_gpu_test.copy_tensor_host_to_device",
                        TFRT_KERNEL(TestCpyTensorHtoD));
}

}  // namespace gpu
}  // namespace tfrt
