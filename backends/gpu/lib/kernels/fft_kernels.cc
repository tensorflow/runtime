// Copyright 2022 The TensorFlow Runtime Authors
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

// This file implements the tfrt_gpu.fft kernel.
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/kernels/kernels_detail.h"
#include "tfrt/gpu/wrapper/fft_wrapper.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/support/fp16.h"

namespace tfrt {
namespace gpu {

// tfrt_gpu.fft_create_handle creates an FFT handle
static Expected<GpuFftHandle> FftCreate(
    Argument<GpuContext> context,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int64_t> batch, ArrayAttribute<int64_t> dims,
    ArrayAttribute<int64_t> in_strides, ArrayAttribute<int64_t> out_strides,
    Attribute<int> type) {
  if (!llvm::is_sorted(in_strides.data(), std::greater<int64_t>()) ||
      !llvm::is_sorted(out_strides.data(), std::greater<int64_t>())) {
    return MakeStringError("Only row-major layout is supported");
  }
  auto get_dimensions = [](ArrayRef<int64_t> strides) {
    llvm::SmallVector<int64_t, 3> dimensions(strides.size() - 1);
    for (int i = 0; i < dimensions.size(); ++i) {
      assert(strides[i + 1] != 0 && strides[i] % strides[i + 1] == 0);
      dimensions[i] = strides[i] / strides[i + 1];
    }
    return dimensions;
  };
  llvm::SmallVector<int64_t, 4> in_dims = get_dimensions(in_strides.data());
  llvm::SmallVector<int64_t, 4> out_dims = get_dimensions(out_strides.data());

  if (dims.size() != in_dims.size() || dims.size() != out_dims.size())
    return MakeStringError("Inconsistent dims/strides lengths");

  auto current = wrapper::CtxSetCurrent(context->get());
  if (!current) return current.takeError();

  auto handle = wrapper::FftCreate(*current);
  if (!handle) return handle.takeError();
  if (auto error = wrapper::FftDisableAutoAllocation(handle->get()))
    return std::move(error);

  auto fft_type = wrapper::FftType::FromOpaqueValue(*type);
  auto workspace_size = wrapper::FftMakePlanMany(
      handle->get(), fft_type, *batch, dims.data(), in_dims,
      in_strides[dims.size()], in_strides[0], out_dims,
      out_strides[dims.size()], out_strides[0]);
  if (!workspace_size) return workspace_size.takeError();

  return GpuFftHandle(context.ValueRef(), std::move(*handle), fft_type);
}

static Expected<int64_t> FftGetWorkspaceSize(const GpuFftHandle& handle) {
  return wrapper::FftGetWorkspaceSize(handle.get());
}

// tfrt_gpu.fft_exec executes the FFT plan associated with the given handle on a
// given stream.
static Error FftExecute(const GpuStream& stream, const GpuFftHandle& handle,
                        const GpuBuffer& input, const GpuBuffer& output,
                        const GpuBuffer& workspace, const Chain& chain,
                        Attribute<int> direction) {
  auto current = wrapper::CtxSetCurrent(stream.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::FftSetStream(handle.get(), stream.get()))
    return error;

  if (auto error = wrapper::FftSetWorkspace(handle.get(), workspace.pointer(),
                                            workspace.size())) {
    return error;
  }

  return wrapper::FftExec(*current, handle.get(), input.pointer(),
                          output.pointer(), handle.type(),
                          wrapper::FftDirection::FromOpaqueValue(*direction));
}

void RegisterGpuFftKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_gpu.fft.create", TFRT_KERNEL(FftCreate));
  kernel_reg->AddKernel("tfrt_gpu.fft.get_workspace_size",
                        TFRT_KERNEL(FftGetWorkspaceSize));
  kernel_reg->AddKernel("tfrt_gpu.fft.execute", TFRT_KERNEL(FftExecute));
}

}  // namespace gpu
}  // namespace tfrt
