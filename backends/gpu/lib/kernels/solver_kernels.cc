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

// This file implements the tfrt_gpu.solver kernels.
// Support for ROCm has been partially implemented.

#include <cstdint>

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/kernels/kernels_detail.h"
#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/cublas_wrapper.h"
#include "tfrt/gpu/wrapper/rocblas_wrapper.h"
#include "tfrt/gpu/wrapper/cusolver_wrapper.h"
#include "tfrt/gpu/wrapper/rocsolver_wrapper.h"
#include "tfrt/gpu/wrapper/solver_wrapper.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/kernel_registry.h"

namespace tfrt {
namespace gpu {

static llvm::Expected<GpuSolverHandle> SolverCreate(
    Argument<GpuContext> context) {
  auto current = wrapper::CtxSetCurrent(context->get());
  if (!current) return current.takeError();
  auto handle = wrapper::SolverCreate(current->platform());
  if (!handle) return handle.takeError();
  return GpuSolverHandle(context.ValueRef(), std::move(*handle));
}

static llvm::Expected<int64_t> SolverPotrfBufferSize(
    const GpuSolverHandle& handle, const GpuStream& stream, int32_t n,
    int32_t stride, Attribute<int32_t> dataType, Attribute<int32_t> fillMode) {
  auto platform = handle->platform();
  if (platform != wrapper::Platform::CUDA)
    return MakeStringError("Unsupported platform ", platform);

  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::SolverSetStream(handle.get(), stream.get()))
    return std::move(error);

  cudaDataType data_type = wrapper::BlasDataType::FromOpaqueValue(*dataType);
  auto fill_mode = wrapper::BlasFillMode::FromOpaqueValue(*fillMode);

  auto call = [&](auto dummy) {
    using Pointer = wrapper::Pointer<decltype(dummy)>;
    return wrapper::CusolverDnPotrfBufferSize(
        current.get(), handle.get(), fill_mode, n, Pointer(nullptr, platform),
        stride);
  };

  switch (data_type) {
    case CUDA_R_32F:
      return call(float{});
    case CUDA_R_64F:
      return call(double{});
    case CUDA_C_32F:
      return call(cuComplex{});
    case CUDA_C_64F:
      return call(cuDoubleComplex{});
    default:
      return MakeStringError("Unsupported data type ", data_type);
  }
}

static Error SolverPotrf(const GpuSolverHandle& handle, const GpuStream& stream,
                         int32_t n, const GpuBuffer& buffer, int32_t stride,
                         const GpuBuffer& workspace, const GpuBuffer& devInfo,
                         Attribute<int32_t> dataType,
                         Attribute<int32_t> fillMode) {
  auto platform = handle->platform();

  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::SolverSetStream(handle.get(), stream.get()))
    return error;

  auto data_type = wrapper::BlasDataType::FromOpaqueValue(*dataType);
  auto fill_mode = wrapper::BlasFillMode::FromOpaqueValue(*fillMode);

  return wrapper::SolverPotrf(
      *current, handle.get(), data_type, fill_mode, n,
      static_cast<wrapper::Pointer<void>>(buffer.pointer()), stride,
      static_cast<wrapper::Pointer<void>>(workspace.pointer()), workspace.size(),
      static_cast<wrapper::Pointer<int>>(devInfo.pointer()));
}

static Error SolverPotrfBatch(const GpuSolverHandle& handle,
                              const GpuStream& stream, int32_t n,
                              const GpuBuffer& buffer, int32_t stride,
                              const GpuBuffer& devInfo, int32_t batch_size,
                              Attribute<int32_t> dataType,
                              Attribute<int32_t> fillMode) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::SolverSetStream(handle.get(), stream.get()))
    return error;

  auto platform = handle->platform();

  auto data_type = wrapper::BlasDataType::FromOpaqueValue(*dataType);
  auto fill_mode = wrapper::BlasFillMode::FromOpaqueValue(*fillMode);

  auto data_type_size_bytes = wrapper::GetBlasDataTypeSizeBytes(data_type);
  if (!data_type_size_bytes) return data_type_size_bytes.takeError();

  auto pointer_array =
      handle.context()->AllocateHostPoolMemory<void*>(*current, batch_size);
  if (!pointer_array) return pointer_array.takeError();
  void** buffer_array = pointer_array->get().raw(platform);

  char* buffer_ptr = static_cast<char*>(buffer.pointer().raw(platform));
  ptrdiff_t batch_stride_bytes = *data_type_size_bytes * n * n;
  for (int32_t i = 0; i < batch_size; ++i) {
    buffer_array[i] = buffer_ptr + i * batch_stride_bytes;
  }

  wrapper::Pointer<void*> buffer_array_ptr(buffer_array, platform);
  auto devInfoPtr = static_cast<wrapper::Pointer<int>>(devInfo.pointer());
  return wrapper::SolverPotrfBatched(
      *current, handle.get(), data_type, fill_mode, n, buffer_array_ptr,
      stride, devInfoPtr, batch_size);
}

void RegisterGpuSolverKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_gpu.solver.create", TFRT_KERNEL(SolverCreate));
  kernel_reg->AddKernel("tfrt_gpu.solver.potrf.buffer_size",
                        TFRT_KERNEL(SolverPotrfBufferSize));
  kernel_reg->AddKernel("tfrt_gpu.solver.potrf",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(SolverPotrf));
  kernel_reg->AddKernel("tfrt_gpu.solver.potrf.batch",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(SolverPotrfBatch));
}

}  // namespace gpu
}  // namespace tfrt
