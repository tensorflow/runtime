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

// This file implements the tfrt_gpu.blas kernels.
#include <cstdint>

#include "llvm/Support/Errc.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/kernels/kernels_detail.h"
#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/cublas_wrapper.h"
#include "tfrt/gpu/wrapper/rocblas_wrapper.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/support/fp16.h"

namespace tfrt {
namespace gpu {

static Expected<GpuBlasHandle> BlasCreate(Argument<GpuStream> stream) {
  auto current = wrapper::CtxSetCurrent(stream->context()->get());
  if (!current) return current.takeError();
  auto handle = wrapper::BlasCreate(*current);
  if (!handle) return handle.takeError();
  if (auto error = wrapper::BlasSetStream(handle->get(), stream->get()))
    return std::move(error);
  return GpuBlasHandle(stream.ValueRef(), std::move(*handle));
}

template <typename T>
static llvm::Expected<wrapper::Pointer<void>> GetScalePointer(
    AsyncValue* value, mlir::TypeID typeId, wrapper::Platform platform) {
  if (mlir::TypeID::get<T>() != typeId)
    return MakeStringError("unexpected argument type");
  return wrapper::Pointer<void>(&value->get<T>(), platform);
}

static llvm::Expected<wrapper::Pointer<void>> GetScalePointer(
    AsyncValue* value, mlir::TypeID typeId, wrapper::Platform platform) {
  if (value->IsType<fp16>())
    return GetScalePointer<fp16>(value, typeId, platform);
  if (value->IsType<float>())
    return GetScalePointer<float>(value, typeId, platform);
  if (value->IsType<double>()) {
    return GetScalePointer<double>(value, typeId, platform);
  }
  return MakeStringError("pointer type not supported");
}

static llvm::Expected<wrapper::Pointer<void>> GetScalePointer(
    AsyncValue* value, wrapper::BlasDataType data_type) {
  auto type_id = wrapper::GetBlasDataTypeId(data_type);
  return GetScalePointer(value, type_id, data_type.platform());
}

static llvm::Expected<wrapper::Pointer<void>> GetScalePointer(
    AsyncValue* value, wrapper::BlasComputeType compute_type) {
  auto type_id = wrapper::GetBlasComputeTypeId(compute_type);
  return GetScalePointer(value, type_id, compute_type.platform());
}

static Error BlasAxpy(const GpuBlasHandle& handle, int32_t n, AsyncValue* alpha,
                      const GpuBuffer& x, int32_t strideX, const GpuBuffer& y,
                      int32_t strideY, Attribute<int32_t> executionType,
                      Attribute<int32_t> typeAlpha, Attribute<int32_t> typeX,
                      Attribute<int32_t> typeY) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  auto type_alpha = wrapper::BlasDataType::FromOpaqueValue(*typeAlpha);
  auto alpha_ptr = GetScalePointer(alpha, type_alpha);
  if (!alpha_ptr) return alpha_ptr.takeError();

  return wrapper::BlasAxpyEx(
      *current, handle.get(), n, *alpha_ptr, type_alpha, x.pointer(),
      wrapper::BlasDataType::FromOpaqueValue(*typeX), strideX, y.pointer(),
      wrapper::BlasDataType::FromOpaqueValue(*typeY), strideY,
      wrapper::BlasDataType::FromOpaqueValue(*executionType));
}

static wrapper::BlasGemmAlgo BlasGemmAlgo(Attribute<int32_t> algo) {
  return wrapper::BlasGemmAlgo::FromOpaqueValue(*algo);
}

static Error BlasGemm(const GpuBlasHandle& handle, int32_t m, int32_t n,
                      int32_t k, AsyncValue* alpha, const GpuBuffer& A,
                      int32_t heightA, const GpuBuffer& B, int32_t heightB,
                      AsyncValue* beta, const GpuBuffer& C, int32_t heightC,
                      wrapper::BlasGemmAlgo algo,
                      // Needs to be sorted alphabetically by attribute name!
                      Attribute<int32_t> computeType, Attribute<int32_t> transA,
                      Attribute<int32_t> transB, Attribute<int32_t> typeA,
                      Attribute<int32_t> typeB, Attribute<int32_t> typeC) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  auto compute_type = wrapper::BlasComputeType::FromOpaqueValue(*computeType);
  auto alpha_ptr = GetScalePointer(alpha, compute_type);
  if (!alpha_ptr) return alpha_ptr.takeError();
  auto beta_ptr = GetScalePointer(beta, compute_type);
  if (!beta_ptr) return beta_ptr.takeError();

  return wrapper::BlasGemmEx(
      *current, handle.get(), wrapper::BlasOperation::FromOpaqueValue(*transA),
      wrapper::BlasOperation::FromOpaqueValue(*transB), m, n, k, *alpha_ptr,
      A.pointer(), wrapper::BlasDataType::FromOpaqueValue(*typeA), heightA,
      B.pointer(), wrapper::BlasDataType::FromOpaqueValue(*typeB), heightB,
      *beta_ptr, C.pointer(), wrapper::BlasDataType::FromOpaqueValue(*typeC),
      heightC, compute_type, algo);
}

static Error BlasGemmBatch(
    const GpuBlasHandle& handle, int32_t m, int32_t n, int32_t k,
    AsyncValue* alpha, const GpuBuffer& A, int32_t heightA, int64_t strideA,
    const GpuBuffer& B, int32_t heightB, int64_t strideB, AsyncValue* beta,
    const GpuBuffer& C, int32_t heightC, int64_t strideC, int32_t batchCount,
    wrapper::BlasGemmAlgo algo,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> computeType, Attribute<int32_t> transA,
    Attribute<int32_t> transB, Attribute<int32_t> typeA,
    Attribute<int32_t> typeB, Attribute<int32_t> typeC) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  auto compute_type = wrapper::BlasComputeType::FromOpaqueValue(*computeType);
  auto alpha_ptr = GetScalePointer(alpha, compute_type);
  if (!alpha_ptr) return alpha_ptr.takeError();
  auto beta_ptr = GetScalePointer(beta, compute_type);
  if (!beta_ptr) return beta_ptr.takeError();

  return wrapper::BlasGemmStridedBatchedEx(
      *current, handle.get(), wrapper::BlasOperation::FromOpaqueValue(*transA),
      wrapper::BlasOperation::FromOpaqueValue(*transB), m, n, k, *alpha_ptr,
      A.pointer(), wrapper::BlasDataType::FromOpaqueValue(*typeA), heightA,
      strideA, B.pointer(), wrapper::BlasDataType::FromOpaqueValue(*typeB),
      heightB, strideB, *beta_ptr, C.pointer(),
      wrapper::BlasDataType::FromOpaqueValue(*typeC), heightC, strideC,
      batchCount, compute_type, algo);
}

static Error BlasTrsmBatch(
    const GpuBlasHandle& handle, int32_t m, int32_t n, AsyncValue* alpha,
    const GpuBuffer& A, int32_t heightA, const GpuBuffer& B, int32_t heightB,
    int32_t batchCount,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> dataType, Attribute<int32_t> diagType,
    Attribute<int32_t> fillMode, Attribute<int32_t> sideMode,
    Attribute<int32_t> trans) {
  // TODO(hanbinyoon): Also support the ROCm function corresponding to
  // cublas<t>trsmBatched.
  auto platform = handle->platform();
  if (platform != wrapper::Platform::CUDA)
    return MakeStringError("Unsupported platform ", platform);

  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  cudaDataType data_type = wrapper::BlasDataType::FromOpaqueValue(*dataType);
  auto alpha_ptr = GetScalePointer(alpha, data_type);
  if (!alpha_ptr) return alpha_ptr.takeError();

  auto call = [&](auto dummy) {
    using T = decltype(dummy);
    auto pointer_array =
        handle.context()->AllocateHostPoolMemory<T*>(*current, 2 * batchCount);
    if (!pointer_array) return pointer_array.takeError();
    T** b_array = pointer_array->get().raw(platform);
    const T** a_array = const_cast<const T**>(b_array + batchCount);

    auto side_mode = wrapper::BlasSideMode::FromOpaqueValue(*sideMode);
    ptrdiff_t a_batch_stride = side_mode == CUBLAS_SIDE_LEFT ? m * m : n * n;
    ptrdiff_t b_batch_stride = m * n;
    const T* a_ptr = static_cast<const T*>(A.pointer().raw(platform));
    T* b_ptr = static_cast<T*>(A.pointer().raw(platform));
    for (int32_t i = 0; i < batchCount; ++i) {
      a_array[i] = a_ptr + i * a_batch_stride;
      b_array[i] = b_ptr + i * b_batch_stride;
    }

    wrapper::Pointer<const T*> a_array_ptr(a_array, platform);
    wrapper::Pointer<T*> b_array_ptr(b_array, platform);
    auto cast_alpha_ptr = static_cast<wrapper::Pointer<const T>>(*alpha_ptr);

    auto current = wrapper::CtxSetCurrent(handle.context()->get());
    if (!current) return current.takeError();

    return wrapper::CublasTrsmBatched(
        *current, handle.get(), side_mode,
        wrapper::BlasFillMode::FromOpaqueValue(*fillMode),
        wrapper::BlasOperation::FromOpaqueValue(*trans),
        wrapper::BlasDiagType::FromOpaqueValue(*diagType), m, n, cast_alpha_ptr,
        a_array_ptr, heightA, b_array_ptr, heightB, batchCount);
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

void RegisterGpuBlasKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_gpu.blas.create", TFRT_KERNEL(BlasCreate));
  kernel_reg->AddKernel("tfrt_gpu.blas.axpy",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(BlasAxpy));
  kernel_reg->AddKernel("tfrt_gpu.blas.gemm.algo", TFRT_KERNEL(BlasGemmAlgo));
  kernel_reg->AddKernel("tfrt_gpu.blas.gemm",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(BlasGemm));
  kernel_reg->AddKernel("tfrt_gpu.blas.gemm.batch",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(BlasGemmBatch));
  kernel_reg->AddKernel("tfrt_gpu.blas.trsm.batch",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(BlasTrsmBatch));
}
}  // namespace gpu
}  // namespace tfrt
