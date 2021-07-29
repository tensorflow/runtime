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

#include "kernels_detail.h"
#include "llvm/Support/Errc.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/cublas_wrapper.h"
#include "tfrt/gpu/wrapper/rocblas_wrapper.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/host_context/kernel_registry.h"

namespace tfrt {
namespace gpu {

static Expected<GpuBlasHandle> BlasCreate(Argument<GpuStream> stream) {
  auto current = wrapper::CtxSetCurrent(stream->context());
  if (!current) return current.takeError();
  auto handle = wrapper::BlasCreate(*current);
  if (!handle) return handle.takeError();
  if (auto error = wrapper::BlasSetStream(handle->get(), stream->get()))
    return std::move(error);
  return GpuBlasHandle(stream.ValueRef(), std::move(*handle));
}

template <typename T, cudaDataType cuda_type, rocblas_datatype rocm_type>
static llvm::Expected<wrapper::Pointer<T>> GetScalePointer(
    AsyncValue* value, wrapper::BlasDataType data_type) {
  if (data_type != cuda_type && data_type != rocm_type)
    return MakeStringError("unexpected argument type for ", data_type);
  return wrapper::Pointer<T>(&value->get<T>(), data_type.platform());
}

static llvm::Expected<wrapper::Pointer<void>> GetScalePointer(
    AsyncValue* value, wrapper::BlasDataType data_type) {
  if (value->IsType<float>())
    return GetScalePointer<float, CUDA_R_32F, rocblas_datatype_f32_r>(
        value, data_type);
  if (value->IsType<double>()) {
    return GetScalePointer<double, CUDA_R_64F, rocblas_datatype_f64_r>(
        value, data_type);
  }
  return MakeStringError("pointer type not supported");
}

static Error BlasAxpy(const GpuBlasHandle& handle, int32_t n, AsyncValue* alpha,
                      const GpuBuffer& x, int32_t strideX, const GpuBuffer& y,
                      int32_t strideY, Attribute<int32_t> executionType,
                      Attribute<int32_t> typeAlpha, Attribute<int32_t> typeX,
                      Attribute<int32_t> typeY) {
  auto current = wrapper::CtxSetCurrent(handle.context());
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
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();

  auto compute_type = wrapper::BlasDataType::FromOpaqueValue(*computeType);
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
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();

  auto compute_type = wrapper::BlasDataType::FromOpaqueValue(*computeType);
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

void RegisterGpuBlasKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_gpu.blas.create", TFRT_KERNEL(BlasCreate));
  kernel_reg->AddKernel("tfrt_gpu.blas.axpy",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(BlasAxpy));
  kernel_reg->AddKernel("tfrt_gpu.blas.gemm.algo", TFRT_KERNEL(BlasGemmAlgo));
  kernel_reg->AddKernel("tfrt_gpu.blas.gemm",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(BlasGemm));
  kernel_reg->AddKernel("tfrt_gpu.blas.gemm.batch",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(BlasGemmBatch));
}
}  // namespace gpu
}  // namespace tfrt
