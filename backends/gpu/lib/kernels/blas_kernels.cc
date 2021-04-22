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
#include "kernels_detail.h"
#include "llvm/Support/Errc.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/cublas_wrapper.h"
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

static Error BlasSaxpy(const GpuBlasHandle& handle, int32_t n, float alpha,
                       const GpuBuffer& x, int32_t incx, const GpuBuffer& y,
                       int32_t incy) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();
  wrapper::Pointer<const float> alpha_ptr(&alpha, handle->platform());

  return wrapper::BlasSaxpy(*current, handle.get(), n, alpha_ptr,
                            wrapper::Pointer<const float>(x.pointer()), incx,
                            wrapper::Pointer<float>(y.pointer()), incy);
}

static wrapper::BlasOperation ToBlasOperation(bool transpose) {
  return transpose ? wrapper::BlasOperation::kTranspose
                   : wrapper::BlasOperation::kNone;
}

static llvm::Expected<cudaDataType> SafeIntToCublasDataType(int32_t data_type) {
  auto cublas_data_type = static_cast<cudaDataType>(data_type);
  if ((cublas_data_type > cudaDataType::CUDA_C_32U) ||
      (cublas_data_type < cudaDataType::CUDA_R_32F)) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Invalid CublasDataType value: %d",
                                   data_type);
  }
  return cublas_data_type;
}

static llvm::Expected<cublasGemmAlgo_t> SafeIntToCublasGemmAlgo(int32_t algo) {
  auto cublas_algo = static_cast<cublasGemmAlgo_t>(algo);
  if ((cublas_algo > cublasGemmAlgo_t::CUBLAS_GEMM_ALGO15_TENSOR_OP) ||
      (cublas_algo < cublasGemmAlgo_t::CUBLAS_GEMM_DFALT) ||
      ((cublas_algo > cublasGemmAlgo_t::CUBLAS_GEMM_ALGO23) &&
       (cublas_algo < cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP))) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Invalid CublasGemmAlgo value: %d", algo);
  }
  return cublas_algo;
}

static Error BlasSgemm(const GpuBlasHandle& handle, int32_t m, int32_t n,
                       int32_t k, float alpha, const GpuBuffer& A, int32_t lda,
                       const GpuBuffer& B, int32_t ldb, float beta,
                       const GpuBuffer& C, int32_t ldc, Attribute<bool> transa,
                       Attribute<bool> transb) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();
  wrapper::Pointer<const float> alpha_ptr(&alpha, handle->platform());
  wrapper::Pointer<const float> beta_ptr(&beta, handle->platform());

  return wrapper::BlasSgemm(*current, handle.get(), ToBlasOperation(*transa),
                            ToBlasOperation(*transb), m, n, k, alpha_ptr,
                            wrapper::Pointer<const float>(A.pointer()), lda,
                            wrapper::Pointer<const float>(B.pointer()), ldb,
                            beta_ptr, wrapper::Pointer<float>(C.pointer()),
                            ldc);
}

// This function eventually need to make two separate calls to CublasGemmEx and
// corresponding ROCm function, as wrapper BlassGemmEx for CUDA/ROCm is not
// feasible due to mismatch in APIs (algo specification parameter).  Right now
// only CublasGemmEx call is supported.
static Error BlasGemmEx(const GpuBlasHandle& handle, int32_t m, int32_t n,
                        int32_t k, float alpha, const GpuBuffer& A,
                        int32_t Atype, int32_t lda, const GpuBuffer& B,
                        int32_t Btype, int32_t ldb, float beta,
                        const GpuBuffer& C, int32_t Ctype, int32_t ldc,
                        int32_t computeType, int32_t algo,
                        Attribute<bool> transa, Attribute<bool> transb) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();
  wrapper::Pointer<const float> alpha_ptr(&alpha, handle->platform());
  wrapper::Pointer<const float> beta_ptr(&beta, handle->platform());

  auto transa_cublas = ToCublas(ToBlasOperation(*transa));
  auto transb_cublas = ToCublas(ToBlasOperation(*transb));

  auto Atype_blas = SafeIntToCublasDataType(Atype);
  if (!Atype_blas) return Atype_blas.takeError();

  auto Btype_blas = SafeIntToCublasDataType(Btype);
  if (!Btype_blas) return Btype_blas.takeError();

  auto Ctype_blas = SafeIntToCublasDataType(Ctype);
  if (!Ctype_blas) return Ctype_blas.takeError();

  auto computeType_blas = SafeIntToCublasDataType(computeType);
  if (!computeType_blas) return computeType_blas.takeError();

  auto algo_blas = SafeIntToCublasGemmAlgo(algo);
  if (!algo_blas) return algo_blas.takeError();

  return wrapper::CublasGemmEx(
      *current, handle.get(), transa_cublas, transb_cublas, m, n, k, alpha_ptr,
      wrapper::Pointer<const float>(A.pointer()), *Atype_blas, lda,
      wrapper::Pointer<const float>(B.pointer()), *Btype_blas, ldb, beta_ptr,
      wrapper::Pointer<float>(C.pointer()), *Ctype_blas, ldc, *computeType_blas,
      *algo_blas);
}

// Note: return type should really just be llvm::Error, but
// TfrtKernelImpl::HandleReturn does not overload for that. Until we have
// decided whether async kernels are allowed to have no return values (not even
// a chain), simply return an empty tuple.
static Error BlasSyncGemmEx(const GpuBlasHandle& handle, int32_t m, int32_t n,
                            int32_t k, float alpha, const GpuBuffer& A,
                            int32_t Atype, int32_t lda, const GpuBuffer& B,
                            int32_t Btype, int32_t ldb, float beta,
                            const GpuBuffer& C, int32_t Ctype, int32_t ldc,
                            int32_t algo, Attribute<int32_t> computeType,
                            Attribute<bool> transa, Attribute<bool> transb) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();
  wrapper::Pointer<const float> alpha_ptr(&alpha, handle->platform());
  wrapper::Pointer<const float> beta_ptr(&beta, handle->platform());

  auto transa_cublas = ToCublas(ToBlasOperation(*transa));
  auto transb_cublas = ToCublas(ToBlasOperation(*transb));

  auto Atype_blas = SafeIntToCublasDataType(Atype);
  if (!Atype_blas) return Atype_blas.takeError();

  auto Btype_blas = SafeIntToCublasDataType(Btype);
  if (!Btype_blas) return Btype_blas.takeError();

  auto Ctype_blas = SafeIntToCublasDataType(Ctype);
  if (!Ctype_blas) return Ctype_blas.takeError();

  auto computeType_blas = SafeIntToCublasDataType(computeType.get());
  if (!computeType_blas) return computeType_blas.takeError();

  auto algo_blas = SafeIntToCublasGemmAlgo(algo);
  if (!algo_blas) return algo_blas.takeError();

  return wrapper::CublasGemmEx(
      *current, handle.get(), transa_cublas, transb_cublas, m, n, k, alpha_ptr,
      wrapper::Pointer<const float>(A.pointer()), *Atype_blas, lda,
      wrapper::Pointer<const float>(B.pointer()), *Btype_blas, ldb, beta_ptr,
      wrapper::Pointer<float>(C.pointer()), *Ctype_blas, ldc, *computeType_blas,
      *algo_blas);
}

static Error BlasGemmStridedBatchedEx(
    const GpuBlasHandle& handle, int32_t m, int32_t n, int32_t k, float alpha,
    const GpuBuffer& A, int32_t Atype, int32_t lda, int64_t strideA,
    const GpuBuffer& B, int32_t Btype, int32_t ldb, int64_t strideB, float beta,
    const GpuBuffer& C, int32_t Ctype, int32_t ldc, int64_t strideC,
    int32_t batch_count, int32_t computeType, int32_t algo,
    Attribute<bool> transa, Attribute<bool> transb) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();
  wrapper::Pointer<const float> alpha_ptr(&alpha, handle->platform());
  wrapper::Pointer<const float> beta_ptr(&beta, handle->platform());

  auto transa_cublas = ToCublas(ToBlasOperation(*transa));
  auto transb_cublas = ToCublas(ToBlasOperation(*transb));

  auto Atype_blas = SafeIntToCublasDataType(Atype);
  if (!Atype_blas) return Atype_blas.takeError();

  auto Btype_blas = SafeIntToCublasDataType(Btype);
  if (!Btype_blas) return Btype_blas.takeError();

  auto Ctype_blas = SafeIntToCublasDataType(Ctype);
  if (!Ctype_blas) return Ctype_blas.takeError();

  auto computeType_blas = SafeIntToCublasDataType(computeType);
  if (!computeType_blas) return computeType_blas.takeError();

  auto algo_blas = SafeIntToCublasGemmAlgo(algo);
  if (!algo_blas) return algo_blas.takeError();

  return wrapper::CublasGemmStridedBatchedEx(
      *current, handle.get(), transa_cublas, transb_cublas, m, n, k, alpha_ptr,
      wrapper::Pointer<const float>(A.pointer()), *Atype_blas, lda, strideA,
      wrapper::Pointer<const float>(B.pointer()), *Btype_blas, ldb, strideB,
      beta_ptr, wrapper::Pointer<float>(C.pointer()), *Ctype_blas, ldc, strideC,
      batch_count, *computeType_blas, *algo_blas);
}

void RegisterGpuBlasKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_gpu.blas.create", TFRT_KERNEL(BlasCreate));
  kernel_reg->AddKernel("tfrt_gpu.blas.axpy.f32",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(BlasSaxpy));
  kernel_reg->AddKernel("tfrt_gpu.blas.gemm.f32",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(BlasSgemm));
  kernel_reg->AddKernel("tfrt_gpu.blas.gemm.ex",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(BlasGemmEx));
  kernel_reg->AddKernel(
      "tfrt_gpu.blas.gemm.strided.batched.ex",
      TFRT_KERNEL_WITH_CHAIN_RESULT(BlasGemmStridedBatchedEx));
  kernel_reg->AddKernel("tfrt_gpu.blas.sync.gemm_ex",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(BlasSyncGemmEx));
}
}  // namespace gpu
}  // namespace tfrt
