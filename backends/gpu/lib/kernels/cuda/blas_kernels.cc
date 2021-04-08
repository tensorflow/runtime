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

// BLAS kernels
//
// This file defines the C++ functions that implement the BLAS kernels provided
// by the TFRT CUDA runtime.
#include "llvm/Support/Errc.h"
#include "tfrt/gpu/memory/gpu_buffer.h"
#include "tfrt/gpu/stream/blas_wrapper.h"
#include "tfrt/gpu/stream/cublas_wrapper.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {
namespace cuda {

using ::tfrt::gpu::stream::Pointer;
//
// Overloaded helpers for the REPORT_ERROR macro. Allows the macro to use either
// strings or llvm::Errors.
static void ReportErrorInternal(KernelErrorHandler error_handler,
                                string_view error_message, string_view file,
                                int line) {
  return error_handler.ReportError(file, ':', line, ' ', error_message);
}

static void ReportErrorInternal(KernelErrorHandler error_handler,
                                llvm::Error error, string_view file, int line) {
  llvm::handleAllErrors(std::move(error), [&](const llvm::ErrorInfoBase& info) {
    ReportErrorInternal(error_handler, info.message(), file, line);
  });
}

#define REPORT_ERROR(error_handler, error) \
  ReportErrorInternal(error_handler, error, __FILE__, __LINE__)

static void BlasCreate(Argument<gpu::stream::Context> context,
                       Result<gpu::stream::OwningBlasHandle> out_cublas_handle,
                       KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  auto cublas_handle = gpu::stream::BlasCreate(*current);
  if (!cublas_handle) return REPORT_ERROR(handler, cublas_handle.takeError());
  out_cublas_handle.Emplace(std::move(*cublas_handle));
}

static void BlasSetStream(Argument<gpu::stream::OwningBlasHandle> cublas_handle,
                          Argument<gpu::stream::OwningStream> stream,
                          Argument<Chain> in_chain, Result<Chain> out_chain,
                          KernelErrorHandler handler) {
  llvm::Error error =
      gpu::stream::BlasSetStream(cublas_handle->get(), stream->get());
  if (error) return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

static void BlasSaxpy(Argument<gpu::stream::Context> context,
                      Argument<gpu::stream::OwningBlasHandle> cublas_handle,
                      Argument<int32_t> n, Argument<float> alpha,
                      Argument<RCReference<gpu::GpuBuffer>> x,
                      Argument<int32_t> incx,
                      Argument<RCReference<gpu::GpuBuffer>> y,
                      Argument<int32_t> incy, Argument<Chain> in_chain,
                      Result<Chain> out_chain, KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  Pointer<const float> alpha_ptr(&(*alpha), context->platform());

  llvm::Error error =
      gpu::stream::BlasSaxpy(*current, cublas_handle->get(), *n, alpha_ptr,
                             Pointer<const float>(x->get()->pointer()), *incx,
                             Pointer<float>(y->get()->pointer()), *incy);
  if (error) return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

static gpu::stream::BlasOperation ToBlasOperation(bool transpose) {
  return transpose ? gpu::stream::BlasOperation::kTranspose
                   : gpu::stream::BlasOperation::kNone;
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

static void BlasSgemm(Argument<gpu::stream::Context> context,
                      Argument<gpu::stream::OwningBlasHandle> cublas_handle,
                      Argument<int32_t> m, Argument<int32_t> n,
                      Argument<int32_t> k, Argument<float> alpha,
                      Argument<RCReference<gpu::GpuBuffer>> A,
                      Argument<int32_t> lda,
                      Argument<RCReference<gpu::GpuBuffer>> B,
                      Argument<int32_t> ldb, Argument<float> beta,
                      Argument<RCReference<gpu::GpuBuffer>> C,
                      Argument<int32_t> ldc, Argument<Chain> in_chain,
                      Result<Chain> out_chain, Attribute<bool> transa,
                      Attribute<bool> transb, KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  Pointer<const float> alpha_ptr(&(*alpha), context->platform());
  Pointer<const float> beta_ptr(&(*beta), context->platform());

  llvm::Error error = gpu::stream::BlasSgemm(
      *current, cublas_handle->get(), ToBlasOperation(*transa),
      ToBlasOperation(*transb), *m, *n, *k, alpha_ptr,
      Pointer<const float>(A->get()->pointer()), *lda,
      Pointer<const float>(B->get()->pointer()), *ldb, beta_ptr,
      Pointer<float>(C->get()->pointer()), *ldc);
  if (error) return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

// This function eventually need to make two separate calls to CublasGemmEx and
// corresponding ROCm function, as wrapper BlassGemmEx for CUDA/ROCm is not
// feasible due to mismatch in APIs (algo specification parameter).  Right now
// only CublasGemmEx call is supported.
static void BlasGemmEx(Argument<gpu::stream::Context> context,
                       Argument<gpu::stream::OwningBlasHandle> cublas_handle,

                       Argument<int32_t> m, Argument<int32_t> n,
                       Argument<int32_t> k, Argument<float> alpha,
                       Argument<RCReference<gpu::GpuBuffer>> A,
                       Argument<int32_t> Atype, Argument<int32_t> lda,
                       Argument<RCReference<gpu::GpuBuffer>> B,
                       Argument<int32_t> Btype, Argument<int32_t> ldb,
                       Argument<float> beta,
                       Argument<RCReference<gpu::GpuBuffer>> C,
                       Argument<int32_t> Ctype, Argument<int32_t> ldc,
                       Argument<int32_t> computeType, Argument<int32_t> algo,
                       Argument<Chain> in_chain, Result<Chain> out_chain,
                       Attribute<bool> transa, Attribute<bool> transb,
                       KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  Pointer<const float> alpha_ptr(&(*alpha), context->platform());
  Pointer<const float> beta_ptr(&(*beta), context->platform());

  auto transa_cublas = ToCublas(ToBlasOperation(*transa));
  auto transb_cublas = ToCublas(ToBlasOperation(*transb));

  auto Atype_blas = SafeIntToCublasDataType(*Atype);
  if (!Atype_blas) return REPORT_ERROR(handler, Atype_blas.takeError());

  auto Btype_blas = SafeIntToCublasDataType(*Btype);
  if (!Btype_blas) return REPORT_ERROR(handler, Btype_blas.takeError());

  auto Ctype_blas = SafeIntToCublasDataType(*Ctype);
  if (!Ctype_blas) return REPORT_ERROR(handler, Ctype_blas.takeError());

  auto computeType_blas = SafeIntToCublasDataType(*computeType);
  if (!computeType_blas)
    return REPORT_ERROR(handler, computeType_blas.takeError());

  auto algo_blas = SafeIntToCublasGemmAlgo(*algo);
  if (!algo_blas) return REPORT_ERROR(handler, algo_blas.takeError());

  llvm::Error error = gpu::stream::CublasGemmEx(
      *current, cublas_handle->get(), transa_cublas, transb_cublas, *m, *n, *k,
      alpha_ptr, Pointer<const float>(A->get()->pointer()), *Atype_blas, *lda,
      Pointer<const float>(B->get()->pointer()), *Btype_blas, *ldb, beta_ptr,
      Pointer<float>(C->get()->pointer()), *Ctype_blas, *ldc, *computeType_blas,
      *algo_blas);
  if (error) return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

// Note: return type should really just be llvm::Error, but
// TfrtKernelImpl::HandleReturn does not overload for that. Until we have
// decided whether async kernels are allowed to have no return values (not even
// a chain), simply return an empty tuple.
static llvm::Expected<std::tuple<>> BlasSyncGemmEx(
    gpu::stream::Context context,
    const gpu::stream::OwningBlasHandle& cublas_handle,
    const gpu::stream::OwningStream& stream, int32_t m, int32_t n, int32_t k,
    float alpha, const RCReference<gpu::GpuBuffer>& A, int32_t Atype,
    int32_t lda, const RCReference<gpu::GpuBuffer>& B, int32_t Btype,
    int32_t ldb, float beta, const RCReference<gpu::GpuBuffer>& C,
    int32_t Ctype, int32_t ldc, int32_t algo, Attribute<int32_t> computeType,
    Attribute<bool> transa, Attribute<bool> transb) {
  auto current = gpu::stream::CtxSetCurrent(context);
  if (!current) return current.takeError();
  Pointer<const float> alpha_ptr(&alpha, context.platform());
  Pointer<const float> beta_ptr(&beta, context.platform());

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

  if (auto error =
          gpu::stream::BlasSetStream(cublas_handle.get(), stream.get()))
    return std::move(error);

  if (auto error = gpu::stream::CublasGemmEx(
          *current, cublas_handle.get(), transa_cublas, transb_cublas, m, n, k,
          alpha_ptr, Pointer<const float>(A->pointer()), *Atype_blas, lda,
          Pointer<const float>(B->pointer()), *Btype_blas, ldb, beta_ptr,
          Pointer<float>(C->pointer()), *Ctype_blas, ldc, *computeType_blas,
          *algo_blas))
    return std::move(error);

  return std::make_tuple();
}

static void BlasGemmStridedBatchedEx(
    Argument<gpu::stream::Context> context,
    Argument<gpu::stream::OwningBlasHandle> cublas_handle, Argument<int32_t> m,
    Argument<int32_t> n, Argument<int32_t> k, Argument<float> alpha,
    Argument<RCReference<gpu::GpuBuffer>> A, Argument<int32_t> Atype,
    Argument<int32_t> lda, Argument<int64_t> strideA,
    Argument<RCReference<gpu::GpuBuffer>> B, Argument<int32_t> Btype,
    Argument<int32_t> ldb, Argument<int64_t> strideB, Argument<float> beta,
    Argument<RCReference<gpu::GpuBuffer>> C, Argument<int32_t> Ctype,
    Argument<int32_t> ldc, Argument<int64_t> strideC,
    Argument<int32_t> batch_count, Argument<int32_t> computeType,
    Argument<int32_t> algo, Argument<Chain> in_chain, Result<Chain> out_chain,
    Attribute<bool> transa, Attribute<bool> transb,
    KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  Pointer<const float> alpha_ptr(&(*alpha), context->platform());
  Pointer<const float> beta_ptr(&(*beta), context->platform());

  auto transa_cublas = ToCublas(ToBlasOperation(*transa));
  auto transb_cublas = ToCublas(ToBlasOperation(*transb));

  auto Atype_blas = SafeIntToCublasDataType(*Atype);
  if (!Atype_blas) return REPORT_ERROR(handler, Atype_blas.takeError());

  auto Btype_blas = SafeIntToCublasDataType(*Btype);
  if (!Btype_blas) return REPORT_ERROR(handler, Btype_blas.takeError());

  auto Ctype_blas = SafeIntToCublasDataType(*Ctype);
  if (!Ctype_blas) return REPORT_ERROR(handler, Ctype_blas.takeError());

  auto computeType_blas = SafeIntToCublasDataType(*computeType);
  if (!computeType_blas)
    return REPORT_ERROR(handler, computeType_blas.takeError());

  auto algo_blas = SafeIntToCublasGemmAlgo(*algo);
  if (!algo_blas) return REPORT_ERROR(handler, algo_blas.takeError());

  llvm::Error error = gpu::stream::CublasGemmStridedBatchedEx(
      *current, cublas_handle->get(), transa_cublas, transb_cublas, *m, *n, *k,
      alpha_ptr, Pointer<const float>(A->get()->pointer()), *Atype_blas, *lda,
      *strideA, Pointer<const float>(B->get()->pointer()), *Btype_blas, *ldb,
      *strideB, beta_ptr, Pointer<float>(C->get()->pointer()), *Ctype_blas,
      *ldc, *strideC, *batch_count, *computeType_blas, *algo_blas);
  if (error) return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

void RegisterCudaBlasKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_cuda.blas.create", TFRT_KERNEL(BlasCreate));
  kernel_reg->AddKernel("tfrt_cuda.blas.set_stream",
                        TFRT_KERNEL(BlasSetStream));
  kernel_reg->AddKernel("tfrt_cuda.blas.axpy.f32", TFRT_KERNEL(BlasSaxpy));
  kernel_reg->AddKernel("tfrt_cuda.blas.gemm.f32", TFRT_KERNEL(BlasSgemm));
  kernel_reg->AddKernel("tfrt_cuda.blas.gemm.ex", TFRT_KERNEL(BlasGemmEx));
  kernel_reg->AddKernel("tfrt_cuda.blas.gemm.strided.batched.ex",
                        TFRT_KERNEL(BlasGemmStridedBatchedEx));
  kernel_reg->AddKernel("tfrt_cuda.blas.sync.gemm_ex",
                        TFRT_KERNEL(BlasSyncGemmEx));
}
}  // namespace cuda
}  // namespace tfrt
