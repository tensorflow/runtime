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

//===- blas_kernels.cc - BLAS kernels ---------------------------*- C++ -*-===//
//
// This file defines the C++ functions that implement the BLAS kernels provided
// by the TFRT CUDA runtime.
//
//===----------------------------------------------------------------------===//
#include "tfrt/gpu/memory/gpu_buffer.h"
#include "tfrt/gpu/stream/blas_wrapper.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {
namespace cuda {
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
  gpu::stream::Pointer<const float> alpha_ptr(&(*alpha), context->platform());

  llvm::Error error = gpu::stream::BlasSaxpy(
      *current, cublas_handle->get(), *n, alpha_ptr,
      gpu::stream::Pointer<const float>(x->get()->pointer()), *incx,
      gpu::stream::Pointer<float>(y->get()->pointer()), *incy);
  if (error) return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

void RegisterCudaBlasKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("cuda.blas.create", TFRT_KERNEL(BlasCreate));
  kernel_reg->AddKernel("cuda.blas.axpy.f32", TFRT_KERNEL(BlasSaxpy));
  kernel_reg->AddKernel("cuda.blas.set_stream", TFRT_KERNEL(BlasSetStream));
}
}  // namespace cuda
}  // namespace tfrt
