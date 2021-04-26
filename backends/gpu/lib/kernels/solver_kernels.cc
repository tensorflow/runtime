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

// This file implements the tfrt_gpu.solver kernels, at the moment for CUDA
// only. Support for ROCm still needs to be implemented.
#include <cstdint>

#include "kernels_detail.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/cusolver_wrapper.h"
#include "tfrt/gpu/wrapper/solver_wrapper.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/kernel_registry.h"

namespace tfrt {
namespace gpu {

static llvm::Expected<cublasFillMode_t> SafeIntToCublasFillMode(
    int32_t operation) {
  auto cublas_fill_mode = static_cast<cublasFillMode_t>(operation);
  if ((cublas_fill_mode > cublasFillMode_t::CUBLAS_FILL_MODE_FULL) ||
      (cublas_fill_mode < cublasFillMode_t::CUBLAS_FILL_MODE_LOWER)) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Invalid cublasFillMode_t value: %d",
                                   operation);
  }
  return cublas_fill_mode;
}

static llvm::Expected<GpuSolverHandle> SolverCreate(
    Argument<GpuStream> stream) {
  auto current = wrapper::CtxSetCurrent(stream->context());
  if (!current) return current.takeError();
  auto handle = wrapper::SolverCreate(current->platform());
  if (!handle) return handle.takeError();
  if (auto error = wrapper::SolverSetStream(handle->get(), stream->get()))
    return std::move(error);
  return GpuSolverHandle(stream.ValueRef(), std::move(*handle));
}

template <typename T>
llvm::Expected<int32_t> SolverPotrfBufferSize(const GpuSolverHandle& handle,
                                              int32_t uplo, int32_t n,
                                              const GpuBuffer& A, int32_t lda) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();

  auto cublas_uplo = SafeIntToCublasFillMode(uplo);
  if (!cublas_uplo) return cublas_uplo.takeError();

  return wrapper::CusolverDnPotrfBufferSize(
      current.get(), handle.get(), *cublas_uplo, n,
      wrapper::Pointer<T>(A.pointer()), lda);
}

// These functions eventually need to make two separate calls to
// CusolverDn<t>potrf and corresponding ROCm function, as wrappers
// SolverPotrf for CUDA/ROCm is not feasible due to mismatch in APIs
// (Cusolver requires use of CusolverDn<t>potrf_bufferSize). Right now only
// CusolverDnPotrf calls are supported.
template <typename T>
Error SolverPotrf(const GpuSolverHandle& handle, int32_t uplo, int32_t n,
                  const GpuBuffer& A, int32_t lda, const GpuBuffer& Workspace,
                  int32_t Lwork, const GpuBuffer& devInfo) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();

  auto cublas_uplo = SafeIntToCublasFillMode(uplo);
  if (!cublas_uplo) return cublas_uplo.takeError();

  return wrapper::CusolverDnPotrf(current.get(), handle.get(), *cublas_uplo, n,
                                  wrapper::Pointer<T>(A.pointer()), lda,
                                  wrapper::Pointer<T>(Workspace.pointer()),
                                  Lwork,
                                  wrapper::Pointer<int>(devInfo.pointer()));
}

void RegisterGpuSolverKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_cuda.solver.create", TFRT_KERNEL(SolverCreate));
  kernel_reg->AddKernel(
      "tfrt_cuda.solver.dn.s.portf.buffer_size",
      TFRT_KERNEL_WITH_CHAIN_RESULT(SolverPotrfBufferSize<float>));
  kernel_reg->AddKernel(
      "tfrt_cuda.solver.dn.d.portf.buffer_size",
      TFRT_KERNEL_WITH_CHAIN_RESULT(SolverPotrfBufferSize<double>));
  kernel_reg->AddKernel(
      "tfrt_cuda.solver.dn.c.portf.buffer_size",
      TFRT_KERNEL_WITH_CHAIN_RESULT(SolverPotrfBufferSize<cuComplex>));
  kernel_reg->AddKernel(
      "tfrt_cuda.solver.dn.z.portf.buffer_size",
      TFRT_KERNEL_WITH_CHAIN_RESULT(SolverPotrfBufferSize<cuDoubleComplex>));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.s.portf",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(SolverPotrf<float>));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.d.portf",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(SolverPotrf<double>));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.c.portf",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(SolverPotrf<cuComplex>));
  kernel_reg->AddKernel(
      "tfrt_cuda.solver.dn.z.portf",
      TFRT_KERNEL_WITH_CHAIN_RESULT(SolverPotrf<cuDoubleComplex>));
}

}  // namespace gpu
}  // namespace tfrt
