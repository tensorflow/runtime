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
#include <cstdint>

#include "kernels.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/memory/gpu_buffer.h"
#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/cusolver_wrapper.h"
#include "tfrt/gpu/wrapper/solver_wrapper.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/host_context/sync_kernel_utils.h"

namespace tfrt {
namespace gpu {

// TODO(gkg) b/185513974 Need to follow-up with a change to make the solver
// handle be created for a specific stream, holding a ref-count to that stream,
// and remove the destroy op.

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

static llvm::Expected<wrapper::OwningSolverDnHandle> SolverDnCreate(
    const GpuContext& context) {
  auto current = wrapper::CtxSetCurrent(context.get());
  if (!current) return current.takeError();
  return wrapper::SolverDnCreate(current.get());
}

Error SolverDnSetStream(const wrapper::OwningSolverDnHandle& solver_handle,
                        const wrapper::OwningStream& stream) {
  return wrapper::SolverDnSetStream(solver_handle.get(), stream.get());
}

llvm::Expected<int32_t> SolverDnSpotrfBufferSize(
    const GpuContext& context,
    const wrapper::OwningSolverDnHandle& cusolver_handle, int32_t uplo,
    int32_t n, const RCReference<GpuCrtBuffer>& A, int32_t lda) {
  auto current = wrapper::CtxSetCurrent(context.get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnSpotrfBufferSize(
      current.get(), cusolver_handle.get(), *uplo_solver, n,
      wrapper::Pointer<float>(A->pointer()), lda);
}

llvm::Expected<int32_t> SolverDnDpotrfBufferSize(
    const GpuContext& context,
    const wrapper::OwningSolverDnHandle& cusolver_handle, int32_t uplo,
    int32_t n, const RCReference<GpuCrtBuffer>& A, int32_t lda) {
  auto current = wrapper::CtxSetCurrent(context.get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnDpotrfBufferSize(
      current.get(), cusolver_handle.get(), *uplo_solver, n,
      wrapper::Pointer<double>(A->pointer()), lda);
}

llvm::Expected<int32_t> SolverDnCpotrfBufferSize(
    const GpuContext& context,
    const wrapper::OwningSolverDnHandle& cusolver_handle, int32_t uplo,
    int32_t n, const RCReference<GpuCrtBuffer>& A, int32_t lda) {
  auto current = wrapper::CtxSetCurrent(context.get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnCpotrfBufferSize(
      current.get(), cusolver_handle.get(), *uplo_solver, n,
      wrapper::Pointer<cuComplex>(A->pointer()), lda);
}

llvm::Expected<int32_t> SolverDnZpotrfBufferSize(
    const GpuContext& context,
    const wrapper::OwningSolverDnHandle& cusolver_handle, int32_t uplo,
    int32_t n, const RCReference<GpuCrtBuffer>& A, int32_t lda) {
  auto current = wrapper::CtxSetCurrent(context.get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnZpotrfBufferSize(
      current.get(), cusolver_handle.get(), *uplo_solver, n,
      wrapper::Pointer<cuDoubleComplex>(A->pointer()), lda);
}

// These functions eventually need to make two separate calls to
// CusolverDn<t>potrf and corresponding ROCm function, as wrappers
// SolverDn<t>potrf for CUDA/ROCm is not feasible due to mismatch in APIs
// (Cusolver requires use of CusolverDn<t>potrf_bufferSize). Right now only
// CusolverDn<t>potrf calls are supported.
Error SolverDnSpotrf(const GpuContext& context,
                     const wrapper::OwningSolverDnHandle& cusolver_handle,
                     int32_t uplo, int32_t n,
                     const RCReference<GpuCrtBuffer>& A, int32_t lda,
                     const RCReference<GpuCrtBuffer>& Workspace, int32_t Lwork,
                     const RCReference<GpuCrtBuffer>& devInfo) {
  auto current = wrapper::CtxSetCurrent(context.get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnSpotrf(
      current.get(), cusolver_handle.get(), *uplo_solver, n,
      wrapper::Pointer<float>(A->pointer()), lda,
      wrapper::Pointer<float>(Workspace->pointer()), Lwork,
      wrapper::Pointer<int>(devInfo->pointer()));
}

Error SolverDnDpotrf(const GpuContext& context,
                     const wrapper::OwningSolverDnHandle& cusolver_handle,
                     int32_t uplo, int32_t n,
                     const RCReference<GpuCrtBuffer>& A, int32_t lda,
                     const RCReference<GpuCrtBuffer>& Workspace, int32_t Lwork,
                     const RCReference<GpuCrtBuffer>& devInfo) {
  auto current = wrapper::CtxSetCurrent(context.get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnDpotrf(
      current.get(), cusolver_handle.get(), *uplo_solver, n,
      wrapper::Pointer<double>(A->pointer()), lda,
      wrapper::Pointer<double>(Workspace->pointer()), Lwork,
      wrapper::Pointer<int>(devInfo->pointer()));
}

Error SolverDnCpotrf(const GpuContext& context,
                     const wrapper::OwningSolverDnHandle& cusolver_handle,
                     int32_t uplo, int32_t n,
                     const RCReference<GpuCrtBuffer>& A, int32_t lda,
                     const RCReference<GpuCrtBuffer>& Workspace, int32_t Lwork,
                     const RCReference<GpuCrtBuffer>& devInfo) {
  auto current = wrapper::CtxSetCurrent(context.get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnCpotrf(
      current.get(), cusolver_handle.get(), *uplo_solver, n,
      wrapper::Pointer<cuComplex>(A->pointer()), lda,
      wrapper::Pointer<cuComplex>(Workspace->pointer()), Lwork,
      wrapper::Pointer<int>(devInfo->pointer()));
}

Error SolverDnZpotrf(const GpuContext& context,
                     const wrapper::OwningSolverDnHandle& cusolver_handle,
                     int32_t uplo, int32_t n,
                     const RCReference<GpuCrtBuffer>& A, int32_t lda,
                     const RCReference<GpuCrtBuffer>& Workspace, int32_t Lwork,
                     const RCReference<GpuCrtBuffer>& devInfo) {
  auto current = wrapper::CtxSetCurrent(context.get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnZpotrf(
      current.get(), cusolver_handle.get(), *uplo_solver, n,
      wrapper::Pointer<cuDoubleComplex>(A->pointer()), lda,
      wrapper::Pointer<cuDoubleComplex>(Workspace->pointer()), Lwork,
      wrapper::Pointer<int>(devInfo->pointer()));
}

#define TFRT_WITH_CHAIN_RESULT(sync_func) \
  internal::WithChainResult<decltype(&sync_func), &sync_func>::Invoke

void RegisterCudaSolverKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_cuda.solver.create", TFRT_KERNEL(SolverDnCreate));
  kernel_reg->AddKernel("tfrt_cuda.solver.set_stream",
                        TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(SolverDnSetStream)));
  kernel_reg->AddKernel(
      "tfrt_cuda.solver.dn.s.portf.buffer_size",
      TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(SolverDnSpotrfBufferSize)));
  kernel_reg->AddKernel(
      "tfrt_cuda.solver.dn.d.portf.buffer_size",
      TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(SolverDnDpotrfBufferSize)));
  kernel_reg->AddKernel(
      "tfrt_cuda.solver.dn.c.portf.buffer_size",
      TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(SolverDnCpotrfBufferSize)));
  kernel_reg->AddKernel(
      "tfrt_cuda.solver.dn.z.portf.buffer_size",
      TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(SolverDnZpotrfBufferSize)));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.s.portf",
                        TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(SolverDnSpotrf)));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.d.portf",
                        TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(SolverDnDpotrf)));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.c.portf",
                        TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(SolverDnCpotrf)));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.z.portf",
                        TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(SolverDnZpotrf)));
}

}  // namespace gpu
}  // namespace tfrt
