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

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/memory/gpu_buffer.h"
#include "tfrt/gpu/stream/blas_wrapper.h"
#include "tfrt/gpu/stream/cusolver_wrapper.h"
#include "tfrt/gpu/stream/solver_wrapper.h"
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
    Argument<GpuContext> context, Argument<Chain> in_chain) {
  auto current = wrapper::CtxSetCurrent(context.get().get());
  if (!current) return current.takeError();
  return wrapper::SolverDnCreate(current.get());
}

Expected<Chain> SolverDnSetStream(
    const wrapper::OwningSolverDnHandle& solver_handle,
    const wrapper::OwningStream& stream, Argument<Chain> in_chain) {
  auto result = wrapper::SolverDnSetStream(solver_handle.get(), stream.get());
  if (!result) return std::move(result);
  return in_chain.get();
}

llvm::Expected<int32_t> SolverDnSpotrfBufferSize(
    Argument<GpuContext> context,
    const wrapper::OwningSolverDnHandle& cusolver_handle,
    Argument<int32_t> uplo, Argument<int32_t> n,
    Argument<RCReference<GpuBuffer>> A, Argument<int32_t> lda,
    Argument<Chain> in_chain) {
  auto current = wrapper::CtxSetCurrent(context.get().get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(*uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnSpotrfBufferSize(
      current.get(), cusolver_handle.get(), *uplo_solver, *n,
      wrapper::Pointer<float>(A->get()->pointer()), *lda);
}

llvm::Expected<int32_t> SolverDnDpotrfBufferSize(
    Argument<GpuContext> context,
    const wrapper::OwningSolverDnHandle& cusolver_handle,
    Argument<int32_t> uplo, Argument<int32_t> n,
    Argument<RCReference<GpuBuffer>> A, Argument<int32_t> lda,
    Argument<Chain> in_chain) {
  auto current = wrapper::CtxSetCurrent(context.get().get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(*uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnDpotrfBufferSize(
      current.get(), cusolver_handle.get(), *uplo_solver, *n,
      wrapper::Pointer<double>(A->get()->pointer()), *lda);
}

llvm::Expected<int32_t> SolverDnCpotrfBufferSize(
    Argument<GpuContext> context,
    const wrapper::OwningSolverDnHandle& cusolver_handle,
    Argument<int32_t> uplo, Argument<int32_t> n,
    Argument<RCReference<GpuBuffer>> A, Argument<int32_t> lda,
    Argument<Chain> in_chain) {
  auto current = wrapper::CtxSetCurrent(context.get().get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(*uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnCpotrfBufferSize(
      current.get(), cusolver_handle.get(), *uplo_solver, *n,
      wrapper::Pointer<cuComplex>(A->get()->pointer()), *lda);
}

llvm::Expected<int32_t> SolverDnZpotrfBufferSize(
    Argument<GpuContext> context,
    const wrapper::OwningSolverDnHandle& cusolver_handle,
    Argument<int32_t> uplo, Argument<int32_t> n,
    Argument<RCReference<GpuBuffer>> A, Argument<int32_t> lda,
    Argument<Chain> in_chain) {
  auto current = wrapper::CtxSetCurrent(context.get().get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(*uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  return wrapper::CusolverDnZpotrfBufferSize(
      current.get(), cusolver_handle.get(), *uplo_solver, *n,
      wrapper::Pointer<cuDoubleComplex>(A->get()->pointer()), *lda);
}

// These functions eventually need to make two separate calls to
// CusolverDn<t>potrf and corresponding ROCm function, as wrappers
// SolverDn<t>potrf for CUDA/ROCm is not feasible due to mismatch in APIs
// (Cusolver requires use of CusolverDn<t>potrf_bufferSize). Right now only
// CusolverDn<t>potrf calls are supported.
Expected<Chain> SolverDnSpotrf(
    Argument<GpuContext> context,
    const wrapper::OwningSolverDnHandle& cusolver_handle,
    Argument<int32_t> uplo, Argument<int32_t> n,
    Argument<RCReference<GpuBuffer>> A, Argument<int32_t> lda,
    Argument<RCReference<GpuBuffer>> Workspace, Argument<int32_t> Lwork,
    Argument<RCReference<GpuBuffer>> devInfo, Argument<Chain> in_chain) {
  auto current = wrapper::CtxSetCurrent(context.get().get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(*uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  auto result = wrapper::CusolverDnSpotrf(
      current.get(), cusolver_handle.get(), *uplo_solver, *n,
      wrapper::Pointer<float>(A->get()->pointer()), *lda,
      wrapper::Pointer<float>(Workspace->get()->pointer()), *Lwork,
      wrapper::Pointer<int>(devInfo->get()->pointer()));
  if (!result) return std::move(result);
  return in_chain.get();
}

Expected<Chain> SolverDnDpotrf(
    Argument<GpuContext> context,
    const wrapper::OwningSolverDnHandle& cusolver_handle,
    Argument<int32_t> uplo, Argument<int32_t> n,
    Argument<RCReference<GpuBuffer>> A, Argument<int32_t> lda,
    Argument<RCReference<GpuBuffer>> Workspace, Argument<int32_t> Lwork,
    Argument<RCReference<GpuBuffer>> devInfo, Argument<Chain> in_chain) {
  auto current = wrapper::CtxSetCurrent(context.get().get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(*uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  auto result = wrapper::CusolverDnDpotrf(
      current.get(), cusolver_handle.get(), *uplo_solver, *n,
      wrapper::Pointer<double>(A->get()->pointer()), *lda,
      wrapper::Pointer<double>(Workspace->get()->pointer()), *Lwork,
      wrapper::Pointer<int>(devInfo->get()->pointer()));
  if (!result) return std::move(result);
  return in_chain.get();
}

Expected<Chain> SolverDnCpotrf(
    Argument<GpuContext> context,
    const wrapper::OwningSolverDnHandle& cusolver_handle,
    Argument<int32_t> uplo, Argument<int32_t> n,
    Argument<RCReference<GpuBuffer>> A, Argument<int32_t> lda,
    Argument<RCReference<GpuBuffer>> Workspace, Argument<int32_t> Lwork,
    Argument<RCReference<GpuBuffer>> devInfo, Argument<Chain> in_chain) {
  auto current = wrapper::CtxSetCurrent(context.get().get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(*uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  auto result = wrapper::CusolverDnCpotrf(
      current.get(), cusolver_handle.get(), *uplo_solver, *n,
      wrapper::Pointer<cuComplex>(A->get()->pointer()), *lda,
      wrapper::Pointer<cuComplex>(Workspace->get()->pointer()), *Lwork,
      wrapper::Pointer<int>(devInfo->get()->pointer()));
  if (!result) return std::move(result);
  return in_chain.get();
}

Expected<Chain> SolverDnZpotrf(
    Argument<GpuContext> context,
    const wrapper::OwningSolverDnHandle& cusolver_handle,
    Argument<int32_t> uplo, Argument<int32_t> n,
    Argument<RCReference<GpuBuffer>> A, Argument<int32_t> lda,
    Argument<RCReference<GpuBuffer>> Workspace, Argument<int32_t> Lwork,
    Argument<RCReference<GpuBuffer>> devInfo, Argument<Chain> in_chain) {
  auto current = wrapper::CtxSetCurrent(context.get().get());
  if (!current) return current.takeError();

  auto uplo_solver = SafeIntToCublasFillMode(*uplo);
  if (!uplo_solver) return uplo_solver.takeError();

  auto result = wrapper::CusolverDnZpotrf(
      current.get(), cusolver_handle.get(), *uplo_solver, *n,
      wrapper::Pointer<cuDoubleComplex>(A->get()->pointer()), *lda,
      wrapper::Pointer<cuDoubleComplex>(Workspace->get()->pointer()), *Lwork,
      wrapper::Pointer<int>(devInfo->get()->pointer()));
  if (!result) return std::move(result);
  return in_chain.get();
}

void RegisterCudaSolverKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_cuda.solver.create", TFRT_KERNEL(SolverDnCreate));
  kernel_reg->AddKernel("tfrt_cuda.solver.set_stream",
                        TFRT_KERNEL(SolverDnSetStream));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.s.portf.buffer_size",
                        TFRT_KERNEL(SolverDnSpotrfBufferSize));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.d.portf.buffer_size",
                        TFRT_KERNEL(SolverDnDpotrfBufferSize));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.c.portf.buffer_size",
                        TFRT_KERNEL(SolverDnCpotrfBufferSize));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.z.portf.buffer_size",
                        TFRT_KERNEL(SolverDnZpotrfBufferSize));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.s.portf",
                        TFRT_KERNEL(SolverDnSpotrf));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.d.portf",
                        TFRT_KERNEL(SolverDnDpotrf));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.c.portf",
                        TFRT_KERNEL(SolverDnCpotrf));
  kernel_reg->AddKernel("tfrt_cuda.solver.dn.z.portf",
                        TFRT_KERNEL(SolverDnZpotrf));
}

}  // namespace gpu
}  // namespace tfrt
