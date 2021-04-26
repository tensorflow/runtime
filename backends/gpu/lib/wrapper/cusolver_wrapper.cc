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

// Thin wrapper around the cuSOLVER API adding llvm::Error.
#include "tfrt/gpu/wrapper/cusolver_wrapper.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

template void internal::LogResult(llvm::raw_ostream &, cusolverStatus_t);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, cusolverStatus_t status) {
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:
      return os << "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return os << "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return os << "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return os << "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return os << "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return os << "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return os << "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return os << "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return os << "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return os << "CUSOLVER_STATUS_NOT_SUPPORTED";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return os << "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return os << "CUSOLVER_STATUS_INVALID_LICENSE";
    default:
      return os << llvm::formatv("cusolverStatus_t({0})",
                                 static_cast<int>(status));
  }
}

llvm::Expected<OwningSolverHandle> CusolverDnCreate() {
  cusolverDnHandle_t handle = nullptr;
  RETURN_IF_ERROR(cusolverDnCreate(&handle));
  return OwningSolverHandle(handle);
}

llvm::Error CusolverDnDestroy(cusolverDnHandle_t handle) {
  return TO_ERROR(cusolverDnDestroy(handle));
}

llvm::Error CusolverDnSetStream(cusolverDnHandle_t handle,
                                cudaStream_t stream) {
  return TO_ERROR(cusolverDnSetStream(handle, stream));
}

llvm::Expected<Stream> CusolverDnGetStream(cusolverDnHandle_t handle) {
  cudaStream_t stream = nullptr;
  RETURN_IF_ERROR(cusolverDnGetStream(handle, &stream));
  return Stream(stream);
}

llvm::Error CusolverDnPotrf(CurrentContext current, cusolverDnHandle_t handle,
                            cublasFillMode_t uplo, int n, Pointer<float> A,
                            int lda, Pointer<float> Workspace, int Lwork,
                            Pointer<int> devInfo) {
  CheckCudaContext(current);
  return TO_ERROR(cusolverDnSpotrf(handle, uplo, n, ToCuda(A), lda,
                                   ToCuda(Workspace), Lwork, ToCuda(devInfo)));
}

llvm::Error CusolverDnPotrf(CurrentContext current, cusolverDnHandle_t handle,
                            cublasFillMode_t uplo, int n, Pointer<double> A,
                            int lda, Pointer<double> Workspace, int Lwork,
                            Pointer<int> devInfo) {
  CheckCudaContext(current);
  return TO_ERROR(cusolverDnDpotrf(handle, uplo, n, ToCuda(A), lda,
                                   ToCuda(Workspace), Lwork, ToCuda(devInfo)));
}

llvm::Error CusolverDnPotrf(CurrentContext current, cusolverDnHandle_t handle,
                            cublasFillMode_t uplo, int n, Pointer<cuComplex> A,
                            int lda, Pointer<cuComplex> Workspace, int Lwork,
                            Pointer<int> devInfo) {
  CheckCudaContext(current);
  return TO_ERROR(cusolverDnCpotrf(handle, uplo, n, ToCuda(A), lda,
                                   ToCuda(Workspace), Lwork, ToCuda(devInfo)));
}

llvm::Error CusolverDnPotrf(CurrentContext current, cusolverDnHandle_t handle,
                            cublasFillMode_t uplo, int n,
                            Pointer<cuDoubleComplex> A, int lda,
                            Pointer<cuDoubleComplex> Workspace, int Lwork,
                            Pointer<int> devInfo) {
  CheckCudaContext(current);
  return TO_ERROR(cusolverDnZpotrf(handle, uplo, n, ToCuda(A), lda,
                                   ToCuda(Workspace), Lwork, ToCuda(devInfo)));
}

llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              Pointer<float> A, int lda) {
  CheckCudaContext(current);
  int Lwork;
  RETURN_IF_ERROR(
      cusolverDnSpotrf_bufferSize(handle, uplo, n, ToCuda(A), lda, &Lwork));
  return Lwork;
}

llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              Pointer<double> A, int lda) {
  CheckCudaContext(current);
  int Lwork;
  RETURN_IF_ERROR(
      cusolverDnDpotrf_bufferSize(handle, uplo, n, ToCuda(A), lda, &Lwork));
  return Lwork;
}

llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              Pointer<cuComplex> A, int lda) {
  CheckCudaContext(current);
  int Lwork;
  RETURN_IF_ERROR(
      cusolverDnCpotrf_bufferSize(handle, uplo, n, ToCuda(A), lda, &Lwork));
  return Lwork;
}

llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              Pointer<cuDoubleComplex> A,
                                              int lda) {
  CheckCudaContext(current);
  int Lwork;
  RETURN_IF_ERROR(
      cusolverDnZpotrf_bufferSize(handle, uplo, n, ToCuda(A), lda, &Lwork));
  return Lwork;
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
