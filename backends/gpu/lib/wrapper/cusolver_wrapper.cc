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

#include "llvm/Support/FormatVariadic.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream &Print(llvm::raw_ostream &os, cusolverStatus_t status) {
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
                            cudaDataType dataType, cublasFillMode_t fillMode, int n,
                            Pointer<void> A, int heightA, Pointer<void> workspace,
                            int workspaceSize, Pointer<int> devInfo) {
  CheckCudaContext(current);
  switch (dataType) {
    case CUDA_R_32F:
      return TO_ERROR(cusolverDnSpotrf(
          handle, fillMode, n,
          reinterpret_cast<float*>(ToCuda(A)), heightA,
          reinterpret_cast<float*>(ToCuda(workspace)),
          workspaceSize, ToCuda(devInfo)));
    case CUDA_C_32F:
      return TO_ERROR(cusolverDnCpotrf(
          handle, fillMode, n,
          reinterpret_cast<cuComplex*>(ToCuda(A)), heightA,
          reinterpret_cast<cuComplex*>(ToCuda(workspace)),
          workspaceSize, ToCuda(devInfo)));
    case CUDA_R_64F:
      return TO_ERROR(cusolverDnDpotrf(
          handle, fillMode, n,
          reinterpret_cast<double*>(ToCuda(A)), heightA,
          reinterpret_cast<double*>(ToCuda(workspace)),
          workspaceSize, ToCuda(devInfo)));
    case CUDA_C_64F:
      return TO_ERROR(cusolverDnZpotrf(
          handle, fillMode, n,
          reinterpret_cast<cuDoubleComplex*>(ToCuda(A)), heightA,
          reinterpret_cast<cuDoubleComplex*>(ToCuda(workspace)),
          workspaceSize, ToCuda(devInfo)));
    default:
      return MakeStringError("Unsupported type: ", Printed(dataType));
  }
}

llvm::Error CusolverDnPotrfBatched(CurrentContext current,
                                   cusolverDnHandle_t handle, cudaDataType dataType,
                                   cublasFillMode_t fillMode, int n,
                                   Pointer<void *> Aarray, int heightA,
                                   Pointer<int> devInfoArray, int batchSize) {
  CheckCudaContext(current);
  switch (dataType) {
    case CUDA_R_32F:
      return TO_ERROR(cusolverDnSpotrfBatched(
          handle, fillMode, n,
          reinterpret_cast<float**>(ToCuda(Aarray)), heightA,
          ToCuda(devInfoArray), batchSize));
    case CUDA_C_32F:
      return TO_ERROR(cusolverDnCpotrfBatched(
          handle, fillMode, n,
          reinterpret_cast<cuComplex**>(ToCuda(Aarray)), heightA,
          ToCuda(devInfoArray), batchSize));
    case CUDA_R_64F:
      return TO_ERROR(cusolverDnDpotrfBatched(
          handle, fillMode, n,
          reinterpret_cast<double**>(ToCuda(Aarray)), heightA,
          ToCuda(devInfoArray), batchSize));
    case CUDA_C_64F:
      return TO_ERROR(cusolverDnZpotrfBatched(
          handle, fillMode, n,
          reinterpret_cast<cuDoubleComplex**>(ToCuda(Aarray)), heightA,
          ToCuda(devInfoArray), batchSize));
    default:
      return MakeStringError("Unsupported type: ", Printed(dataType));
  }
}

llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t fillMode, int n,
                                              Pointer<float> A, int heightA) {
  CheckCudaContext(current);
  int workspaceSize;
  RETURN_IF_ERROR(cusolverDnSpotrf_bufferSize(handle, fillMode, n, ToCuda(A),
                                              heightA, &workspaceSize));
  return workspaceSize;
}

llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t fillMode, int n,
                                              Pointer<double> A, int heightA) {
  CheckCudaContext(current);
  int workspaceSize;
  RETURN_IF_ERROR(cusolverDnDpotrf_bufferSize(handle, fillMode, n, ToCuda(A),
                                              heightA, &workspaceSize));
  return workspaceSize;
}

llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t fillMode, int n,
                                              Pointer<cuComplex> A,
                                              int heightA) {
  CheckCudaContext(current);
  int workspaceSize;
  RETURN_IF_ERROR(cusolverDnCpotrf_bufferSize(handle, fillMode, n, ToCuda(A),
                                              heightA, &workspaceSize));
  return workspaceSize;
}

llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t fillMode, int n,
                                              Pointer<cuDoubleComplex> A,
                                              int heightA) {
  CheckCudaContext(current);
  int workspaceSize;
  RETURN_IF_ERROR(cusolverDnZpotrf_bufferSize(handle, fillMode, n, ToCuda(A),
                                              heightA, &workspaceSize));
  return workspaceSize;
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
