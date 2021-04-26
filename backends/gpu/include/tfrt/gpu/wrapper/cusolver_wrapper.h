/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Thin wrapper around the cuSOLVER API adding llvm::Error.
#ifndef TFRT_GPU_WRAPPER_CUSOLVER_WRAPPER_H_
#define TFRT_GPU_WRAPPER_CUSOLVER_WRAPPER_H_

#include "cusolverDn.h"  // from @cuda_headers
#include "tfrt/gpu/wrapper/solver_wrapper.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

extern template void internal::LogResult(llvm::raw_ostream &, cusolverStatus_t);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, cusolverStatus_t status);

llvm::Expected<OwningSolverHandle> CusolverDnCreate();
llvm::Error CusolverDnDestroy(cusolverDnHandle_t handle);
llvm::Error CusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t stream);
llvm::Expected<Stream> CusolverDnGetStream(cusolverDnHandle_t handle);
llvm::Error CusolverDnPotrf(CurrentContext current, cusolverDnHandle_t handle,
                            cublasFillMode_t uplo, int n, Pointer<float> A,
                            int lda, Pointer<float> Workspace, int Lwork,
                            Pointer<int> devInfo);
llvm::Error CusolverDnPotrf(CurrentContext current, cusolverDnHandle_t handle,
                            cublasFillMode_t uplo, int n, Pointer<double> A,
                            int lda, Pointer<double> Workspace, int Lwork,
                            Pointer<int> devInfo);
llvm::Error CusolverDnPotrf(CurrentContext current, cusolverDnHandle_t handle,
                            cublasFillMode_t uplo, int n, Pointer<cuComplex> A,
                            int lda, Pointer<cuComplex> Workspace, int Lwork,
                            Pointer<int> devInfo);
llvm::Error CusolverDnPotrf(CurrentContext current, cusolverDnHandle_t handle,
                            cublasFillMode_t uplo, int n,
                            Pointer<cuDoubleComplex> A, int lda,
                            Pointer<cuDoubleComplex> Workspace, int Lwork,
                            Pointer<int> devInfo);
llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              Pointer<float> A, int lda);
llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              Pointer<double> A, int lda);
llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              Pointer<cuComplex> A, int lda);
llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              Pointer<cuDoubleComplex> A,
                                              int lda);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CUSOLVER_WRAPPER_H_
