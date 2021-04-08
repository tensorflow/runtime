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
#ifndef TFRT_GPU_STREAM_CUSOLVER_WRAPPER_H_
#define TFRT_GPU_STREAM_CUSOLVER_WRAPPER_H_

#include "cusolverDn.h"  // from @cuda_headers
#include "tfrt/gpu/stream/solver_wrapper.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace gpu {
namespace stream {

struct CusolverErrorData {
  cusolverStatus_t result;
  const char *expr;
  StackTrace stack_trace;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const CusolverErrorData &data);
// Wraps a cusolverStatus_t into an llvm::ErrorInfo.
using CusolverErrorInfo = TupleErrorInfo<CusolverErrorData>;
cusolverStatus_t GetResult(const CusolverErrorInfo &info);

llvm::Expected<OwningSolverDnHandle> CusolverDnCreate(CurrentContext current);
llvm::Error CusolverDnDestroy(cusolverDnHandle_t handle);
llvm::Error CusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t stream);
llvm::Expected<Stream> CusolverDnGetStream(cusolverDnHandle_t handle);
llvm::Expected<int> CusolverDnSpotrf(CurrentContext current,
                                     cusolverDnHandle_t handle,
                                     cublasFillMode_t uplo, int n,
                                     Pointer<float> A, int lda,
                                     Pointer<float> Workspace, int Lwork);
llvm::Expected<int> CusolverDnDpotrf(CurrentContext current,
                                     cusolverDnHandle_t handle,
                                     cublasFillMode_t uplo, int n,
                                     Pointer<double> A, int lda,
                                     Pointer<double> Workspace, int Lwork);
llvm::Expected<int> CusolverDnCpotrf(CurrentContext current,
                                     cusolverDnHandle_t handle,
                                     cublasFillMode_t uplo, int n,
                                     Pointer<cuComplex> A, int lda,
                                     Pointer<cuComplex> Workspace, int Lwork);
llvm::Expected<int> CusolverDnZpotrf(CurrentContext current,
                                     cusolverDnHandle_t handle,
                                     cublasFillMode_t uplo, int n,
                                     Pointer<cuDoubleComplex> A, int lda,
                                     Pointer<cuDoubleComplex> Workspace,
                                     int Lwork);
llvm::Expected<int> CusolverDnSpotrfBufferSize(CurrentContext current,
                                               cusolverDnHandle_t handle,
                                               cublasFillMode_t uplo, int n,
                                               Pointer<float> A, int lda);
llvm::Expected<int> CusolverDnDpotrfBufferSize(CurrentContext current,
                                               cusolverDnHandle_t handle,
                                               cublasFillMode_t uplo, int n,
                                               Pointer<double> A, int lda);
llvm::Expected<int> CusolverDnCpotrfBufferSize(CurrentContext current,
                                               cusolverDnHandle_t handle,
                                               cublasFillMode_t uplo, int n,
                                               Pointer<cuComplex> A, int lda);
llvm::Expected<int> CusolverDnZpotrfBufferSize(CurrentContext current,
                                               cusolverDnHandle_t handle,
                                               cublasFillMode_t uplo, int n,
                                               Pointer<cuDoubleComplex> A,
                                               int lda);

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_STREAM_CUSOLVER_WRAPPER_H_
