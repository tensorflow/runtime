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
#include "tfrt/gpu/wrapper/cuda_type_traits.h"
#include "tfrt/gpu/wrapper/solver_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

raw_ostream &Print(raw_ostream &os, cusolverStatus_t status);

llvm::Expected<OwningSolverHandle> CusolverDnCreate();
llvm::Error CusolverDnDestroy(cusolverDnHandle_t handle);
llvm::Error CusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t stream);
llvm::Expected<Stream> CusolverDnGetStream(cusolverDnHandle_t handle);
llvm::Error CusolverDnPotrf(CurrentContext current, cusolverDnHandle_t handle,
                            cudaDataType dataType, cublasFillMode_t fillMode, int n,
                            Pointer<void> A, int heightA, Pointer<void> workspace,
                            int workspaceSize, Pointer<int> devInfo);
llvm::Error CusolverDnPotrfBatched(CurrentContext current,
                                   cusolverDnHandle_t handle, cudaDataType dataType,
                                   cublasFillMode_t fillMode, int n,
                                   Pointer<void *> Aarray, int heightA,
                                   Pointer<int> devInfoArray, int batchSize);
llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t fillMode, int n,
                                              Pointer<float> A, int heightA);
llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t fillMode, int n,
                                              Pointer<double> A, int heightA);
llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t fillMode, int n,
                                              Pointer<cuComplex> A,
                                              int heightA);
llvm::Expected<int> CusolverDnPotrfBufferSize(CurrentContext current,
                                              cusolverDnHandle_t handle,
                                              cublasFillMode_t fillMode, int n,
                                              Pointer<cuDoubleComplex> A,
                                              int heightA);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CUSOLVER_WRAPPER_H_
