/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

// Thin wrapper around the rocsolver API adding llvm::Error.
#ifndef TFRT_GPU_WRAPPER_ROCSOLVER_WRAPPER_H_
#define TFRT_GPU_WRAPPER_ROCSOLVER_WRAPPER_H_

#include "tfrt/gpu/wrapper/rocsolver_stub.h"
#include "tfrt/gpu/wrapper/solver_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::Expected<OwningSolverHandle> RocsolverCreate();
llvm::Error RocsolverDestroy(rocblas_handle handle);
llvm::Error RocsolverSetStream(rocblas_handle handle, hipStream_t stream);
llvm::Expected<Stream> RocsolverGetStream(rocblas_handle handle);
llvm::Error RocsolverPotrf(CurrentContext current, rocblas_handle handle,
                           rocblas_datatype dataType, rocblas_fill fillMode, int n,
                           Pointer<void> A, int heightA, Pointer<int> devInfo);
llvm::Error RocsolverPotrfBatched(CurrentContext current, rocblas_handle handle,
                                  rocblas_datatype dataType, rocblas_fill fillMode,
                                  int n, Pointer<void *> Aarray,
                                  int heightA, Pointer<int> devInfo,
                                  int batchSize);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_ROCSOLVER_WRAPPER_H_
