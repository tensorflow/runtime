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

// Thin abstraction layer for cuSOLVER and rocSOLVER.
#ifndef TFRT_GPU_WRAPPER_SOLVER_WRAPPER_H_
#define TFRT_GPU_WRAPPER_SOLVER_WRAPPER_H_

#include <cstddef>
#include <memory>

#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/gpu/wrapper/blas_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

// Non-owning handles of GPU resources.
using SolverHandle = Resource<cusolverDnHandle_t, rocsolver_handle>;

namespace internal {
// Helper to wrap resources and memory into RAII types.
struct SolverHandleDeleter {
  using pointer = SolverHandle;
  void operator()(SolverHandle handle) const;
};
}  // namespace internal

// RAII wrappers for resources. Instances own the underlying resource.
using OwningSolverHandle =
    internal::OwningResource<internal::SolverHandleDeleter>;

llvm::Expected<OwningSolverHandle> SolverCreate(Platform platform);
llvm::Error SolverDestroy(SolverHandle handle);
llvm::Error SolverSetStream(SolverHandle handle, Stream stream);
llvm::Expected<Stream> SolverGetStream(SolverHandle handle);
llvm::Error SolverPotrf(CurrentContext current, SolverHandle handle,
                        BlasDataType dataType, BlasFillMode fillMode, int n,
                        Pointer<void> buffer, int stride,
                        Pointer<void> workspace, int workspaceSize,
                        Pointer<int> devInfo);
llvm::Error SolverPotrfBatched(CurrentContext current, SolverHandle handle,
                               BlasDataType dataType, BlasFillMode fillMode, int n,
                               Pointer<void*> Aarray, int heightA,
                               Pointer<int> devInfoArray, int batchSize);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_SOLVER_WRAPPER_H_
