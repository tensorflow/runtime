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

// Thin abstraction layer for cuSOLVER and rocSOLVER.
#include "tfrt/gpu/wrapper/solver_wrapper.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/wrapper/cublas_wrapper.h"
#include "tfrt/gpu/wrapper/rocblas_wrapper.h"
#include "tfrt/gpu/wrapper/cusolver_wrapper.h"
#include "tfrt/gpu/wrapper/rocsolver_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

void internal::SolverHandleDeleter::operator()(SolverHandle handle) const {
  LogIfError(SolverDestroy(handle));
}

llvm::Expected<OwningSolverHandle> SolverCreate(Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CusolverDnCreate();
    case Platform::ROCm:
      return RocsolverCreate();
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error SolverDestroy(SolverHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CusolverDnDestroy(handle);
    case Platform::ROCm:
      return RocsolverDestroy(handle);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error SolverSetStream(SolverHandle handle, Stream stream) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CusolverDnSetStream(handle, stream);
    case Platform::ROCm:
      return RocsolverSetStream(handle, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<Stream> SolverGetStream(SolverHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CusolverDnGetStream(handle);
    case Platform::ROCm:
      return RocsolverGetStream(handle);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error SolverPotrf(CurrentContext current, SolverHandle handle,
                        BlasDataType dataType, BlasFillMode fillMode, int n,
                        Pointer<void> buffer, int stride,
                        Pointer<void> workspace, int workspaceSize,
                        Pointer<int> devInfo) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CusolverDnPotrf(current, handle, dataType, fillMode, n, buffer,
                             stride, workspace, workspaceSize, devInfo);
    case Platform::ROCm:
      return RocsolverPotrf(current, handle, dataType, fillMode, n, buffer,
                            stride, devInfo);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error SolverPotrfBatched(CurrentContext current, SolverHandle handle,
                               BlasDataType dataType, BlasFillMode fillMode, int n,
                               Pointer<void*> Aarray, int heightA,
                               Pointer<int> devInfoArray, int batchSize) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CusolverDnPotrfBatched(current, handle, dataType, fillMode, n,
          Aarray, heightA, devInfoArray, batchSize);
    case Platform::ROCm:
      return RocsolverPotrfBatched(current, handle, dataType, fillMode, n,
          Aarray, heightA, devInfoArray, batchSize);
    default:
      return InvalidPlatform(platform);
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
