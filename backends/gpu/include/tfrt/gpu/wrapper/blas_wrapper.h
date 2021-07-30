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

// Thin abstraction layer for cuBLAS and MIOpen.
#ifndef TFRT_GPU_WRAPPER_BLAS_WRAPPER_H_
#define TFRT_GPU_WRAPPER_BLAS_WRAPPER_H_

#include <cstddef>
#include <memory>

#include "tfrt/gpu/wrapper/wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

// Platform-discriminated enums.
struct BlasDataTypeTag;
using BlasDataType = Enum<BlasDataTypeTag>;
struct BlasOperationTag;
using BlasOperation = Enum<BlasOperationTag>;
struct BlasGemmAlgoTag;
using BlasGemmAlgo = Enum<BlasGemmAlgoTag>;

// Non-owning handles of GPU resources.
using BlasHandle = Resource<cublasHandle_t, rocblas_handle>;

namespace internal {
// Helper to wrap resources and memory into RAII types.
struct BlasHandleDeleter {
  using pointer = BlasHandle;
  void operator()(BlasHandle handle) const;
};
}  // namespace internal

// RAII wrappers for resources. Instances own the underlying resource.
//
// They are implemented as std::unique_ptrs with custom deleters.
//
// Use get() and release() to access the non-owning handle, please use with
// appropriate care.
using OwningBlasHandle = internal::OwningResource<internal::BlasHandleDeleter>;

llvm::Expected<OwningBlasHandle> BlasCreate(CurrentContext current);
llvm::Error BlasDestroy(BlasHandle handle);
llvm::Error BlasSetStream(BlasHandle handle, Stream stream);
llvm::Expected<Stream> BlasGetStream(BlasHandle handle);

llvm::Error BlasAxpyEx(CurrentContext current, BlasHandle handle, int n,
                       Pointer<const void> alpha, BlasDataType alphaType,
                       Pointer<const void> x, BlasDataType typeX, int strideX,
                       Pointer<void> y, BlasDataType typeY, int strideY,
                       BlasDataType executionType);

llvm::Error BlasGemmEx(CurrentContext current, BlasHandle handle,
                       BlasOperation transA, BlasOperation transB, int m, int n,
                       int k, Pointer<const void> alpha, Pointer<const void> A,
                       BlasDataType typeA, int heightA, Pointer<const void> B,
                       BlasDataType typeB, int heightB,
                       Pointer<const void> beta, Pointer<void> C,
                       BlasDataType typeC, int heightC,
                       BlasDataType computeType, BlasGemmAlgo algo);
llvm::Error BlasGemmStridedBatchedEx(
    CurrentContext current, BlasHandle handle, BlasOperation transA,
    BlasOperation transB, int m, int n, int k, Pointer<const void> alpha,
    Pointer<const void> A, BlasDataType typeA, int heightA, int64_t strideA,
    Pointer<const void> B, BlasDataType typeB, int heightB, int64_t strideB,
    Pointer<const void> beta, Pointer<void> C, BlasDataType typeC, int heightC,
    int64_t strideC, int batchCount, BlasDataType computeType,
    BlasGemmAlgo algo);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_BLAS_WRAPPER_H_
