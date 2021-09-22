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

// Thin abstraction layer for cuBLAS and MIOpen.
#include "tfrt/gpu/wrapper/blas_wrapper.h"

#include "cublas.h"  // from @cuda_headers
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/wrapper/cublas_wrapper.h"
#include "tfrt/gpu/wrapper/rocblas_stub.h"
#include "tfrt/gpu/wrapper/rocblas_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

void internal::BlasHandleDeleter::operator()(BlasHandle handle) const {
  LogIfError(BlasDestroy(handle));
}

llvm::Expected<OwningBlasHandle> BlasCreate(CurrentContext current) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CublasCreate(current);
    case Platform::ROCm:
      return RocblasCreate(current);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error BlasDestroy(BlasHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CublasDestroy(handle);
    case Platform::ROCm:
      return RocblasDestroy(handle);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error BlasSetStream(BlasHandle handle, Stream stream) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CublasSetStream(handle, stream);
    case Platform::ROCm:
      return RocblasSetStream(handle, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<Stream> BlasGetStream(BlasHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CublasGetStream(handle);
    case Platform::ROCm:
      return RocblasGetStream(handle);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error BlasAxpyEx(CurrentContext current, BlasHandle handle, int n,
                       Pointer<const void> alpha, BlasDataType alphaType,
                       Pointer<const void> x, BlasDataType typeX, int strideX,
                       Pointer<void> y, BlasDataType typeY, int strideY,
                       BlasDataType executionType) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CublasAxpyEx(current, handle, n, alpha, alphaType, x, typeX,
                          strideX, y, typeY, strideY, executionType);
    case Platform::ROCm:
      return RocblasAxpyEx(current, handle, n, alpha, alphaType, x, typeX,
                           strideX, y, typeY, strideY, executionType);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error BlasGemmEx(CurrentContext current, BlasHandle handle,
                       BlasOperation transA, BlasOperation transB, int m, int n,
                       int k, Pointer<const void> alpha, Pointer<const void> A,
                       BlasDataType typeA, int heightA, Pointer<const void> B,
                       BlasDataType typeB, int heightB,
                       Pointer<const void> beta, Pointer<void> C,
                       BlasDataType typeC, int heightC,
                       BlasComputeType computeType, BlasGemmAlgo algo) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CublasGemmEx(current, handle, transA, transB, m, n, k, alpha, A,
                          typeA, heightA, B, typeB, heightB, beta, C, typeC,
                          heightC, computeType, algo);
    case Platform::ROCm:
      return RocblasGemmEx(current, handle, transA, transB, m, n, k, alpha, A,
                           typeA, heightA, B, typeB, heightB, beta, C, typeC,
                           heightC,
                           // Note: pass C as input and output.
                           C, typeC, heightC, computeType, algo);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error BlasGemmStridedBatchedEx(
    CurrentContext current, BlasHandle handle, BlasOperation transA,
    BlasOperation transB, int m, int n, int k, Pointer<const void> alpha,
    Pointer<const void> A, BlasDataType typeA, int heightA, int64_t strideA,
    Pointer<const void> B, BlasDataType typeB, int heightB, int64_t strideB,
    Pointer<const void> beta, Pointer<void> C, BlasDataType typeC, int heightC,
    int64_t strideC, int batchCount, BlasComputeType computeType,
    BlasGemmAlgo algo) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CublasGemmStridedBatchedEx(
          current, handle, transA, transB, m, n, k, alpha, A, typeA, heightA,
          strideA, B, typeB, heightB, strideB, beta, C, typeC, heightC, strideC,
          batchCount, computeType, algo);
    case Platform::ROCm:
      return RocblasGemmStridedBatchedEx(
          current, handle, transA, transB, m, n, k, alpha, A, typeA, heightA,
          strideA, B, typeB, heightB, strideB, beta, C, typeC, heightC, strideC,
          // Note: pass C as input and output.
          C, typeC, heightC, strideC, batchCount, computeType, algo);
    default:
      return InvalidPlatform(platform);
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
