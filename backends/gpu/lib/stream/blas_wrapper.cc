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
#include "tfrt/gpu/stream/blas_wrapper.h"

#include "cublas.h"  // from @cuda_headers
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/stream/cublas_wrapper.h"
#include "tfrt/gpu/stream/rocblas_stub.h"
#include "tfrt/gpu/stream/rocblas_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace stream {

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

llvm::Error BlasSaxpy(CurrentContext current, BlasHandle handle, int n,
                      Pointer<const float> alpha, Pointer<const float> x,
                      int incx, Pointer<float> y, int incy) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CublasSaxpy(current, handle, n, alpha, x, incx, y, incy);
    case Platform::ROCm:
      return RocblasSaxpy(current, handle, n, alpha, x, incx, y, incy);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error BlasSgemm(CurrentContext current, BlasHandle handle,
                      BlasOperation transa, BlasOperation transb, int m, int n,
                      int k, Pointer<const float> alpha, Pointer<const float> A,
                      int lda, Pointer<const float> B, int ldb,
                      Pointer<const float> beta, Pointer<float> C, int ldc) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CublasSgemm(current, handle, ToCublas(transa), ToCublas(transb), m,
                         n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    case Platform::ROCm:
      return RocblasSgemm(current, handle, ToRocblas(transa), ToRocblas(transb),
                          m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    default:
      return InvalidPlatform(platform);
  }
}

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt
