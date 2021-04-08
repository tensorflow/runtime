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

// Wrappers for BLAS
//
// This file declares utilities for conveniently calling BLAS functions.
// In the future, it will also be a place to deal with BLAS library bugs and
// differences between library versions.
//
// Currently, only cuBLAS is supported.
#ifndef TFRT_GPU_BLAS_SUPPORT_H_
#define TFRT_GPU_BLAS_SUPPORT_H_

#include "tfrt/gpu/stream/cublas_wrapper.h"
#include "tfrt/gpu/stream/stream_wrapper.h"

namespace tfrt {
namespace gpu {

// TODO(iga): TFRT does not support complex numbers yet. Add specializations for
// them when complex types are defined.
inline llvm::Error CublasGemm(stream::CurrentContext current,
                              cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              stream::Pointer<const float> alpha,
                              stream::Pointer<const float> A, int lda,
                              stream::Pointer<const float> B, int ldb,
                              stream::Pointer<const float> beta,
                              stream::Pointer<float> C, int ldc) {
  return stream::CublasSgemm(current, handle, transa, transb, m, n, k, alpha, A,
                             lda, B, ldb, beta, C, ldc);
}

inline llvm::Error CublasGemm(stream::CurrentContext current,
                              cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              stream::Pointer<const double> alpha,
                              stream::Pointer<const double> A, int lda,
                              stream::Pointer<const double> B, int ldb,
                              stream::Pointer<const double> beta,
                              stream::Pointer<double> C, int ldc) {
  return stream::CublasDgemm(current, handle, transa, transb, m, n, k, alpha, A,
                             lda, B, ldb, beta, C, ldc);
}

inline llvm::Error CublasGemm(stream::CurrentContext current,
                              cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              stream::Pointer<const __half> alpha,
                              stream::Pointer<const __half> A, int lda,
                              stream::Pointer<const __half> B, int ldb,
                              stream::Pointer<const __half> beta,
                              stream::Pointer<__half> C, int ldc) {
  return stream::CublasHgemm(current, handle, transa, transb, m, n, k, alpha, A,
                             lda, B, ldb, beta, C, ldc);
}

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_BLAS_SUPPORT_H_
