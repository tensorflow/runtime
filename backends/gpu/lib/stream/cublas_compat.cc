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

//===- cublas_stub.cc -------------------------------------------*- C++ -*-===//
//
// Implementation of the cuBLAS API forwarding calls to symbols dynamically
// loaded from the real library.
//
//===----------------------------------------------------------------------===//

// This is a backward compartible wrapper for cublasGemmEx to support
// cuBLAS v10 and v11 builds.

#include "cublas.h"  // from @cuda_headers

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx_v10(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha, /* host or device pointer */
    const void *A, cudaDataType Atype, int lda, const void *B,
    cudaDataType Btype, int ldb, const void *beta, /* host or device pointer */
    void *C, cudaDataType Ctype, int ldc, cudaDataType computeType,
    cublasGemmAlgo_t algo) {
  return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B,
                      Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
}
