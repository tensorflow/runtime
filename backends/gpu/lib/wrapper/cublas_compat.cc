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

// Implementation of the cuBLAS API forwarding calls to symbols dynamically
// loaded from the real library.

// This is a backward compartible wrapper for cublasGemmEx to support
// cuBLAS v10 and v11 builds.

#include <cstdint>

#include "cublas.h"  // from @cuda_headers

extern "C" {

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx_v10(
    cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
    int m, int n, int k, const void *alpha, /* host or device pointer */
    const void *A, cudaDataType Atype, int heightA, const void *B,
    cudaDataType Btype, int heightB,
    const void *beta, /* host or device pointer */
    void *C, cudaDataType Ctype, int heightC, cudaDataType computeType,
    cublasGemmAlgo_t algo) {
  return cublasGemmEx(handle, transA, transB, m, n, k, alpha, A, Atype, heightA,
                      B, Btype, heightB, beta, C, Ctype, heightC, computeType,
                      algo);
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx_v10(
    cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
    int m, int n, int k, const void *alpha, /* host or device pointer */
    const void *Aarray[], cudaDataType Atype, int heightA, const void *Barray[],
    cudaDataType Btype, int heightB,
    const void *beta, /* host or device pointer */
    void *Carray[], cudaDataType Ctype, int heightC, int batchCount,
    cudaDataType computeType, cublasGemmAlgo_t algo) {
  return cublasGemmBatchedEx(handle, transA, transB, m, n, k, alpha, Aarray,
                             Atype, heightA, Barray, Btype, heightB, beta,
                             Carray, Ctype, heightC, batchCount, computeType,
                             algo);
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx_v10(
    cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
    int m, int n, int k, const void *alpha, /* host or device pointer */
    const void *A, cudaDataType Atype, int heightA, int64_t strideA,
    const void *B, cudaDataType Btype, int heightB, int64_t strideB,
    const void *beta, /* host or device pointer */
    void *C, cudaDataType Ctype, int heightC, int64_t strideC, int batchCount,
    cudaDataType computeType, cublasGemmAlgo_t algo) {
  return cublasGemmStridedBatchedEx(handle, transA, transB, m, n, k, alpha, A,
                                    Atype, heightA, strideA, B, Btype, heightB,
                                    strideB, beta, C, Ctype, heightC, strideC,
                                    batchCount, computeType, algo);
}

}  // extern "C"
