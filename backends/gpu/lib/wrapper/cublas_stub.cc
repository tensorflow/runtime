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

#include <type_traits>

#include "cublas.h"  // from @cuda_headers
#include "symbol_loader.h"

// Memoizes load of the .so for this CUDA library.
static void *LoadSymbol(const char *symbol_name) {
  static SymbolLoader loader("libcublas.so");
  return loader.GetAddressOfSymbol(symbol_name);
}

template <typename Func>
static Func *GetFunctionPointer(const char *symbol_name, Func *func = nullptr) {
  return reinterpret_cast<Func *>(LoadSymbol(symbol_name));
}

// Calls function 'symbol_name' in shared library with 'args'.
// TODO(csigg): Change to 'auto Func' when C++17 is allowed.
template <typename Func, Func *, typename... Args>
static cublasStatus_t DynamicCall(const char *symbol_name, Args &&...args) {
  static auto func_ptr = GetFunctionPointer<Func>(symbol_name);
  if (!func_ptr) return CUBLAS_STATUS_NOT_INITIALIZED;
  return func_ptr(std::forward<Args>(args)...);
}

#define CUBLASWINAPI

extern "C" {
#include "cublas_stub.cc.inc"

// The functions below are overloaded (for backwards compatibility) and
// therefore need to explicitly specify the function type.

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha, /* host or device pointer */
    const void *A, cudaDataType Atype, int lda, const void *B,
    cudaDataType Btype, int ldb, const void *beta, /* host or device pointer */
    void *C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType,
    cublasGemmAlgo_t algo) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
      const void *, const void *, cudaDataType, int, const void *, cudaDataType,
      int, const void *, void *, cudaDataType, int, cublasComputeType_t,
      cublasGemmAlgo_t);
  return DynamicCall<std::remove_pointer_t<FuncPtr>, &cublasGemmEx>(
      "cublasGemmEx", handle, transa, transb, m, n, k, alpha, A, Atype, lda, B,
      Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha, /* host or device pointer */
    const void *const Aarray[], cudaDataType Atype, int lda,
    const void *const Barray[], cudaDataType Btype, int ldb,
    const void *beta, /* host or device pointer */
    void *const Carray[], cudaDataType Ctype, int ldc, int batchCount,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
      const void *, const void *const[], cudaDataType, int, const void *const[],
      cudaDataType, int, const void *, void *const[], cudaDataType, int, int,
      cublasComputeType_t, cublasGemmAlgo_t);
  return DynamicCall<std::remove_pointer_t<FuncPtr>, &cublasGemmBatchedEx>(
      "cublasGemmBatchedEx", handle, transa, transb, m, n, k, alpha, Aarray,
      Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount,
      computeType, algo);
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha, /* host or device pointer */
    const void *A, cudaDataType Atype, int lda,
    long long int strideA,  // NOLINT: google-runtime-int and runtime/int
    const void *B, cudaDataType Btype, int ldb,
    long long int strideB,  // NOLINT
    const void *beta,       /* host or device pointer */
    void *C, cudaDataType Ctype, int ldc,
    long long int strideC,  // NOLINT
    int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
      const void *, const void *, cudaDataType, int,
      long long int,  // NOLINT
      const void *, cudaDataType, int,
      long long int,  // NOLINT
      const void *, void *, cudaDataType, int,
      long long int,  // NOLINT
      int, cublasComputeType_t, cublasGemmAlgo_t);
  return DynamicCall<std::remove_pointer_t<FuncPtr>,
                     &cublasGemmStridedBatchedEx>(
      "cublasGemmStridedBatchedEx", handle, transa, transb, m, n, k, alpha, A,
      Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC,
      batchCount, computeType, algo);
}

}  // extern "C"
