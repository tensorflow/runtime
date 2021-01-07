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
template <typename Func, Func, typename... Args>
static cublasStatus_t DynamicCall(const char *symbol_name, Args &&...args) {
  static auto func_ptr = GetFunctionPointer<Func>(symbol_name);
  if (!func_ptr) return CUBLAS_STATUS_NOT_INITIALIZED;
  return func_ptr(std::forward<Args>(args)...);
}

#define CUBLASWINAPI

extern "C" {
#include "cublas_stub.cc.inc"

cublasStatus_t CUBLASWINAPI cublasHgemm(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const __half *alpha, /* host or device pointer */
    const __half *A, int lda, const __half *B, int ldb,
    const __half *beta, /* host or device pointer */
    __half *C, int ldc) {
  return DynamicCall<decltype(cublasHgemm), cublasHgemm>(
      "cublasHgemm", handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
      beta, C, ldc);
}

// cuBLAS broke backwards compatibility from v10 to v11 by changing the
// computeType argument from cudaDataType to cublasComputeType_t. This function
// implements the v10 interface in a forward-compatible way if the stub is
// compiled with v11 headers.
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx_v10(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha, /* host or device pointer */
    const void *A, cudaDataType Atype, int lda, const void *B,
    cudaDataType Btype, int ldb, const void *beta, /* host or device pointer */
    void *C, cudaDataType Ctype, int ldc, cudaDataType computeType,
    cublasGemmAlgo_t algo) {
  static auto func_ptr = LoadSymbol("cublasGemmEx");
  if (!func_ptr) return CUBLAS_STATUS_NOT_INITIALIZED;
  static auto version_pair = [&] {
    int version = 0;
    auto status = cublasGetVersion_v2(handle, &version);
    return std::make_pair(version, status);
  }();
  if (version_pair.second != CUBLAS_STATUS_SUCCESS) return version_pair.second;
  if (version_pair.first >= 11000) {
#if CUBLAS_VER_MAJOR >= 11
    cublasComputeType_t migratedComputeType = CUBLAS_COMPUTE_32F;
    auto status =
        cublasMigrateComputeType(handle, computeType, &migratedComputeType);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
        cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
        const void *, const void *, cudaDataType, int, const void *,
        cudaDataType, int, const void *, void *, cudaDataType, int,
        cublasComputeType_t, cublasGemmAlgo_t);
    return reinterpret_cast<FuncPtr>(func_ptr)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
        beta, C, Ctype, ldc, migratedComputeType, algo);
#else
    return CUBLAS_STATUS_NOT_SUPPORTED;
#endif
  }
  return reinterpret_cast<decltype(cublasGemmEx_v10) *>(func_ptr)(
      handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
      beta, C, Ctype, ldc, computeType, algo);
}

}  // extern "C"
