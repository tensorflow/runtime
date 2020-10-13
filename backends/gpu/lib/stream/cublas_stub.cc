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
#include <dlfcn.h>

#include "cublas.h"  // from @cuda_headers
#include "tfrt/support/logging.h"

static void *LoadSymbol(const char *symbol_name) {
  static void *handle = [&] {
    auto ptr = dlopen("libcublas.so", RTLD_LAZY);
    if (!ptr) TFRT_LOG_ERROR << "Failed to load libcublas.so";
    return ptr;
  }();
  return handle ? dlsym(handle, symbol_name) : nullptr;
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
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
      const __half *, const __half *, int, const __half *, int, const __half *,
      __half *, int);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LoadSymbol("cublasHgemm"));
  if (!func_ptr) return CUBLAS_STATUS_NOT_INITIALIZED;
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,
                  C, ldc);
}
}
