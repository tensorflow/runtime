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
#ifndef TFRT_BACKENDS_GPU_LIB_OPS_TF_BLAS_SUPPORT_H_
#define TFRT_BACKENDS_GPU_LIB_OPS_TF_BLAS_SUPPORT_H_

#include "tfrt/gpu/wrapper/cublas_wrapper.h"

namespace tfrt {
namespace gpu {

// TODO(iga): TFRT does not support complex numbers yet. Add specializations for
// them when complex types are defined.
inline llvm::Error CublasGemm(wrapper::CurrentContext current,
                              cublasHandle_t handle, cublasOperation_t transA,
                              cublasOperation_t transB, int m, int n, int k,
                              wrapper::Pointer<const float> alpha,
                              wrapper::Pointer<const float> A, int heightA,
                              wrapper::Pointer<const float> B, int heightB,
                              wrapper::Pointer<const float> beta,
                              wrapper::Pointer<float> C, int heightC) {
  return wrapper::CublasSgemm(current, handle, transA, transB, m, n, k, alpha,
                              A, heightA, B, heightB, beta, C, heightC);
}

inline llvm::Error CublasGemm(wrapper::CurrentContext current,
                              cublasHandle_t handle, cublasOperation_t transA,
                              cublasOperation_t transB, int m, int n, int k,
                              wrapper::Pointer<const double> alpha,
                              wrapper::Pointer<const double> A, int heightA,
                              wrapper::Pointer<const double> B, int heightB,
                              wrapper::Pointer<const double> beta,
                              wrapper::Pointer<double> C, int heightC) {
  return wrapper::CublasDgemm(current, handle, transA, transB, m, n, k, alpha,
                              A, heightA, B, heightB, beta, C, heightC);
}

inline llvm::Error CublasGemm(wrapper::CurrentContext current,
                              cublasHandle_t handle, cublasOperation_t transA,
                              cublasOperation_t transB, int m, int n, int k,
                              wrapper::Pointer<const __half> alpha,
                              wrapper::Pointer<const __half> A, int heightA,
                              wrapper::Pointer<const __half> B, int heightB,
                              wrapper::Pointer<const __half> beta,
                              wrapper::Pointer<__half> C, int heightC) {
  return wrapper::CublasHgemm(current, handle, transA, transB, m, n, k, alpha,
                              A, heightA, B, heightB, beta, C, heightC);
}

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_GPU_LIB_OPS_TF_BLAS_SUPPORT_H_
