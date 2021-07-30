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

// Thin wrapper around the cuBLAS API adding llvm::Error.
#ifndef TFRT_GPU_WRAPPER_CUBLAS_WRAPPER_H_
#define TFRT_GPU_WRAPPER_CUBLAS_WRAPPER_H_

#include "cublas.h"  // from @cuda_headers
#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/cuda_type_traits.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cublasStatus_t status);

template <>
Expected<cudaDataType> Parse<cudaDataType>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cudaDataType value);

template <>
Expected<cublasOperation_t> Parse<cublasOperation_t>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cublasOperation_t value);

template <>
Expected<cublasGemmAlgo_t> Parse<cublasGemmAlgo_t>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cublasGemmAlgo_t value);

template <>
struct PlatformTypeTraits<BlasDataTypeTag, cudaDataType>
    : public CudaPlatformType {};
template <>
struct PlatformTypeTraits<BlasOperationTag, cublasOperation_t>
    : public CudaPlatformType {};
template <>
struct PlatformTypeTraits<BlasGemmAlgoTag, cublasGemmAlgo_t>
    : public CudaPlatformType {};

llvm::Expected<OwningBlasHandle> CublasCreate(CurrentContext current);
llvm::Error CublasDestroy(cublasHandle_t handle);
llvm::Expected<int> CublasGetVersion(cublasHandle_t handle);
llvm::Error CublasSetStream(cublasHandle_t handle, cudaStream_t stream);
llvm::Expected<Stream> CublasGetStream(cublasHandle_t handle);
llvm::Error CublasSetPointerMode(cublasHandle_t handle,
                                 cublasPointerMode_t mode);
llvm::Expected<cublasPointerMode_t> CublasGetPointerMode(cublasHandle_t handle);
llvm::Error CublasSetMathMode(cublasHandle_t handle, cublasMath_t math_type);
llvm::Expected<cublasMath_t> CublasGetMathMode(cublasHandle_t handle);

llvm::Error CublasAxpyEx(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const void> alpha, /* host or device pointer */
                         cudaDataType alphaType, Pointer<const void> x,
                         cudaDataType typeX, int strideX, Pointer<void> y,
                         cudaDataType typeY, int strideY,
                         cudaDataType executionType);

llvm::Error CublasGemmEx(CurrentContext current, cublasHandle_t handle,
                         cublasOperation_t transA, cublasOperation_t transB,
                         int m, int n, int k, Pointer<const void> alpha,
                         Pointer<const void> A, cudaDataType typeA, int heightA,
                         Pointer<const void> B, cudaDataType typeB, int heightB,
                         Pointer<const void> beta, Pointer<void> C,
                         cudaDataType typeC, int heightC,
                         cudaDataType computeType, cublasGemmAlgo_t algo);
llvm::Error CublasGemmBatchedEx(
    CurrentContext current, cublasHandle_t handle, cublasOperation_t transA,
    cublasOperation_t transB, int m, int n, int k, Pointer<const void> alpha,
    llvm::ArrayRef<Pointer<const void>> Aarray, cudaDataType typeA, int heightA,
    llvm::ArrayRef<Pointer<const void>> Barray, cudaDataType typeB, int heightB,
    Pointer<const void> beta, llvm::ArrayRef<Pointer<void>> Carray,
    cudaDataType typeC, int heightC, int batchCount, cudaDataType computeType,
    cublasGemmAlgo_t algo);
llvm::Error CublasGemmStridedBatchedEx(
    CurrentContext current, cublasHandle_t handle, cublasOperation_t transA,
    cublasOperation_t transB, int m, int n, int k, Pointer<const void> alpha,
    Pointer<const void> A, cudaDataType typeA, int heightA, int64_t strideA,
    Pointer<const void> B, cudaDataType typeB, int heightB, int64_t strideB,
    Pointer<const void> beta, Pointer<void> C, cudaDataType typeC, int heightC,
    int64_t strideC, int batchCount, cudaDataType computeType,
    cublasGemmAlgo_t algo);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CUBLAS_WRAPPER_H_
