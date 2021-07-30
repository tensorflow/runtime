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

// Thin wrapper around the cuBLAS API adding llvm::Error.
#include "tfrt/gpu/wrapper/cublas_wrapper.h"

#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

template llvm::raw_ostream& internal::operator<<(
    llvm::raw_ostream&, const ErrorData<cublasStatus_t>&);

llvm::Expected<OwningBlasHandle> CublasCreate(CurrentContext current) {
  CheckCudaContext(current);
  cublasHandle_t handle = nullptr;
  RETURN_IF_ERROR(cublasCreate_v2(&handle));
  return OwningBlasHandle(handle);
}

llvm::Error CublasDestroy(cublasHandle_t handle) {
  return TO_ERROR(cublasDestroy_v2(handle));
}

llvm::Expected<int> CublasGetVersion(cublasHandle_t handle) {
  int version = 0;
  RETURN_IF_ERROR(cublasGetVersion_v2(handle, &version));
  return version;
}

llvm::Error CublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
  return TO_ERROR(cublasSetStream_v2(handle, stream));
}

llvm::Expected<Stream> CublasGetStream(cublasHandle_t handle) {
  cudaStream_t stream = nullptr;
  RETURN_IF_ERROR(cublasGetStream_v2(handle, &stream));
  return Stream(stream);
}

llvm::Error CublasSetPointerMode(cublasHandle_t handle,
                                 cublasPointerMode_t mode) {
  return TO_ERROR(cublasSetPointerMode_v2(handle, mode));
}

llvm::Expected<cublasPointerMode_t> CublasGetPointerMode(
    cublasHandle_t handle) {
  cublasPointerMode_t mode;
  RETURN_IF_ERROR(cublasGetPointerMode_v2(handle, &mode));
  return mode;
}

llvm::Error CublasSetMathMode(cublasHandle_t handle, cublasMath_t math_type) {
  return TO_ERROR(cublasSetMathMode(handle, math_type));
}

llvm::Expected<cublasMath_t> CublasGetMathMode(cublasHandle_t handle) {
  cublasMath_t mode;
  RETURN_IF_ERROR(cublasGetMathMode(handle, &mode));
  return mode;
}

llvm::Error CublasAxpyEx(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const void> alpha, /* host or device pointer */
                         cudaDataType alphaType, Pointer<const void> x,
                         cudaDataType typeX, int strideX, Pointer<void> y,
                         cudaDataType typeY, int strideY,
                         cudaDataType executionType) {
  CheckCudaContext(current);
  return TO_ERROR(cublasAxpyEx(handle, n, ToCuda(alpha), alphaType, ToCuda(x),
                               typeX, strideX, ToCuda(y), typeY, strideY,
                               executionType));
}

// This function is defined in cublas_stub.cc (or cublas_compat.cc for google-
// internal builds). It is a backward compartible wrapper for the cublasGemmEx
// to accomodate for the API change between cuBLAS v10 and v11.
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx_v10(
    cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
    int m, int n, int k, const void* alpha, /* host or device pointer */
    const void* A, cudaDataType typeA, int heightA, const void* B,
    cudaDataType typeB, int heightB,
    const void* beta, /* host or device pointer */
    void* C, cudaDataType typeC, int heightC, cudaDataType computeType,
    cublasGemmAlgo_t algo);

llvm::Error CublasGemmEx(CurrentContext current, cublasHandle_t handle,
                         cublasOperation_t transA, cublasOperation_t transB,
                         int m, int n, int k, Pointer<const void> alpha,
                         Pointer<const void> A, cudaDataType typeA, int heightA,
                         Pointer<const void> B, cudaDataType typeB, int heightB,
                         Pointer<const void> beta, Pointer<void> C,
                         cudaDataType typeC, int heightC,
                         cudaDataType computeType, cublasGemmAlgo_t algo) {
  CheckCudaContext(current);
  return TO_ERROR(cublasGemmEx_v10(
      handle, transA, transB, m, n, k, ToCuda(alpha), ToCuda(A), typeA, heightA,
      ToCuda(B), typeB, heightB, ToCuda(beta), ToCuda(C), typeC, heightC,
      computeType, algo));
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx_v10(
    cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
    int m, int n, int k, const void* alpha, /* host or device pointer */
    const void* Aarray[], cudaDataType typeA, int heightA, const void* Barray[],
    cudaDataType typeB, int heightB,
    const void* beta, /* host or device pointer */
    void* Carray[], cudaDataType typeC, int heightC, int batchCount,
    cudaDataType computeType, cublasGemmAlgo_t algo);

llvm::Error CublasGemmBatchedEx(
    CurrentContext current, cublasHandle_t handle, cublasOperation_t transA,
    cublasOperation_t transB, int m, int n, int k, Pointer<const void> alpha,
    llvm::ArrayRef<Pointer<const void>> Aarray, cudaDataType typeA, int heightA,
    llvm::ArrayRef<Pointer<const void>> Barray, cudaDataType typeB, int heightB,
    Pointer<const void> beta, llvm::ArrayRef<Pointer<void>> Carray,
    cudaDataType typeC, int heightC, int batchCount, cudaDataType computeType,
    cublasGemmAlgo_t algo) {
  CheckCudaContext(current);
  return TO_ERROR(cublasGemmBatchedEx_v10(
      handle, transA, transB, m, n, k, ToCuda(alpha), ToCuda(Aarray).data(),
      typeA, heightA, ToCuda(Barray).data(), typeB, heightB, ToCuda(beta),
      ToCuda(Carray).data(), typeC, heightC, batchCount, computeType, algo));
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx_v10(
    cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
    int m, int n, int k, const void* alpha, /* host or device pointer */
    const void* A, cudaDataType typeA, int heightA, int64_t strideA,
    const void* B, cudaDataType typeB, int heightB, int64_t strideB,
    const void* beta, /* host or device pointer */
    void* C, cudaDataType typeC, int heightC, int64_t strideC, int batchCount,
    cudaDataType computeType, cublasGemmAlgo_t algo);

llvm::Error CublasGemmStridedBatchedEx(
    CurrentContext current, cublasHandle_t handle, cublasOperation_t transA,
    cublasOperation_t transB, int m, int n, int k, Pointer<const void> alpha,
    Pointer<const void> A, cudaDataType typeA, int heightA, int64_t strideA,
    Pointer<const void> B, cudaDataType typeB, int heightB, int64_t strideB,
    Pointer<const void> beta, Pointer<void> C, cudaDataType typeC, int heightC,
    int64_t strideC, int batchCount, cudaDataType computeType,
    cublasGemmAlgo_t algo) {
  CheckCudaContext(current);
  return TO_ERROR(cublasGemmStridedBatchedEx_v10(
      handle, transA, transB, m, n, k, ToCuda(alpha), ToCuda(A), typeA, heightA,
      strideA, ToCuda(B), typeB, heightB, strideB, ToCuda(beta), ToCuda(C),
      typeC, heightC, strideC, batchCount, computeType, algo));
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
