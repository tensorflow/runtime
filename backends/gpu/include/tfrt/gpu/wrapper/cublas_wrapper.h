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

// The functions below are not used and might be removed.

llvm::Error CublasSnrm2(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const float> x, int incx,
                        Pointer<float> result);
llvm::Error CublasDnrm2(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const double> x, int incx,
                        Pointer<double> result);
llvm::Error CublasScnrm2(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuComplex> x, int incx,
                         Pointer<float> result);
llvm::Error CublasDznrm2(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuDoubleComplex> x, int incx,
                         Pointer<double> result);
llvm::Error CublasSdot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<const float> x, int incx, Pointer<const float> y,
                       int incy, Pointer<float> result);
llvm::Error CublasDdot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<const double> x, int incx,
                       Pointer<const double> y, int incy,
                       Pointer<double> result);
llvm::Error CublasCdotu(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> result);
llvm::Error CublasCdotc(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> result);
llvm::Error CublasZdotu(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> result);
llvm::Error CublasZdotc(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> result);
llvm::Error CublasSscal(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const float> alpha, Pointer<float> x, int incx);
llvm::Error CublasDscal(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const double> alpha, Pointer<double> x,
                        int incx);
llvm::Error CublasCscal(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuComplex> alpha, Pointer<cuComplex> x,
                        int incx);
llvm::Error CublasCsscal(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const float> alpha, Pointer<cuComplex> x,
                         int incx);
llvm::Error CublasZscal(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<cuDoubleComplex> x, int incx);
llvm::Error CublasZdscal(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const double> alpha,
                         Pointer<cuDoubleComplex> x, int incx);
llvm::Error CublasSaxpy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const float> alpha, Pointer<const float> x,
                        int incx, Pointer<float> y, int incy);
llvm::Error CublasDaxpy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const double> alpha, Pointer<const double> x,
                        int incx, Pointer<double> y, int incy);
llvm::Error CublasCaxpy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<cuComplex> y, int incy);
llvm::Error CublasZaxpy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<cuDoubleComplex> y, int incy);
llvm::Error CublasScopy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const float> x, int incx, Pointer<float> y,
                        int incy);
llvm::Error CublasDcopy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const double> x, int incx, Pointer<double> y,
                        int incy);
llvm::Error CublasCcopy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<cuComplex> y, int incy);
llvm::Error CublasZcopy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<cuDoubleComplex> y, int incy);
llvm::Error CublasSswap(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<float> x, int incx, Pointer<float> y, int incy);
llvm::Error CublasDswap(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<double> x, int incx, Pointer<double> y,
                        int incy);
llvm::Error CublasCswap(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<cuComplex> x, int incx, Pointer<cuComplex> y,
                        int incy);
llvm::Error CublasZswap(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<cuDoubleComplex> x, int incx,
                        Pointer<cuDoubleComplex> y, int incy);
llvm::Error CublasIsamax(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const float> x, int incx, Pointer<int> result);
llvm::Error CublasIdamax(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const double> x, int incx,
                         Pointer<int> result);
llvm::Error CublasIcamax(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuComplex> x, int incx,
                         Pointer<int> result);
llvm::Error CublasIzamax(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuDoubleComplex> x, int incx,
                         Pointer<int> result);
llvm::Error CublasIsamin(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const float> x, int incx, Pointer<int> result);
llvm::Error CublasIdamin(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const double> x, int incx,
                         Pointer<int> result);
llvm::Error CublasIcamin(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuComplex> x, int incx,
                         Pointer<int> result);
llvm::Error CublasIzamin(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuDoubleComplex> x, int incx,
                         Pointer<int> result);
llvm::Error CublasSasum(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const float> x, int incx,
                        Pointer<float> result);
llvm::Error CublasDasum(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const double> x, int incx,
                        Pointer<double> result);
llvm::Error CublasScasum(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuComplex> x, int incx,
                         Pointer<float> result);
llvm::Error CublasDzasum(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuDoubleComplex> x, int incx,
                         Pointer<double> result);
llvm::Error CublasSrot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<float> x, int incx, Pointer<float> y, int incy,
                       Pointer<const float> c, Pointer<const float> s);
llvm::Error CublasDrot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<double> x, int incx, Pointer<double> y, int incy,
                       Pointer<const double> c, Pointer<const double> s);
llvm::Error CublasCrot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<cuComplex> x, int incx, Pointer<cuComplex> y,
                       int incy, Pointer<const float> c,
                       Pointer<const cuComplex> s);
llvm::Error CublasCsrot(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<cuComplex> x, int incx, Pointer<cuComplex> y,
                        int incy, Pointer<const float> c,
                        Pointer<const float> s);
llvm::Error CublasZrot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<cuDoubleComplex> x, int incx,
                       Pointer<cuDoubleComplex> y, int incy,
                       Pointer<const double> c,
                       Pointer<const cuDoubleComplex> s);
llvm::Error CublasZdrot(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<cuDoubleComplex> x, int incx,
                        Pointer<cuDoubleComplex> y, int incy,
                        Pointer<const double> c, Pointer<const double> s);
llvm::Error CublasSrotg(CurrentContext current, cublasHandle_t handle,
                        Pointer<float> a, Pointer<float> b, Pointer<float> c,
                        Pointer<float> s);
llvm::Error CublasDrotg(CurrentContext current, cublasHandle_t handle,
                        Pointer<double> a, Pointer<double> b, Pointer<double> c,
                        Pointer<double> s);
llvm::Error CublasCrotg(CurrentContext current, cublasHandle_t handle,
                        Pointer<cuComplex> a, Pointer<cuComplex> b,
                        Pointer<float> c, Pointer<cuComplex> s);
llvm::Error CublasZrotg(CurrentContext current, cublasHandle_t handle,
                        Pointer<cuDoubleComplex> a, Pointer<cuDoubleComplex> b,
                        Pointer<double> c, Pointer<cuDoubleComplex> s);
llvm::Error CublasSrotm(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<float> x, int incx, Pointer<float> y, int incy,
                        Pointer<const float> param);
llvm::Error CublasDrotm(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<double> x, int incx, Pointer<double> y,
                        int incy, Pointer<const double> param);
llvm::Error CublasSrotmg(CurrentContext current, cublasHandle_t handle,
                         Pointer<float> d1, Pointer<float> d2,
                         Pointer<float> x1, Pointer<const float> y1,
                         Pointer<float> param);
llvm::Error CublasDrotmg(CurrentContext current, cublasHandle_t handle,
                         Pointer<double> d1, Pointer<double> d2,
                         Pointer<double> x1, Pointer<const double> y1,
                         Pointer<double> param);
llvm::Error CublasSgemv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n,
                        Pointer<const float> alpha, Pointer<const float> A,
                        int lda, Pointer<const float> x, int incx,
                        Pointer<const float> beta, Pointer<float> y, int incy);
llvm::Error CublasDgemv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n,
                        Pointer<const double> alpha, Pointer<const double> A,
                        int lda, Pointer<const double> x, int incx,
                        Pointer<const double> beta, Pointer<double> y,
                        int incy);
llvm::Error CublasCgemv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> y,
                        int incy);
llvm::Error CublasZgemv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy);
llvm::Error CublasSgbmv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n, int kl, int ku,
                        Pointer<const float> alpha, Pointer<const float> A,
                        int lda, Pointer<const float> x, int incx,
                        Pointer<const float> beta, Pointer<float> y, int incy);
llvm::Error CublasDgbmv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n, int kl, int ku,
                        Pointer<const double> alpha, Pointer<const double> A,
                        int lda, Pointer<const double> x, int incx,
                        Pointer<const double> beta, Pointer<double> y,
                        int incy);
llvm::Error CublasCgbmv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n, int kl, int ku,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> y,
                        int incy);
llvm::Error CublasZgbmv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n, int kl, int ku,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy);
llvm::Error CublasStrmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const float> A,
                        int lda, Pointer<float> x, int incx);
llvm::Error CublasDtrmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const double> A,
                        int lda, Pointer<double> x, int incx);
llvm::Error CublasCtrmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<cuComplex> x, int incx);
llvm::Error CublasZtrmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<cuDoubleComplex> x, int incx);
llvm::Error CublasStbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const float> A, int lda, Pointer<float> x,
                        int incx);
llvm::Error CublasDtbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const double> A, int lda, Pointer<double> x,
                        int incx);
llvm::Error CublasCtbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<cuComplex> x, int incx);
llvm::Error CublasZtbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<cuDoubleComplex> x, int incx);
llvm::Error CublasStpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const float> AP,
                        Pointer<float> x, int incx);
llvm::Error CublasDtpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const double> AP,
                        Pointer<double> x, int incx);
llvm::Error CublasCtpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuComplex> AP, Pointer<cuComplex> x,
                        int incx);
llvm::Error CublasZtpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuDoubleComplex> AP,
                        Pointer<cuDoubleComplex> x, int incx);
llvm::Error CublasStrsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const float> A,
                        int lda, Pointer<float> x, int incx);
llvm::Error CublasDtrsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const double> A,
                        int lda, Pointer<double> x, int incx);
llvm::Error CublasCtrsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<cuComplex> x, int incx);
llvm::Error CublasZtrsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<cuDoubleComplex> x, int incx);
llvm::Error CublasStpsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const float> AP,
                        Pointer<float> x, int incx);
llvm::Error CublasDtpsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const double> AP,
                        Pointer<double> x, int incx);
llvm::Error CublasCtpsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuComplex> AP, Pointer<cuComplex> x,
                        int incx);
llvm::Error CublasZtpsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuDoubleComplex> AP,
                        Pointer<cuDoubleComplex> x, int incx);
llvm::Error CublasStbsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const float> A, int lda, Pointer<float> x,
                        int incx);
llvm::Error CublasDtbsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const double> A, int lda, Pointer<double> x,
                        int incx);
llvm::Error CublasCtbsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<cuComplex> x, int incx);
llvm::Error CublasZtbsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<cuDoubleComplex> x, int incx);
llvm::Error CublasSsymv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const float> alpha, Pointer<const float> A,
                        int lda, Pointer<const float> x, int incx,
                        Pointer<const float> beta, Pointer<float> y, int incy);
llvm::Error CublasDsymv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const double> alpha, Pointer<const double> A,
                        int lda, Pointer<const double> x, int incx,
                        Pointer<const double> beta, Pointer<double> y,
                        int incy);
llvm::Error CublasCsymv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> y,
                        int incy);
llvm::Error CublasZsymv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy);
llvm::Error CublasChemv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> y,
                        int incy);
llvm::Error CublasZhemv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy);
llvm::Error CublasSsbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n, int k,
                        Pointer<const float> alpha, Pointer<const float> A,
                        int lda, Pointer<const float> x, int incx,
                        Pointer<const float> beta, Pointer<float> y, int incy);
llvm::Error CublasDsbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n, int k,
                        Pointer<const double> alpha, Pointer<const double> A,
                        int lda, Pointer<const double> x, int incx,
                        Pointer<const double> beta, Pointer<double> y,
                        int incy);
llvm::Error CublasChbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n, int k,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> y,
                        int incy);
llvm::Error CublasZhbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n, int k,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy);
llvm::Error CublasSspmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const float> alpha, Pointer<const float> AP,
                        Pointer<const float> x, int incx,
                        Pointer<const float> beta, Pointer<float> y, int incy);
llvm::Error CublasDspmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const double> alpha, Pointer<const double> AP,
                        Pointer<const double> x, int incx,
                        Pointer<const double> beta, Pointer<double> y,
                        int incy);
llvm::Error CublasChpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> AP, Pointer<const cuComplex> x,
                        int incx, Pointer<const cuComplex> beta,
                        Pointer<cuComplex> y, int incy);
llvm::Error CublasZhpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> AP,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy);
llvm::Error CublasSger(CurrentContext current, cublasHandle_t handle, int m,
                       int n, Pointer<const float> alpha,
                       Pointer<const float> x, int incx, Pointer<const float> y,
                       int incy, Pointer<float> A, int lda);
llvm::Error CublasDger(CurrentContext current, cublasHandle_t handle, int m,
                       int n, Pointer<const double> alpha,
                       Pointer<const double> x, int incx,
                       Pointer<const double> y, int incy, Pointer<double> A,
                       int lda);
llvm::Error CublasCgeru(CurrentContext current, cublasHandle_t handle, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> A, int lda);
llvm::Error CublasCgerc(CurrentContext current, cublasHandle_t handle, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> A, int lda);
llvm::Error CublasZgeru(CurrentContext current, cublasHandle_t handle, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> A, int lda);
llvm::Error CublasZgerc(CurrentContext current, cublasHandle_t handle, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> A, int lda);
llvm::Error CublasSsyr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n, Pointer<const float> alpha,
                       Pointer<const float> x, int incx, Pointer<float> A,
                       int lda);
llvm::Error CublasDsyr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const double> alpha, Pointer<const double> x,
                       int incx, Pointer<double> A, int lda);
llvm::Error CublasCsyr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const cuComplex> alpha,
                       Pointer<const cuComplex> x, int incx,
                       Pointer<cuComplex> A, int lda);
llvm::Error CublasZsyr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const cuDoubleComplex> alpha,
                       Pointer<const cuDoubleComplex> x, int incx,
                       Pointer<cuDoubleComplex> A, int lda);
llvm::Error CublasCher(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n, Pointer<const float> alpha,
                       Pointer<const cuComplex> x, int incx,
                       Pointer<cuComplex> A, int lda);
llvm::Error CublasZher(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const double> alpha,
                       Pointer<const cuDoubleComplex> x, int incx,
                       Pointer<cuDoubleComplex> A, int lda);
llvm::Error CublasSspr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n, Pointer<const float> alpha,
                       Pointer<const float> x, int incx, Pointer<float> AP);
llvm::Error CublasDspr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const double> alpha, Pointer<const double> x,
                       int incx, Pointer<double> AP);
llvm::Error CublasChpr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n, Pointer<const float> alpha,
                       Pointer<const cuComplex> x, int incx,
                       Pointer<cuComplex> AP);
llvm::Error CublasZhpr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const double> alpha,
                       Pointer<const cuDoubleComplex> x, int incx,
                       Pointer<cuDoubleComplex> AP);
llvm::Error CublasSsyr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const float> alpha, Pointer<const float> x,
                        int incx, Pointer<const float> y, int incy,
                        Pointer<float> A, int lda);
llvm::Error CublasDsyr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const double> alpha, Pointer<const double> x,
                        int incx, Pointer<const double> y, int incy,
                        Pointer<double> A, int lda);
llvm::Error CublasCsyr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> A, int lda);
llvm::Error CublasZsyr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> A, int lda);
llvm::Error CublasCher2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> A, int lda);
llvm::Error CublasZher2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> A, int lda);
llvm::Error CublasSspr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const float> alpha, Pointer<const float> x,
                        int incx, Pointer<const float> y, int incy,
                        Pointer<float> AP);
llvm::Error CublasDspr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const double> alpha, Pointer<const double> x,
                        int incx, Pointer<const double> y, int incy,
                        Pointer<double> AP);
llvm::Error CublasChpr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> AP);
llvm::Error CublasZhpr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> AP);
llvm::Error CublasHgemm(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k, Pointer<const __half> alpha,
                        Pointer<const __half> A, int lda,
                        Pointer<const __half> B, int ldb,
                        Pointer<const __half> beta, Pointer<__half> C, int ldc);
llvm::Error CublasSgemm(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k, Pointer<const float> alpha,
                        Pointer<const float> A, int lda, Pointer<const float> B,
                        int ldb, Pointer<const float> beta, Pointer<float> C,
                        int ldc);
llvm::Error CublasDgemm(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k, Pointer<const double> alpha,
                        Pointer<const double> A, int lda,
                        Pointer<const double> B, int ldb,
                        Pointer<const double> beta, Pointer<double> C, int ldc);
llvm::Error CublasCgemm(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> B, int ldb,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> C,
                        int ldc);
llvm::Error CublasZgemm(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> B, int ldb,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> C, int ldc);
llvm::Error CublasSsyrk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const float> alpha,
                        Pointer<const float> A, int lda,
                        Pointer<const float> beta, Pointer<float> C, int ldc);
llvm::Error CublasDsyrk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const double> alpha,
                        Pointer<const double> A, int lda,
                        Pointer<const double> beta, Pointer<double> C, int ldc);
llvm::Error CublasCsyrk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> C,
                        int ldc);
llvm::Error CublasZsyrk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> C, int ldc);
llvm::Error CublasCherk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const float> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const float> beta, Pointer<cuComplex> C,
                        int ldc);
llvm::Error CublasZherk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const double> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const double> beta, Pointer<cuDoubleComplex> C,
                        int ldc);
llvm::Error CublasSsyr2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const float> alpha,
                         Pointer<const float> A, int lda,
                         Pointer<const float> B, int ldb,
                         Pointer<const float> beta, Pointer<float> C, int ldc);
llvm::Error CublasDsyr2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const double> alpha,
                         Pointer<const double> A, int lda,
                         Pointer<const double> B, int ldb,
                         Pointer<const double> beta, Pointer<double> C,
                         int ldc);
llvm::Error CublasCsyr2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const cuComplex> alpha,
                         Pointer<const cuComplex> A, int lda,
                         Pointer<const cuComplex> B, int ldb,
                         Pointer<const cuComplex> beta, Pointer<cuComplex> C,
                         int ldc);
llvm::Error CublasZsyr2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const cuDoubleComplex> alpha,
                         Pointer<const cuDoubleComplex> A, int lda,
                         Pointer<const cuDoubleComplex> B, int ldb,
                         Pointer<const cuDoubleComplex> beta,
                         Pointer<cuDoubleComplex> C, int ldc);
llvm::Error CublasCher2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const cuComplex> alpha,
                         Pointer<const cuComplex> A, int lda,
                         Pointer<const cuComplex> B, int ldb,
                         Pointer<const float> beta, Pointer<cuComplex> C,
                         int ldc);
llvm::Error CublasZher2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const cuDoubleComplex> alpha,
                         Pointer<const cuDoubleComplex> A, int lda,
                         Pointer<const cuDoubleComplex> B, int ldb,
                         Pointer<const double> beta, Pointer<cuDoubleComplex> C,
                         int ldc);
llvm::Error CublasSsymm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const float> alpha,
                        Pointer<const float> A, int lda, Pointer<const float> B,
                        int ldb, Pointer<const float> beta, Pointer<float> C,
                        int ldc);
llvm::Error CublasDsymm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const double> alpha,
                        Pointer<const double> A, int lda,
                        Pointer<const double> B, int ldb,
                        Pointer<const double> beta, Pointer<double> C, int ldc);
llvm::Error CublasCsymm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> B, int ldb,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> C,
                        int ldc);
llvm::Error CublasZsymm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> B, int ldb,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> C, int ldc);
llvm::Error CublasChemm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> B, int ldb,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> C,
                        int ldc);
llvm::Error CublasZhemm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> B, int ldb,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> C, int ldc);
llvm::Error CublasStrsm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const float> alpha,
                        Pointer<const float> A, int lda, Pointer<float> B,
                        int ldb);
llvm::Error CublasDtrsm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const double> alpha,
                        Pointer<const double> A, int lda, Pointer<double> B,
                        int ldb);
llvm::Error CublasCtrsm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<cuComplex> B, int ldb);
llvm::Error CublasZtrsm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<cuDoubleComplex> B, int ldb);
llvm::Error CublasStrmm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const float> alpha,
                        Pointer<const float> A, int lda, Pointer<const float> B,
                        int ldb, Pointer<float> C, int ldc);
llvm::Error CublasDtrmm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const double> alpha,
                        Pointer<const double> A, int lda,
                        Pointer<const double> B, int ldb, Pointer<double> C,
                        int ldc);
llvm::Error CublasCtrmm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> B, int ldb,
                        Pointer<cuComplex> C, int ldc);
llvm::Error CublasZtrmm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> B, int ldb,
                        Pointer<cuDoubleComplex> C, int ldc);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CUBLAS_WRAPPER_H_
