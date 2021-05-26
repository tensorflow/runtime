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

// Thin wrapper around the rocBLAS API adding llvm::Error.
#ifndef TFRT_GPU_WRAPPER_ROCBLAS_WRAPPER_H_
#define TFRT_GPU_WRAPPER_ROCBLAS_WRAPPER_H_

#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/rocblas_stub.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_status status);

template <>
Expected<rocblas_datatype> Parse<rocblas_datatype>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_datatype value);

template <>
Expected<rocblas_operation> Parse<rocblas_operation>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_operation value);

template <>
Expected<rocblas_gemm_algo> Parse<rocblas_gemm_algo>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_gemm_algo value);

template <>
struct PlatformTypeTraits<BlasDataTypeTag, rocblas_datatype>
    : public CudaPlatformType {};
template <>
struct PlatformTypeTraits<BlasOperationTag, rocblas_operation>
    : public CudaPlatformType {};
template <>
struct PlatformTypeTraits<BlasGemmAlgoTag, rocblas_gemm_algo>
    : public CudaPlatformType {};

llvm::Expected<OwningBlasHandle> RocblasCreate(CurrentContext current);
llvm::Error RocblasDestroy(rocblas_handle handle);
llvm::Error RocblasSetStream(rocblas_handle handle, hipStream_t stream);
llvm::Expected<Stream> RocblasGetStream(rocblas_handle handle);
llvm::Error RocblasSetPointerMode(rocblas_handle handle,
                                  rocblas_pointer_mode mode);
llvm::Expected<rocblas_pointer_mode> RocblasGetPointerMode(
    rocblas_handle handle);

llvm::Error RocblasAxpyEx(
    CurrentContext current, rocblas_handle handle, int n,
    Pointer<const void> alpha, /* host or device pointer */
    rocblas_datatype alphaType, Pointer<const void> x, rocblas_datatype typeX,
    int strideX, Pointer<void> y, rocblas_datatype typeY, int strideY,
    rocblas_datatype executionType);

llvm::Error RocblasGemmEx(CurrentContext current, rocblas_handle handle,
                          rocblas_operation transA, rocblas_operation transB,
                          int m, int n, int k, Pointer<const void> alpha,
                          Pointer<const void> A, rocblas_datatype typeA,
                          int heightA, Pointer<const void> B,
                          rocblas_datatype typeB, int heightB,
                          Pointer<const void> beta, Pointer<const void> C,
                          rocblas_datatype typeC, int heightC, Pointer<void> D,
                          rocblas_datatype typeD, int heightD,
                          rocblas_datatype computeType, rocblas_gemm_algo algo);
llvm::Error RocblasGemmStridedBatchedEx(
    CurrentContext current, rocblas_handle handle, rocblas_operation transA,
    rocblas_operation transB, int m, int n, int k, Pointer<const void> alpha,
    Pointer<const void> A, rocblas_datatype typeA, int heightA, int64_t strideA,
    Pointer<const void> B, rocblas_datatype typeB, int heightB, int64_t strideB,
    Pointer<const void> beta, Pointer<void> C, rocblas_datatype typeC,
    int heightC, int64_t strideC, Pointer<void> D, rocblas_datatype typeD,
    int heightD, int64_t strideD, int batchCount, rocblas_datatype computeType,
    rocblas_gemm_algo algo);

// The functions below are not used and might be removed.

llvm::Error RocblasSnrm2(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const float> x, int incx,
                         Pointer<float> result);
llvm::Error RocblasDnrm2(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const double> x, int incx,
                         Pointer<double> result);
llvm::Error RocblasScnrm2(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_float_complex> x, int incx,
                          Pointer<float> result);
llvm::Error RocblasDznrm2(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_double_complex> x, int incx,
                          Pointer<double> result);
llvm::Error RocblasSdot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<const float> x, int incx,
                        Pointer<const float> y, int incy,
                        Pointer<float> result);
llvm::Error RocblasDdot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<const double> x, int incx,
                        Pointer<const double> y, int incy,
                        Pointer<double> result);
llvm::Error RocblasCdotu(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> result);
llvm::Error RocblasCdotc(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> result);
llvm::Error RocblasZdotu(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> result);
llvm::Error RocblasZdotc(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> result);
llvm::Error RocblasSscal(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const float> alpha, Pointer<float> x,
                         int incx);
llvm::Error RocblasDscal(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const double> alpha, Pointer<double> x,
                         int incx);
llvm::Error RocblasCscal(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<rocblas_float_complex> x, int incx);
llvm::Error RocblasCsscal(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const float> alpha,
                          Pointer<rocblas_float_complex> x, int incx);
llvm::Error RocblasZscal(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<rocblas_double_complex> x, int incx);
llvm::Error RocblasZdscal(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const double> alpha,
                          Pointer<rocblas_double_complex> x, int incx);
llvm::Error RocblasSaxpy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const float> alpha, Pointer<const float> x,
                         int incx, Pointer<float> y, int incy);
llvm::Error RocblasDaxpy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const double> alpha, Pointer<const double> x,
                         int incx, Pointer<double> y, int incy);
llvm::Error RocblasCaxpy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<rocblas_float_complex> y, int incy);
llvm::Error RocblasZaxpy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<rocblas_double_complex> y, int incy);
llvm::Error RocblasScopy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const float> x, int incx, Pointer<float> y,
                         int incy);
llvm::Error RocblasDcopy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const double> x, int incx, Pointer<double> y,
                         int incy);
llvm::Error RocblasCcopy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<rocblas_float_complex> y, int incy);
llvm::Error RocblasZcopy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<rocblas_double_complex> y, int incy);
llvm::Error RocblasSswap(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<float> x, int incx, Pointer<float> y,
                         int incy);
llvm::Error RocblasDswap(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<double> x, int incx, Pointer<double> y,
                         int incy);
llvm::Error RocblasCswap(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<rocblas_float_complex> x, int incx,
                         Pointer<rocblas_float_complex> y, int incy);
llvm::Error RocblasZswap(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<rocblas_double_complex> x, int incx,
                         Pointer<rocblas_double_complex> y, int incy);
llvm::Error RocblasIsamax(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const float> x, int incx,
                          Pointer<int> result);
llvm::Error RocblasIdamax(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const double> x, int incx,
                          Pointer<int> result);
llvm::Error RocblasIcamax(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_float_complex> x, int incx,
                          Pointer<int> result);
llvm::Error RocblasIzamax(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_double_complex> x, int incx,
                          Pointer<int> result);
llvm::Error RocblasIsamin(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const float> x, int incx,
                          Pointer<int> result);
llvm::Error RocblasIdamin(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const double> x, int incx,
                          Pointer<int> result);
llvm::Error RocblasIcamin(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_float_complex> x, int incx,
                          Pointer<int> result);
llvm::Error RocblasIzamin(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_double_complex> x, int incx,
                          Pointer<int> result);
llvm::Error RocblasSasum(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const float> x, int incx,
                         Pointer<float> result);
llvm::Error RocblasDasum(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const double> x, int incx,
                         Pointer<double> result);
llvm::Error RocblasScasum(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_float_complex> x, int incx,
                          Pointer<float> result);
llvm::Error RocblasDzasum(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_double_complex> x, int incx,
                          Pointer<double> result);
llvm::Error RocblasSrot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<float> x, int incx, Pointer<float> y, int incy,
                        Pointer<const float> c, Pointer<const float> s);
llvm::Error RocblasDrot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<double> x, int incx, Pointer<double> y,
                        int incy, Pointer<const double> c,
                        Pointer<const double> s);
llvm::Error RocblasCrot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<rocblas_float_complex> x, int incx,
                        Pointer<rocblas_float_complex> y, int incy,
                        Pointer<const float> c,
                        Pointer<const rocblas_float_complex> s);
llvm::Error RocblasCsrot(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<rocblas_float_complex> x, int incx,
                         Pointer<rocblas_float_complex> y, int incy,
                         Pointer<const float> c, Pointer<const float> s);
llvm::Error RocblasZrot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<rocblas_double_complex> x, int incx,
                        Pointer<rocblas_double_complex> y, int incy,
                        Pointer<const double> c,
                        Pointer<const rocblas_double_complex> s);
llvm::Error RocblasZdrot(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<rocblas_double_complex> x, int incx,
                         Pointer<rocblas_double_complex> y, int incy,
                         Pointer<const double> c, Pointer<const double> s);
llvm::Error RocblasSrotg(CurrentContext current, rocblas_handle handle,
                         Pointer<float> a, Pointer<float> b, Pointer<float> c,
                         Pointer<float> s);
llvm::Error RocblasDrotg(CurrentContext current, rocblas_handle handle,
                         Pointer<double> a, Pointer<double> b,
                         Pointer<double> c, Pointer<double> s);
llvm::Error RocblasCrotg(CurrentContext current, rocblas_handle handle,
                         Pointer<rocblas_float_complex> a,
                         Pointer<rocblas_float_complex> b, Pointer<float> c,
                         Pointer<rocblas_float_complex> s);
llvm::Error RocblasZrotg(CurrentContext current, rocblas_handle handle,
                         Pointer<rocblas_double_complex> a,
                         Pointer<rocblas_double_complex> b, Pointer<double> c,
                         Pointer<rocblas_double_complex> s);
llvm::Error RocblasSrotm(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<float> x, int incx, Pointer<float> y, int incy,
                         Pointer<const float> param);
llvm::Error RocblasDrotm(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<double> x, int incx, Pointer<double> y,
                         int incy, Pointer<const double> param);
llvm::Error RocblasSrotmg(CurrentContext current, rocblas_handle handle,
                          Pointer<float> d1, Pointer<float> d2,
                          Pointer<float> x1, Pointer<const float> y1,
                          Pointer<float> param);
llvm::Error RocblasDrotmg(CurrentContext current, rocblas_handle handle,
                          Pointer<double> d1, Pointer<double> d2,
                          Pointer<double> x1, Pointer<const double> y1,
                          Pointer<double> param);
llvm::Error RocblasSgemv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n,
                         Pointer<const float> alpha, Pointer<const float> A,
                         int lda, Pointer<const float> x, int incx,
                         Pointer<const float> beta, Pointer<float> y, int incy);
llvm::Error RocblasDgemv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n,
                         Pointer<const double> alpha, Pointer<const double> A,
                         int lda, Pointer<const double> x, int incx,
                         Pointer<const double> beta, Pointer<double> y,
                         int incy);
llvm::Error RocblasCgemv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy);
llvm::Error RocblasZgemv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy);
llvm::Error RocblasSgbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n, int kl, int ku,
                         Pointer<const float> alpha, Pointer<const float> A,
                         int lda, Pointer<const float> x, int incx,
                         Pointer<const float> beta, Pointer<float> y, int incy);
llvm::Error RocblasDgbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n, int kl, int ku,
                         Pointer<const double> alpha, Pointer<const double> A,
                         int lda, Pointer<const double> x, int incx,
                         Pointer<const double> beta, Pointer<double> y,
                         int incy);
llvm::Error RocblasCgbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n, int kl, int ku,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy);
llvm::Error RocblasZgbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n, int kl, int ku,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy);
llvm::Error RocblasStrmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const float> A,
                         int lda, Pointer<float> x, int incx);
llvm::Error RocblasDtrmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const double> A,
                         int lda, Pointer<double> x, int incx);
llvm::Error RocblasCtrmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> x, int incx);
llvm::Error RocblasZtrmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> x, int incx);
llvm::Error RocblasStbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const float> A, int lda, Pointer<float> x,
                         int incx);
llvm::Error RocblasDtbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const double> A, int lda, Pointer<double> x,
                         int incx);
llvm::Error RocblasCtbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> x, int incx);
llvm::Error RocblasZtbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> x, int incx);
llvm::Error RocblasStpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const float> AP,
                         Pointer<float> x, int incx);
llvm::Error RocblasDtpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const double> AP,
                         Pointer<double> x, int incx);
llvm::Error RocblasCtpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_float_complex> AP,
                         Pointer<rocblas_float_complex> x, int incx);
llvm::Error RocblasZtpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_double_complex> AP,
                         Pointer<rocblas_double_complex> x, int incx);
llvm::Error RocblasStrsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const float> A,
                         int lda, Pointer<float> x, int incx);
llvm::Error RocblasDtrsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const double> A,
                         int lda, Pointer<double> x, int incx);
llvm::Error RocblasCtrsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> x, int incx);
llvm::Error RocblasZtrsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> x, int incx);
llvm::Error RocblasStpsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const float> AP,
                         Pointer<float> x, int incx);
llvm::Error RocblasDtpsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const double> AP,
                         Pointer<double> x, int incx);
llvm::Error RocblasCtpsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_float_complex> AP,
                         Pointer<rocblas_float_complex> x, int incx);
llvm::Error RocblasZtpsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_double_complex> AP,
                         Pointer<rocblas_double_complex> x, int incx);
llvm::Error RocblasStbsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const float> A, int lda, Pointer<float> x,
                         int incx);
llvm::Error RocblasDtbsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const double> A, int lda, Pointer<double> x,
                         int incx);
llvm::Error RocblasCtbsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> x, int incx);
llvm::Error RocblasZtbsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> x, int incx);
llvm::Error RocblasSsymv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const float> alpha,
                         Pointer<const float> A, int lda,
                         Pointer<const float> x, int incx,
                         Pointer<const float> beta, Pointer<float> y, int incy);
llvm::Error RocblasDsymv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const double> alpha,
                         Pointer<const double> A, int lda,
                         Pointer<const double> x, int incx,
                         Pointer<const double> beta, Pointer<double> y,
                         int incy);
llvm::Error RocblasCsymv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy);
llvm::Error RocblasZsymv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy);
llvm::Error RocblasChemv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy);
llvm::Error RocblasZhemv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy);
llvm::Error RocblasSsbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, int k,
                         Pointer<const float> alpha, Pointer<const float> A,
                         int lda, Pointer<const float> x, int incx,
                         Pointer<const float> beta, Pointer<float> y, int incy);
llvm::Error RocblasDsbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, int k,
                         Pointer<const double> alpha, Pointer<const double> A,
                         int lda, Pointer<const double> x, int incx,
                         Pointer<const double> beta, Pointer<double> y,
                         int incy);
llvm::Error RocblasChbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, int k,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy);
llvm::Error RocblasZhbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, int k,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy);
llvm::Error RocblasSspmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const float> alpha,
                         Pointer<const float> AP, Pointer<const float> x,
                         int incx, Pointer<const float> beta, Pointer<float> y,
                         int incy);
llvm::Error RocblasDspmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const double> alpha,
                         Pointer<const double> AP, Pointer<const double> x,
                         int incx, Pointer<const double> beta,
                         Pointer<double> y, int incy);
llvm::Error RocblasChpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> AP,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy);
llvm::Error RocblasZhpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> AP,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy);
llvm::Error RocblasSger(CurrentContext current, rocblas_handle handle, int m,
                        int n, Pointer<const float> alpha,
                        Pointer<const float> x, int incx,
                        Pointer<const float> y, int incy, Pointer<float> A,
                        int lda);
llvm::Error RocblasDger(CurrentContext current, rocblas_handle handle, int m,
                        int n, Pointer<const double> alpha,
                        Pointer<const double> x, int incx,
                        Pointer<const double> y, int incy, Pointer<double> A,
                        int lda);
llvm::Error RocblasCgeru(CurrentContext current, rocblas_handle handle, int m,
                         int n, Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> A, int lda);
llvm::Error RocblasCgerc(CurrentContext current, rocblas_handle handle, int m,
                         int n, Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> A, int lda);
llvm::Error RocblasZgeru(CurrentContext current, rocblas_handle handle, int m,
                         int n, Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> A, int lda);
llvm::Error RocblasZgerc(CurrentContext current, rocblas_handle handle, int m,
                         int n, Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> A, int lda);
llvm::Error RocblasSsyr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const float> alpha,
                        Pointer<const float> x, int incx, Pointer<float> A,
                        int lda);
llvm::Error RocblasDsyr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const double> alpha,
                        Pointer<const double> x, int incx, Pointer<double> A,
                        int lda);
llvm::Error RocblasCsyr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n,
                        Pointer<const rocblas_float_complex> alpha,
                        Pointer<const rocblas_float_complex> x, int incx,
                        Pointer<rocblas_float_complex> A, int lda);
llvm::Error RocblasZsyr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n,
                        Pointer<const rocblas_double_complex> alpha,
                        Pointer<const rocblas_double_complex> x, int incx,
                        Pointer<rocblas_double_complex> A, int lda);
llvm::Error RocblasCher(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const float> alpha,
                        Pointer<const rocblas_float_complex> x, int incx,
                        Pointer<rocblas_float_complex> A, int lda);
llvm::Error RocblasZher(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const double> alpha,
                        Pointer<const rocblas_double_complex> x, int incx,
                        Pointer<rocblas_double_complex> A, int lda);
llvm::Error RocblasSspr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const float> alpha,
                        Pointer<const float> x, int incx, Pointer<float> AP);
llvm::Error RocblasDspr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const double> alpha,
                        Pointer<const double> x, int incx, Pointer<double> AP);
llvm::Error RocblasChpr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const float> alpha,
                        Pointer<const rocblas_float_complex> x, int incx,
                        Pointer<rocblas_float_complex> AP);
llvm::Error RocblasZhpr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const double> alpha,
                        Pointer<const rocblas_double_complex> x, int incx,
                        Pointer<rocblas_double_complex> AP);
llvm::Error RocblasSsyr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const float> alpha,
                         Pointer<const float> x, int incx,
                         Pointer<const float> y, int incy, Pointer<float> A,
                         int lda);
llvm::Error RocblasDsyr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const double> alpha,
                         Pointer<const double> x, int incx,
                         Pointer<const double> y, int incy, Pointer<double> A,
                         int lda);
llvm::Error RocblasCsyr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> A, int lda);
llvm::Error RocblasZsyr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> A, int lda);
llvm::Error RocblasCher2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> A, int lda);
llvm::Error RocblasZher2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> A, int lda);
llvm::Error RocblasSspr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const float> alpha,
                         Pointer<const float> x, int incx,
                         Pointer<const float> y, int incy, Pointer<float> AP);
llvm::Error RocblasDspr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const double> alpha,
                         Pointer<const double> x, int incx,
                         Pointer<const double> y, int incy, Pointer<double> AP);
llvm::Error RocblasChpr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> AP);
llvm::Error RocblasZhpr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> AP);
llvm::Error RocblasHgemm(CurrentContext current, rocblas_handle handle,
                         rocblas_operation transa, rocblas_operation transb,
                         int m, int n, int k, Pointer<const rocblas_half> alpha,
                         Pointer<const rocblas_half> A, int lda,
                         Pointer<const rocblas_half> B, int ldb,
                         Pointer<const rocblas_half> beta,
                         Pointer<rocblas_half> C, int ldc);
llvm::Error RocblasSgemm(CurrentContext current, rocblas_handle handle,
                         rocblas_operation transa, rocblas_operation transb,
                         int m, int n, int k, Pointer<const float> alpha,
                         Pointer<const float> A, int lda,
                         Pointer<const float> B, int ldb,
                         Pointer<const float> beta, Pointer<float> C, int ldc);
llvm::Error RocblasDgemm(CurrentContext current, rocblas_handle handle,
                         rocblas_operation transa, rocblas_operation transb,
                         int m, int n, int k, Pointer<const double> alpha,
                         Pointer<const double> A, int lda,
                         Pointer<const double> B, int ldb,
                         Pointer<const double> beta, Pointer<double> C,
                         int ldc);
llvm::Error RocblasCgemm(CurrentContext current, rocblas_handle handle,
                         rocblas_operation transa, rocblas_operation transb,
                         int m, int n, int k,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> B, int ldb,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> C, int ldc);
llvm::Error RocblasZgemm(CurrentContext current, rocblas_handle handle,
                         rocblas_operation transa, rocblas_operation transb,
                         int m, int n, int k,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> B, int ldb,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> C, int ldc);
llvm::Error RocblasSsyrk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const float> alpha,
                         Pointer<const float> A, int lda,
                         Pointer<const float> beta, Pointer<float> C, int ldc);
llvm::Error RocblasDsyrk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const double> alpha,
                         Pointer<const double> A, int lda,
                         Pointer<const double> beta, Pointer<double> C,
                         int ldc);
llvm::Error RocblasCsyrk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> C, int ldc);
llvm::Error RocblasZsyrk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> C, int ldc);
llvm::Error RocblasCherk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const float> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const float> beta,
                         Pointer<rocblas_float_complex> C, int ldc);
llvm::Error RocblasZherk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const double> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const double> beta,
                         Pointer<rocblas_double_complex> C, int ldc);
llvm::Error RocblasSsyr2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const float> alpha,
                          Pointer<const float> A, int lda,
                          Pointer<const float> B, int ldb,
                          Pointer<const float> beta, Pointer<float> C, int ldc);
llvm::Error RocblasDsyr2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const double> alpha,
                          Pointer<const double> A, int lda,
                          Pointer<const double> B, int ldb,
                          Pointer<const double> beta, Pointer<double> C,
                          int ldc);
llvm::Error RocblasCsyr2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const rocblas_float_complex> alpha,
                          Pointer<const rocblas_float_complex> A, int lda,
                          Pointer<const rocblas_float_complex> B, int ldb,
                          Pointer<const rocblas_float_complex> beta,
                          Pointer<rocblas_float_complex> C, int ldc);
llvm::Error RocblasZsyr2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const rocblas_double_complex> alpha,
                          Pointer<const rocblas_double_complex> A, int lda,
                          Pointer<const rocblas_double_complex> B, int ldb,
                          Pointer<const rocblas_double_complex> beta,
                          Pointer<rocblas_double_complex> C, int ldc);
llvm::Error RocblasCher2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const rocblas_float_complex> alpha,
                          Pointer<const rocblas_float_complex> A, int lda,
                          Pointer<const rocblas_float_complex> B, int ldb,
                          Pointer<const float> beta,
                          Pointer<rocblas_float_complex> C, int ldc);
llvm::Error RocblasZher2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const rocblas_double_complex> alpha,
                          Pointer<const rocblas_double_complex> A, int lda,
                          Pointer<const rocblas_double_complex> B, int ldb,
                          Pointer<const double> beta,
                          Pointer<rocblas_double_complex> C, int ldc);
llvm::Error RocblasSsymm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const float> alpha, Pointer<const float> A,
                         int lda, Pointer<const float> B, int ldb,
                         Pointer<const float> beta, Pointer<float> C, int ldc);
llvm::Error RocblasDsymm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const double> alpha, Pointer<const double> A,
                         int lda, Pointer<const double> B, int ldb,
                         Pointer<const double> beta, Pointer<double> C,
                         int ldc);
llvm::Error RocblasCsymm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> B, int ldb,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> C, int ldc);
llvm::Error RocblasZsymm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> B, int ldb,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> C, int ldc);
llvm::Error RocblasChemm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> B, int ldb,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> C, int ldc);
llvm::Error RocblasZhemm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> B, int ldb,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> C, int ldc);
llvm::Error RocblasStrsm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const float> alpha,
                         Pointer<const float> A, int lda, Pointer<float> B,
                         int ldb);
llvm::Error RocblasDtrsm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const double> alpha,
                         Pointer<const double> A, int lda, Pointer<double> B,
                         int ldb);
llvm::Error RocblasCtrsm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> B, int ldb);
llvm::Error RocblasZtrsm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> B, int ldb);
llvm::Error RocblasStrmm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const float> alpha,
                         Pointer<const float> A, int lda, Pointer<float> B,
                         int ldb);
llvm::Error RocblasDtrmm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const double> alpha,
                         Pointer<const double> A, int lda, Pointer<double> B,
                         int ldb);
llvm::Error RocblasCtrmm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> B, int ldb);
llvm::Error RocblasZtrmm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> B, int ldb);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_ROCBLAS_WRAPPER_H_
