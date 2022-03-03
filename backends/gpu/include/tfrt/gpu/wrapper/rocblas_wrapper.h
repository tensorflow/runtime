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

raw_ostream& Print(raw_ostream& os, rocblas_status status);
raw_ostream& Print(raw_ostream& os, rocblas_datatype value);
raw_ostream& Print(raw_ostream& os, rocblas_diagonal value);
raw_ostream& Print(raw_ostream& os, rocblas_operation value);
raw_ostream& Print(raw_ostream& os, rocblas_gemm_algo value);
raw_ostream& Print(raw_ostream& os, rocblas_fill value);
raw_ostream& Print(raw_ostream& os, rocblas_side value);

Expected<rocblas_datatype> Parse(llvm::StringRef name, rocblas_datatype);
Expected<rocblas_diagonal> Parse(llvm::StringRef name, rocblas_diagonal);
Expected<rocblas_operation> Parse(llvm::StringRef name, rocblas_operation);
Expected<rocblas_gemm_algo> Parse(llvm::StringRef name, rocblas_gemm_algo);
Expected<rocblas_fill> Parse(llvm::StringRef name, rocblas_fill);
Expected<rocblas_side> Parse(llvm::StringRef name, rocblas_side);

namespace internal {
template <>
struct EnumPlatform<BlasDataType, rocblas_datatype> : RocmPlatformType {};
// Also rocblas_datatype (type only differs for CUDA).
template <>
struct EnumPlatform<BlasComputeType, rocblas_datatype> : RocmPlatformType {};
template <>
struct EnumPlatform<BlasDiagType, rocblas_diagonal> : RocmPlatformType {};
template <>
struct EnumPlatform<BlasOperation, rocblas_operation> : RocmPlatformType {};
template <>
struct EnumPlatform<BlasGemmAlgo, rocblas_gemm_algo> : RocmPlatformType {};
template <>
struct EnumPlatform<BlasFillMode, rocblas_fill> : RocmPlatformType {};
template <>
struct EnumPlatform<BlasSideMode, rocblas_side> : RocmPlatformType {};

template <>
struct EnumStream<BlasDataType, Platform::ROCm>
    : EnumStreamPtrs<rocblas_datatype, Parse, Print> {};
template <>
struct EnumStream<BlasComputeType, Platform::ROCm>
    : EnumStreamPtrs<rocblas_datatype, Parse, Print> {};
template <>
struct EnumStream<BlasDiagType, Platform::ROCm>
    : EnumStreamPtrs<rocblas_diagonal, Parse, Print> {};
template <>
struct EnumStream<BlasOperation, Platform::ROCm>
    : EnumStreamPtrs<rocblas_operation, Parse, Print> {};
template <>
struct EnumStream<BlasGemmAlgo, Platform::ROCm>
    : EnumStreamPtrs<rocblas_gemm_algo, Parse, Print> {};
template <>
struct EnumStream<BlasFillMode, Platform::ROCm>
    : EnumStreamPtrs<rocblas_fill, Parse, Print> {};
template <>
struct EnumStream<BlasSideMode, Platform::ROCm>
    : EnumStreamPtrs<rocblas_side, Parse, Print> {};
}  // namespace internal

llvm::Expected<size_t> GetRocblasDataTypeSizeBytes(rocblas_datatype data_type);
mlir::TypeID GetRocblasDatatypeId(rocblas_datatype data_type);

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

llvm::Error RocblasTrsmBatched(CurrentContext current, cublasHandle_t handle,
                               rocblas_datatype dataType, rocblas_side sideMode,
                               rocblas_fill fillMode, rocblas_operation trans,
                               rocblas_diagonal diag, int m, int n,
                               Pointer<const void> alpha,
                               Pointer<const void*> A, int lda,
                               Pointer<void*> B, int ldb, int batchCount);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_ROCBLAS_WRAPPER_H_
