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
Expected<rocblas_diagonal> Parse<rocblas_diagonal>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_diagonal value);

template <>
Expected<rocblas_operation> Parse<rocblas_operation>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_operation value);

template <>
Expected<rocblas_gemm_algo> Parse<rocblas_gemm_algo>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_gemm_algo value);

template <>
Expected<rocblas_fill> Parse<rocblas_fill>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_fill value);

template <>
Expected<rocblas_side> Parse<rocblas_side>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_side value);

template <>
struct PlatformTypeTraits<BlasDataTypeTag, rocblas_datatype>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<BlasDiagTypeTag, rocblas_diagonal>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<BlasComputeTypeTag, rocblas_datatype>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<BlasOperationTag, rocblas_operation>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<BlasGemmAlgoTag, rocblas_gemm_algo>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<BlasFillModeTag, rocblas_fill>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<BlasSideModeTag, rocblas_side>
    : public RocmPlatformType {};

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
// TODO(hanbinyoon): Add TrsmBatched functions.

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_ROCBLAS_WRAPPER_H_
