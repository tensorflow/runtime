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

// Thin wrapper around the rocBLAS API adding llvm::Error.
#include "tfrt/gpu/wrapper/rocblas_wrapper.h"

#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

template llvm::raw_ostream& internal::operator<<(
    llvm::raw_ostream&, const ErrorData<rocblas_status>&);

llvm::Expected<OwningBlasHandle> RocblasCreate(CurrentContext current) {
  CheckHipContext(current);
  rocblas_handle handle = nullptr;
  RETURN_IF_ERROR(rocblas_create_handle(&handle));
  return OwningBlasHandle(handle);
}

llvm::Error RocblasDestroy(rocblas_handle handle) {
  return TO_ERROR(rocblas_destroy_handle(handle));
}

llvm::Error RocblasSetStream(rocblas_handle handle, hipStream_t stream) {
  return TO_ERROR(rocblas_set_stream(handle, stream));
}

llvm::Expected<Stream> RocblasGetStream(rocblas_handle handle) {
  hipStream_t stream = nullptr;
  RETURN_IF_ERROR(rocblas_get_stream(handle, &stream));
  return Stream(stream);
}

llvm::Error RocblasSetPointerMode(rocblas_handle handle,
                                  rocblas_pointer_mode mode) {
  return TO_ERROR(rocblas_set_pointer_mode(handle, mode));
}

llvm::Expected<rocblas_pointer_mode> RocblasGetPointerMode(
    rocblas_handle handle) {
  rocblas_pointer_mode mode;
  RETURN_IF_ERROR(rocblas_get_pointer_mode(handle, &mode));
  return mode;
}

llvm::Error RocblasAxpyEx(
    CurrentContext current, rocblas_handle handle, int n,
    Pointer<const void> alpha, /* host or device pointer */
    rocblas_datatype alphaType, Pointer<const void> x, rocblas_datatype typeX,
    int strideX, Pointer<void> y, rocblas_datatype typeY, int strideY,
    rocblas_datatype executionType) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_axpy_ex(handle, n, ToRocm(alpha), alphaType,
                                  ToRocm(x), typeX, strideX, ToRocm(y), typeY,
                                  strideY, executionType));
}

llvm::Error RocblasGemmEx(
    CurrentContext current, rocblas_handle handle, rocblas_operation transA,
    rocblas_operation transB, int m, int n, int k, Pointer<const void> alpha,
    Pointer<const void> A, rocblas_datatype typeA, int heightA,
    Pointer<const void> B, rocblas_datatype typeB, int heightB,
    Pointer<const void> beta, Pointer<const void> C, rocblas_datatype typeC,
    int heightC, Pointer<void> D, rocblas_datatype typeD, int heightD,
    rocblas_datatype computeType, rocblas_gemm_algo algo) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_gemm_ex(handle, transA, transB, m, n, k, ToRocm(alpha), ToRocm(A),
                      typeA, heightA, ToRocm(B), typeB, heightB, ToRocm(beta),
                      ToRocm(C), typeC, heightC, ToRocm(D), typeD, heightD,
                      computeType, algo, /*solution_index=*/0, /*flags=*/0));
}

llvm::Error RocblasGemmStridedBatchedEx(
    CurrentContext current, rocblas_handle handle, rocblas_operation transA,
    rocblas_operation transB, int m, int n, int k, Pointer<const void> alpha,
    Pointer<const void> A, rocblas_datatype typeA, int heightA, int64_t strideA,
    Pointer<const void> B, rocblas_datatype typeB, int heightB, int64_t strideB,
    Pointer<const void> beta, Pointer<void> C, rocblas_datatype typeC,
    int heightC, int64_t strideC, Pointer<void> D, rocblas_datatype typeD,
    int heightD, int64_t strideD, int batchCount, rocblas_datatype computeType,
    rocblas_gemm_algo algo) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_gemm_strided_batched_ex(
      handle, transA, transB, m, n, k, ToRocm(alpha), ToRocm(A), typeA, heightA,
      strideA, ToRocm(B), typeB, heightB, strideB, ToRocm(beta), ToRocm(C),
      typeC, heightC, strideC, ToRocm(D), typeD, heightD, strideD, batchCount,
      computeType, algo, /*solution_index=*/0, /*flags=*/0));
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
