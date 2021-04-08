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
#include "tfrt/gpu/stream/cublas_wrapper.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "wrapper_detail.h"

#define RETURN_IF_ERROR(expr)                                 \
  while (cublasStatus_t _result = expr) {                     \
    return llvm::make_error<CublasErrorInfo>(                 \
        CublasErrorData{_result, #expr, CreateStackTrace()}); \
  }

#define TO_ERROR(expr)                                                   \
  [](cublasStatus_t _result) -> llvm::Error {                            \
    if (_result == CUBLAS_STATUS_SUCCESS) return llvm::Error::success(); \
    return llvm::make_error<CublasErrorInfo>(                            \
        CublasErrorData{_result, #expr, CreateStackTrace()});            \
  }(expr)

namespace tfrt {
namespace gpu {
namespace stream {

static llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return os << "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return os << "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return os << "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return os << "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return os << "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return os << "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return os << "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return os << "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return os << "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return os << "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return os << llvm::formatv("cublasStatus_t({0})",
                                 static_cast<int>(status));
  }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const CublasErrorData& data) {
  os << "'" << data.expr << "': " << data.result;
  if (data.stack_trace) os << ", stack trace:\n" << data.stack_trace;
  return os;
}

cublasStatus_t GetResult(const CublasErrorInfo& info) {
  return info.get<CublasErrorData>().result;
}

template <typename T>
static T* ToCuda(Pointer<T> ptr) {
  return ptr.raw(Platform::CUDA);
}

cublasOperation_t ToCublas(BlasOperation operation) {
  switch (operation) {
    case BlasOperation::kNone:
      return CUBLAS_OP_N;
    case BlasOperation::kTranspose:
      return CUBLAS_OP_T;
    case BlasOperation::kConjugateTranspose:
      return CUBLAS_OP_C;
  }
  llvm_unreachable(
      StrCat("Unrecognized BlasOperation value: ", operation).c_str());
}

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

llvm::Error CublasSnrm2(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const float> x, int incx,
                        Pointer<float> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSnrm2_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasDnrm2(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const double> x, int incx,
                        Pointer<double> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDnrm2_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasScnrm2(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuComplex> x, int incx,
                         Pointer<float> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasScnrm2_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasDznrm2(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuDoubleComplex> x, int incx,
                         Pointer<double> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDznrm2_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasSdot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<const float> x, int incx, Pointer<const float> y,
                       int incy, Pointer<float> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSdot_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                ToCuda(result)));
}

llvm::Error CublasDdot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<const double> x, int incx,
                       Pointer<const double> y, int incy,
                       Pointer<double> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDdot_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                ToCuda(result)));
}

llvm::Error CublasCdotu(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCdotu_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                 ToCuda(result)));
}

llvm::Error CublasCdotc(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCdotc_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                 ToCuda(result)));
}

llvm::Error CublasZdotu(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZdotu_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                 ToCuda(result)));
}

llvm::Error CublasZdotc(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZdotc_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                 ToCuda(result)));
}

llvm::Error CublasSscal(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const float> alpha, Pointer<float> x,
                        int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSscal_v2(handle, n, ToCuda(alpha), ToCuda(x), incx));
}

llvm::Error CublasDscal(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const double> alpha, Pointer<double> x,
                        int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDscal_v2(handle, n, ToCuda(alpha), ToCuda(x), incx));
}

llvm::Error CublasCscal(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuComplex> alpha, Pointer<cuComplex> x,
                        int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCscal_v2(handle, n, ToCuda(alpha), ToCuda(x), incx));
}

llvm::Error CublasCsscal(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const float> alpha, Pointer<cuComplex> x,
                         int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCsscal_v2(handle, n, ToCuda(alpha), ToCuda(x), incx));
}

llvm::Error CublasZscal(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<cuDoubleComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZscal_v2(handle, n, ToCuda(alpha), ToCuda(x), incx));
}

llvm::Error CublasZdscal(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const double> alpha,
                         Pointer<cuDoubleComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZdscal_v2(handle, n, ToCuda(alpha), ToCuda(x), incx));
}

llvm::Error CublasSaxpy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const float> alpha, Pointer<const float> x,
                        int incx, Pointer<float> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSaxpy_v2(handle, n, ToCuda(alpha), ToCuda(x), incx,
                                 ToCuda(y), incy));
}

llvm::Error CublasDaxpy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const double> alpha, Pointer<const double> x,
                        int incx, Pointer<double> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDaxpy_v2(handle, n, ToCuda(alpha), ToCuda(x), incx,
                                 ToCuda(y), incy));
}

llvm::Error CublasCaxpy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<cuComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCaxpy_v2(handle, n, ToCuda(alpha), ToCuda(x), incx,
                                 ToCuda(y), incy));
}

llvm::Error CublasZaxpy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<cuDoubleComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZaxpy_v2(handle, n, ToCuda(alpha), ToCuda(x), incx,
                                 ToCuda(y), incy));
}

llvm::Error CublasScopy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const float> x, int incx, Pointer<float> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasScopy_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy));
}

llvm::Error CublasDcopy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const double> x, int incx, Pointer<double> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDcopy_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy));
}

llvm::Error CublasCcopy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<cuComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCcopy_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy));
}

llvm::Error CublasZcopy(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<cuDoubleComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZcopy_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy));
}

llvm::Error CublasSswap(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<float> x, int incx, Pointer<float> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSswap_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy));
}

llvm::Error CublasDswap(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<double> x, int incx, Pointer<double> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDswap_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy));
}

llvm::Error CublasCswap(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<cuComplex> x, int incx, Pointer<cuComplex> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCswap_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy));
}

llvm::Error CublasZswap(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<cuDoubleComplex> x, int incx,
                        Pointer<cuDoubleComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZswap_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy));
}

llvm::Error CublasIsamax(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const float> x, int incx,
                         Pointer<int> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasIsamax_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasIdamax(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const double> x, int incx,
                         Pointer<int> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasIdamax_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasIcamax(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuComplex> x, int incx,
                         Pointer<int> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasIcamax_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasIzamax(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuDoubleComplex> x, int incx,
                         Pointer<int> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasIzamax_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasIsamin(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const float> x, int incx,
                         Pointer<int> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasIsamin_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasIdamin(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const double> x, int incx,
                         Pointer<int> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasIdamin_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasIcamin(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuComplex> x, int incx,
                         Pointer<int> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasIcamin_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasIzamin(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuDoubleComplex> x, int incx,
                         Pointer<int> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasIzamin_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasSasum(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const float> x, int incx,
                        Pointer<float> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSasum_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasDasum(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<const double> x, int incx,
                        Pointer<double> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDasum_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasScasum(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuComplex> x, int incx,
                         Pointer<float> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasScasum_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasDzasum(CurrentContext current, cublasHandle_t handle, int n,
                         Pointer<const cuDoubleComplex> x, int incx,
                         Pointer<double> result) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDzasum_v2(handle, n, ToCuda(x), incx, ToCuda(result)));
}

llvm::Error CublasSrot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<float> x, int incx, Pointer<float> y, int incy,
                       Pointer<const float> c, Pointer<const float> s) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSrot_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                ToCuda(c), ToCuda(s)));
}

llvm::Error CublasDrot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<double> x, int incx, Pointer<double> y, int incy,
                       Pointer<const double> c, Pointer<const double> s) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDrot_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                ToCuda(c), ToCuda(s)));
}

llvm::Error CublasCrot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<cuComplex> x, int incx, Pointer<cuComplex> y,
                       int incy, Pointer<const float> c,
                       Pointer<const cuComplex> s) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCrot_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                ToCuda(c), ToCuda(s)));
}

llvm::Error CublasCsrot(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<cuComplex> x, int incx, Pointer<cuComplex> y,
                        int incy, Pointer<const float> c,
                        Pointer<const float> s) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCsrot_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                 ToCuda(c), ToCuda(s)));
}

llvm::Error CublasZrot(CurrentContext current, cublasHandle_t handle, int n,
                       Pointer<cuDoubleComplex> x, int incx,
                       Pointer<cuDoubleComplex> y, int incy,
                       Pointer<const double> c,
                       Pointer<const cuDoubleComplex> s) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZrot_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                ToCuda(c), ToCuda(s)));
}

llvm::Error CublasZdrot(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<cuDoubleComplex> x, int incx,
                        Pointer<cuDoubleComplex> y, int incy,
                        Pointer<const double> c, Pointer<const double> s) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZdrot_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                 ToCuda(c), ToCuda(s)));
}

llvm::Error CublasSrotg(CurrentContext current, cublasHandle_t handle,
                        Pointer<float> a, Pointer<float> b, Pointer<float> c,
                        Pointer<float> s) {
  CheckCudaContext(current);
  return TO_ERROR(
      cublasSrotg_v2(handle, ToCuda(a), ToCuda(b), ToCuda(c), ToCuda(s)));
}

llvm::Error CublasDrotg(CurrentContext current, cublasHandle_t handle,
                        Pointer<double> a, Pointer<double> b, Pointer<double> c,
                        Pointer<double> s) {
  CheckCudaContext(current);
  return TO_ERROR(
      cublasDrotg_v2(handle, ToCuda(a), ToCuda(b), ToCuda(c), ToCuda(s)));
}

llvm::Error CublasCrotg(CurrentContext current, cublasHandle_t handle,
                        Pointer<cuComplex> a, Pointer<cuComplex> b,
                        Pointer<float> c, Pointer<cuComplex> s) {
  CheckCudaContext(current);
  return TO_ERROR(
      cublasCrotg_v2(handle, ToCuda(a), ToCuda(b), ToCuda(c), ToCuda(s)));
}

llvm::Error CublasZrotg(CurrentContext current, cublasHandle_t handle,
                        Pointer<cuDoubleComplex> a, Pointer<cuDoubleComplex> b,
                        Pointer<double> c, Pointer<cuDoubleComplex> s) {
  CheckCudaContext(current);
  return TO_ERROR(
      cublasZrotg_v2(handle, ToCuda(a), ToCuda(b), ToCuda(c), ToCuda(s)));
}

llvm::Error CublasSrotm(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<float> x, int incx, Pointer<float> y, int incy,
                        Pointer<const float> param) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSrotm_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                 ToCuda(param)));
}

llvm::Error CublasDrotm(CurrentContext current, cublasHandle_t handle, int n,
                        Pointer<double> x, int incx, Pointer<double> y,
                        int incy, Pointer<const double> param) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDrotm_v2(handle, n, ToCuda(x), incx, ToCuda(y), incy,
                                 ToCuda(param)));
}

llvm::Error CublasSrotmg(CurrentContext current, cublasHandle_t handle,
                         Pointer<float> d1, Pointer<float> d2,
                         Pointer<float> x1, Pointer<const float> y1,
                         Pointer<float> param) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSrotmg_v2(handle, ToCuda(d1), ToCuda(d2), ToCuda(x1),
                                  ToCuda(y1), ToCuda(param)));
}

llvm::Error CublasDrotmg(CurrentContext current, cublasHandle_t handle,
                         Pointer<double> d1, Pointer<double> d2,
                         Pointer<double> x1, Pointer<const double> y1,
                         Pointer<double> param) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDrotmg_v2(handle, ToCuda(d1), ToCuda(d2), ToCuda(x1),
                                  ToCuda(y1), ToCuda(param)));
}

llvm::Error CublasSgemv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n,
                        Pointer<const float> alpha, Pointer<const float> A,
                        int lda, Pointer<const float> x, int incx,
                        Pointer<const float> beta, Pointer<float> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSgemv_v2(handle, trans, m, n, ToCuda(alpha), ToCuda(A),
                                 lda, ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasDgemv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n,
                        Pointer<const double> alpha, Pointer<const double> A,
                        int lda, Pointer<const double> x, int incx,
                        Pointer<const double> beta, Pointer<double> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDgemv_v2(handle, trans, m, n, ToCuda(alpha), ToCuda(A),
                                 lda, ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasCgemv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCgemv_v2(handle, trans, m, n, ToCuda(alpha), ToCuda(A),
                                 lda, ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasZgemv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZgemv_v2(handle, trans, m, n, ToCuda(alpha), ToCuda(A),
                                 lda, ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasSgbmv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n, int kl, int ku,
                        Pointer<const float> alpha, Pointer<const float> A,
                        int lda, Pointer<const float> x, int incx,
                        Pointer<const float> beta, Pointer<float> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSgbmv_v2(handle, trans, m, n, kl, ku, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(x), incx, ToCuda(beta),
                                 ToCuda(y), incy));
}

llvm::Error CublasDgbmv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n, int kl, int ku,
                        Pointer<const double> alpha, Pointer<const double> A,
                        int lda, Pointer<const double> x, int incx,
                        Pointer<const double> beta, Pointer<double> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDgbmv_v2(handle, trans, m, n, kl, ku, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(x), incx, ToCuda(beta),
                                 ToCuda(y), incy));
}

llvm::Error CublasCgbmv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n, int kl, int ku,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCgbmv_v2(handle, trans, m, n, kl, ku, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(x), incx, ToCuda(beta),
                                 ToCuda(y), incy));
}

llvm::Error CublasZgbmv(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t trans, int m, int n, int kl, int ku,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZgbmv_v2(handle, trans, m, n, kl, ku, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(x), incx, ToCuda(beta),
                                 ToCuda(y), incy));
}

llvm::Error CublasStrmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const float> A,
                        int lda, Pointer<float> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasStrmv_v2(handle, uplo, trans, diag, n, ToCuda(A), lda,
                                 ToCuda(x), incx));
}

llvm::Error CublasDtrmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const double> A,
                        int lda, Pointer<double> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDtrmv_v2(handle, uplo, trans, diag, n, ToCuda(A), lda,
                                 ToCuda(x), incx));
}

llvm::Error CublasCtrmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<cuComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCtrmv_v2(handle, uplo, trans, diag, n, ToCuda(A), lda,
                                 ToCuda(x), incx));
}

llvm::Error CublasZtrmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<cuDoubleComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZtrmv_v2(handle, uplo, trans, diag, n, ToCuda(A), lda,
                                 ToCuda(x), incx));
}

llvm::Error CublasStbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const float> A, int lda, Pointer<float> x,
                        int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasStbmv_v2(handle, uplo, trans, diag, n, k, ToCuda(A),
                                 lda, ToCuda(x), incx));
}

llvm::Error CublasDtbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const double> A, int lda, Pointer<double> x,
                        int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDtbmv_v2(handle, uplo, trans, diag, n, k, ToCuda(A),
                                 lda, ToCuda(x), incx));
}

llvm::Error CublasCtbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<cuComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCtbmv_v2(handle, uplo, trans, diag, n, k, ToCuda(A),
                                 lda, ToCuda(x), incx));
}

llvm::Error CublasZtbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<cuDoubleComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZtbmv_v2(handle, uplo, trans, diag, n, k, ToCuda(A),
                                 lda, ToCuda(x), incx));
}

llvm::Error CublasStpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const float> AP,
                        Pointer<float> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasStpmv_v2(handle, uplo, trans, diag, n, ToCuda(AP),
                                 ToCuda(x), incx));
}

llvm::Error CublasDtpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const double> AP,
                        Pointer<double> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDtpmv_v2(handle, uplo, trans, diag, n, ToCuda(AP),
                                 ToCuda(x), incx));
}

llvm::Error CublasCtpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuComplex> AP, Pointer<cuComplex> x,
                        int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCtpmv_v2(handle, uplo, trans, diag, n, ToCuda(AP),
                                 ToCuda(x), incx));
}

llvm::Error CublasZtpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuDoubleComplex> AP,
                        Pointer<cuDoubleComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZtpmv_v2(handle, uplo, trans, diag, n, ToCuda(AP),
                                 ToCuda(x), incx));
}

llvm::Error CublasStrsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const float> A,
                        int lda, Pointer<float> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasStrsv_v2(handle, uplo, trans, diag, n, ToCuda(A), lda,
                                 ToCuda(x), incx));
}

llvm::Error CublasDtrsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const double> A,
                        int lda, Pointer<double> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDtrsv_v2(handle, uplo, trans, diag, n, ToCuda(A), lda,
                                 ToCuda(x), incx));
}

llvm::Error CublasCtrsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<cuComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCtrsv_v2(handle, uplo, trans, diag, n, ToCuda(A), lda,
                                 ToCuda(x), incx));
}

llvm::Error CublasZtrsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<cuDoubleComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZtrsv_v2(handle, uplo, trans, diag, n, ToCuda(A), lda,
                                 ToCuda(x), incx));
}

llvm::Error CublasStpsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const float> AP,
                        Pointer<float> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasStpsv_v2(handle, uplo, trans, diag, n, ToCuda(AP),
                                 ToCuda(x), incx));
}

llvm::Error CublasDtpsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, Pointer<const double> AP,
                        Pointer<double> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDtpsv_v2(handle, uplo, trans, diag, n, ToCuda(AP),
                                 ToCuda(x), incx));
}

llvm::Error CublasCtpsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuComplex> AP, Pointer<cuComplex> x,
                        int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCtpsv_v2(handle, uplo, trans, diag, n, ToCuda(AP),
                                 ToCuda(x), incx));
}

llvm::Error CublasZtpsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n,
                        Pointer<const cuDoubleComplex> AP,
                        Pointer<cuDoubleComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZtpsv_v2(handle, uplo, trans, diag, n, ToCuda(AP),
                                 ToCuda(x), incx));
}

llvm::Error CublasStbsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const float> A, int lda, Pointer<float> x,
                        int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasStbsv_v2(handle, uplo, trans, diag, n, k, ToCuda(A),
                                 lda, ToCuda(x), incx));
}

llvm::Error CublasDtbsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const double> A, int lda, Pointer<double> x,
                        int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDtbsv_v2(handle, uplo, trans, diag, n, k, ToCuda(A),
                                 lda, ToCuda(x), incx));
}

llvm::Error CublasCtbsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<cuComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCtbsv_v2(handle, uplo, trans, diag, n, k, ToCuda(A),
                                 lda, ToCuda(x), incx));
}

llvm::Error CublasZtbsv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans,
                        cublasDiagType_t diag, int n, int k,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<cuDoubleComplex> x, int incx) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZtbsv_v2(handle, uplo, trans, diag, n, k, ToCuda(A),
                                 lda, ToCuda(x), incx));
}

llvm::Error CublasSsymv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const float> alpha, Pointer<const float> A,
                        int lda, Pointer<const float> x, int incx,
                        Pointer<const float> beta, Pointer<float> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSsymv_v2(handle, uplo, n, ToCuda(alpha), ToCuda(A), lda,
                                 ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasDsymv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const double> alpha, Pointer<const double> A,
                        int lda, Pointer<const double> x, int incx,
                        Pointer<const double> beta, Pointer<double> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDsymv_v2(handle, uplo, n, ToCuda(alpha), ToCuda(A), lda,
                                 ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasCsymv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCsymv_v2(handle, uplo, n, ToCuda(alpha), ToCuda(A), lda,
                                 ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasZsymv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZsymv_v2(handle, uplo, n, ToCuda(alpha), ToCuda(A), lda,
                                 ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasChemv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasChemv_v2(handle, uplo, n, ToCuda(alpha), ToCuda(A), lda,
                                 ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasZhemv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZhemv_v2(handle, uplo, n, ToCuda(alpha), ToCuda(A), lda,
                                 ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasSsbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n, int k,
                        Pointer<const float> alpha, Pointer<const float> A,
                        int lda, Pointer<const float> x, int incx,
                        Pointer<const float> beta, Pointer<float> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSsbmv_v2(handle, uplo, n, k, ToCuda(alpha), ToCuda(A),
                                 lda, ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasDsbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n, int k,
                        Pointer<const double> alpha, Pointer<const double> A,
                        int lda, Pointer<const double> x, int incx,
                        Pointer<const double> beta, Pointer<double> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDsbmv_v2(handle, uplo, n, k, ToCuda(alpha), ToCuda(A),
                                 lda, ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasChbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n, int k,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasChbmv_v2(handle, uplo, n, k, ToCuda(alpha), ToCuda(A),
                                 lda, ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasZhbmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n, int k,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZhbmv_v2(handle, uplo, n, k, ToCuda(alpha), ToCuda(A),
                                 lda, ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasSspmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const float> alpha, Pointer<const float> AP,
                        Pointer<const float> x, int incx,
                        Pointer<const float> beta, Pointer<float> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSspmv_v2(handle, uplo, n, ToCuda(alpha), ToCuda(AP),
                                 ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasDspmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const double> alpha, Pointer<const double> AP,
                        Pointer<const double> x, int incx,
                        Pointer<const double> beta, Pointer<double> y,
                        int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDspmv_v2(handle, uplo, n, ToCuda(alpha), ToCuda(AP),
                                 ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasChpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> AP, Pointer<const cuComplex> x,
                        int incx, Pointer<const cuComplex> beta,
                        Pointer<cuComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasChpmv_v2(handle, uplo, n, ToCuda(alpha), ToCuda(AP),
                                 ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasZhpmv(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> AP,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> y, int incy) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZhpmv_v2(handle, uplo, n, ToCuda(alpha), ToCuda(AP),
                                 ToCuda(x), incx, ToCuda(beta), ToCuda(y),
                                 incy));
}

llvm::Error CublasSger(CurrentContext current, cublasHandle_t handle, int m,
                       int n, Pointer<const float> alpha,
                       Pointer<const float> x, int incx, Pointer<const float> y,
                       int incy, Pointer<float> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSger_v2(handle, m, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasDger(CurrentContext current, cublasHandle_t handle, int m,
                       int n, Pointer<const double> alpha,
                       Pointer<const double> x, int incx,
                       Pointer<const double> y, int incy, Pointer<double> A,
                       int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDger_v2(handle, m, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasCgeru(CurrentContext current, cublasHandle_t handle, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCgeru_v2(handle, m, n, ToCuda(alpha), ToCuda(x), incx,
                                 ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasCgerc(CurrentContext current, cublasHandle_t handle, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCgerc_v2(handle, m, n, ToCuda(alpha), ToCuda(x), incx,
                                 ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasZgeru(CurrentContext current, cublasHandle_t handle, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZgeru_v2(handle, m, n, ToCuda(alpha), ToCuda(x), incx,
                                 ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasZgerc(CurrentContext current, cublasHandle_t handle, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZgerc_v2(handle, m, n, ToCuda(alpha), ToCuda(x), incx,
                                 ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasSsyr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n, Pointer<const float> alpha,
                       Pointer<const float> x, int incx, Pointer<float> A,
                       int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSsyr_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(A), lda));
}

llvm::Error CublasDsyr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const double> alpha, Pointer<const double> x,
                       int incx, Pointer<double> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDsyr_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(A), lda));
}

llvm::Error CublasCsyr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const cuComplex> alpha,
                       Pointer<const cuComplex> x, int incx,
                       Pointer<cuComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCsyr_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(A), lda));
}

llvm::Error CublasZsyr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const cuDoubleComplex> alpha,
                       Pointer<const cuDoubleComplex> x, int incx,
                       Pointer<cuDoubleComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZsyr_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(A), lda));
}

llvm::Error CublasCher(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n, Pointer<const float> alpha,
                       Pointer<const cuComplex> x, int incx,
                       Pointer<cuComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCher_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(A), lda));
}

llvm::Error CublasZher(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const double> alpha,
                       Pointer<const cuDoubleComplex> x, int incx,
                       Pointer<cuDoubleComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZher_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(A), lda));
}

llvm::Error CublasSspr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n, Pointer<const float> alpha,
                       Pointer<const float> x, int incx, Pointer<float> AP) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSspr_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(AP)));
}

llvm::Error CublasDspr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const double> alpha, Pointer<const double> x,
                       int incx, Pointer<double> AP) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDspr_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(AP)));
}

llvm::Error CublasChpr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n, Pointer<const float> alpha,
                       Pointer<const cuComplex> x, int incx,
                       Pointer<cuComplex> AP) {
  CheckCudaContext(current);
  return TO_ERROR(cublasChpr_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(AP)));
}

llvm::Error CublasZhpr(CurrentContext current, cublasHandle_t handle,
                       cublasFillMode_t uplo, int n,
                       Pointer<const double> alpha,
                       Pointer<const cuDoubleComplex> x, int incx,
                       Pointer<cuDoubleComplex> AP) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZhpr_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x), incx,
                                ToCuda(AP)));
}

llvm::Error CublasSsyr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const float> alpha, Pointer<const float> x,
                        int incx, Pointer<const float> y, int incy,
                        Pointer<float> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSsyr2_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x),
                                 incx, ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasDsyr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const double> alpha, Pointer<const double> x,
                        int incx, Pointer<const double> y, int incy,
                        Pointer<double> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDsyr2_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x),
                                 incx, ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasCsyr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCsyr2_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x),
                                 incx, ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasZsyr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZsyr2_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x),
                                 incx, ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasCher2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCher2_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x),
                                 incx, ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasZher2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> A, int lda) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZher2_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x),
                                 incx, ToCuda(y), incy, ToCuda(A), lda));
}

llvm::Error CublasSspr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const float> alpha, Pointer<const float> x,
                        int incx, Pointer<const float> y, int incy,
                        Pointer<float> AP) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSspr2_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x),
                                 incx, ToCuda(y), incy, ToCuda(AP)));
}

llvm::Error CublasDspr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const double> alpha, Pointer<const double> x,
                        int incx, Pointer<const double> y, int incy,
                        Pointer<double> AP) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDspr2_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x),
                                 incx, ToCuda(y), incy, ToCuda(AP)));
}

llvm::Error CublasChpr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> x, int incx,
                        Pointer<const cuComplex> y, int incy,
                        Pointer<cuComplex> AP) {
  CheckCudaContext(current);
  return TO_ERROR(cublasChpr2_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x),
                                 incx, ToCuda(y), incy, ToCuda(AP)));
}

llvm::Error CublasZhpr2(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, int n,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> x, int incx,
                        Pointer<const cuDoubleComplex> y, int incy,
                        Pointer<cuDoubleComplex> AP) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZhpr2_v2(handle, uplo, n, ToCuda(alpha), ToCuda(x),
                                 incx, ToCuda(y), incy, ToCuda(AP)));
}

llvm::Error CublasHgemm(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k, Pointer<const __half> alpha,
                        Pointer<const __half> A, int lda,
                        Pointer<const __half> B, int ldb,
                        Pointer<const __half> beta, Pointer<__half> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasHgemm(handle, transa, transb, m, n, k, ToCuda(alpha),
                              ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                              ToCuda(C), ldc));
}

llvm::Error CublasSgemm(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k, Pointer<const float> alpha,
                        Pointer<const float> A, int lda, Pointer<const float> B,
                        int ldb, Pointer<const float> beta, Pointer<float> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSgemm_v2(handle, transa, transb, m, n, k, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                 ToCuda(C), ldc));
}

llvm::Error CublasDgemm(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k, Pointer<const double> alpha,
                        Pointer<const double> A, int lda,
                        Pointer<const double> B, int ldb,
                        Pointer<const double> beta, Pointer<double> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDgemm_v2(handle, transa, transb, m, n, k, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                 ToCuda(C), ldc));
}

llvm::Error CublasCgemm(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> B, int ldb,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCgemm_v2(handle, transa, transb, m, n, k, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                 ToCuda(C), ldc));
}

llvm::Error CublasZgemm(CurrentContext current, cublasHandle_t handle,
                        cublasOperation_t transa, cublasOperation_t transb,
                        int m, int n, int k,
                        Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> B, int ldb,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> C, int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZgemm_v2(handle, transa, transb, m, n, k, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                 ToCuda(C), ldc));
}

llvm::Error CublasSsyrk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const float> alpha,
                        Pointer<const float> A, int lda,
                        Pointer<const float> beta, Pointer<float> C, int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSsyrk_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(beta), ToCuda(C), ldc));
}

llvm::Error CublasDsyrk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const double> alpha,
                        Pointer<const double> A, int lda,
                        Pointer<const double> beta, Pointer<double> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDsyrk_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(beta), ToCuda(C), ldc));
}

llvm::Error CublasCsyrk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCsyrk_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(beta), ToCuda(C), ldc));
}

llvm::Error CublasZsyrk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> C, int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZsyrk_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(beta), ToCuda(C), ldc));
}

llvm::Error CublasCherk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const float> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const float> beta, Pointer<cuComplex> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCherk_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(beta), ToCuda(C), ldc));
}

llvm::Error CublasZherk(CurrentContext current, cublasHandle_t handle,
                        cublasFillMode_t uplo, cublasOperation_t trans, int n,
                        int k, Pointer<const double> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const double> beta, Pointer<cuDoubleComplex> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZherk_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(beta), ToCuda(C), ldc));
}

llvm::Error CublasSsyr2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const float> alpha,
                         Pointer<const float> A, int lda,
                         Pointer<const float> B, int ldb,
                         Pointer<const float> beta, Pointer<float> C, int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSsyr2k_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                  ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                  ToCuda(C), ldc));
}

llvm::Error CublasDsyr2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const double> alpha,
                         Pointer<const double> A, int lda,
                         Pointer<const double> B, int ldb,
                         Pointer<const double> beta, Pointer<double> C,
                         int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDsyr2k_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                  ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                  ToCuda(C), ldc));
}

llvm::Error CublasCsyr2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const cuComplex> alpha,
                         Pointer<const cuComplex> A, int lda,
                         Pointer<const cuComplex> B, int ldb,
                         Pointer<const cuComplex> beta, Pointer<cuComplex> C,
                         int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCsyr2k_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                  ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                  ToCuda(C), ldc));
}

llvm::Error CublasZsyr2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const cuDoubleComplex> alpha,
                         Pointer<const cuDoubleComplex> A, int lda,
                         Pointer<const cuDoubleComplex> B, int ldb,
                         Pointer<const cuDoubleComplex> beta,
                         Pointer<cuDoubleComplex> C, int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZsyr2k_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                  ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                  ToCuda(C), ldc));
}

llvm::Error CublasCher2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const cuComplex> alpha,
                         Pointer<const cuComplex> A, int lda,
                         Pointer<const cuComplex> B, int ldb,
                         Pointer<const float> beta, Pointer<cuComplex> C,
                         int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCher2k_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                  ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                  ToCuda(C), ldc));
}

llvm::Error CublasZher2k(CurrentContext current, cublasHandle_t handle,
                         cublasFillMode_t uplo, cublasOperation_t trans, int n,
                         int k, Pointer<const cuDoubleComplex> alpha,
                         Pointer<const cuDoubleComplex> A, int lda,
                         Pointer<const cuDoubleComplex> B, int ldb,
                         Pointer<const double> beta, Pointer<cuDoubleComplex> C,
                         int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZher2k_v2(handle, uplo, trans, n, k, ToCuda(alpha),
                                  ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                  ToCuda(C), ldc));
}

llvm::Error CublasSsymm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const float> alpha,
                        Pointer<const float> A, int lda, Pointer<const float> B,
                        int ldb, Pointer<const float> beta, Pointer<float> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasSsymm_v2(handle, side, uplo, m, n, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                 ToCuda(C), ldc));
}

llvm::Error CublasDsymm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const double> alpha,
                        Pointer<const double> A, int lda,
                        Pointer<const double> B, int ldb,
                        Pointer<const double> beta, Pointer<double> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDsymm_v2(handle, side, uplo, m, n, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                 ToCuda(C), ldc));
}

llvm::Error CublasCsymm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> B, int ldb,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCsymm_v2(handle, side, uplo, m, n, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                 ToCuda(C), ldc));
}

llvm::Error CublasZsymm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> B, int ldb,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> C, int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZsymm_v2(handle, side, uplo, m, n, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                 ToCuda(C), ldc));
}

llvm::Error CublasChemm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> B, int ldb,
                        Pointer<const cuComplex> beta, Pointer<cuComplex> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasChemm_v2(handle, side, uplo, m, n, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                 ToCuda(C), ldc));
}

llvm::Error CublasZhemm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> B, int ldb,
                        Pointer<const cuDoubleComplex> beta,
                        Pointer<cuDoubleComplex> C, int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZhemm_v2(handle, side, uplo, m, n, ToCuda(alpha),
                                 ToCuda(A), lda, ToCuda(B), ldb, ToCuda(beta),
                                 ToCuda(C), ldc));
}

llvm::Error CublasStrsm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const float> alpha,
                        Pointer<const float> A, int lda, Pointer<float> B,
                        int ldb) {
  CheckCudaContext(current);
  return TO_ERROR(cublasStrsm_v2(handle, side, uplo, trans, diag, m, n,
                                 ToCuda(alpha), ToCuda(A), lda, ToCuda(B),
                                 ldb));
}

llvm::Error CublasDtrsm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const double> alpha,
                        Pointer<const double> A, int lda, Pointer<double> B,
                        int ldb) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n,
                                 ToCuda(alpha), ToCuda(A), lda, ToCuda(B),
                                 ldb));
}

llvm::Error CublasCtrsm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<cuComplex> B, int ldb) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCtrsm_v2(handle, side, uplo, trans, diag, m, n,
                                 ToCuda(alpha), ToCuda(A), lda, ToCuda(B),
                                 ldb));
}

llvm::Error CublasZtrsm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<cuDoubleComplex> B, int ldb) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZtrsm_v2(handle, side, uplo, trans, diag, m, n,
                                 ToCuda(alpha), ToCuda(A), lda, ToCuda(B),
                                 ldb));
}

llvm::Error CublasStrmm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const float> alpha,
                        Pointer<const float> A, int lda, Pointer<const float> B,
                        int ldb, Pointer<float> C, int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasStrmm_v2(handle, side, uplo, trans, diag, m, n,
                                 ToCuda(alpha), ToCuda(A), lda, ToCuda(B), ldb,
                                 ToCuda(C), ldc));
}

llvm::Error CublasDtrmm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const double> alpha,
                        Pointer<const double> A, int lda,
                        Pointer<const double> B, int ldb, Pointer<double> C,
                        int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n,
                                 ToCuda(alpha), ToCuda(A), lda, ToCuda(B), ldb,
                                 ToCuda(C), ldc));
}

llvm::Error CublasCtrmm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const cuComplex> alpha,
                        Pointer<const cuComplex> A, int lda,
                        Pointer<const cuComplex> B, int ldb,
                        Pointer<cuComplex> C, int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasCtrmm_v2(handle, side, uplo, trans, diag, m, n,
                                 ToCuda(alpha), ToCuda(A), lda, ToCuda(B), ldb,
                                 ToCuda(C), ldc));
}

llvm::Error CublasZtrmm(CurrentContext current, cublasHandle_t handle,
                        cublasSideMode_t side, cublasFillMode_t uplo,
                        cublasOperation_t trans, cublasDiagType_t diag, int m,
                        int n, Pointer<const cuDoubleComplex> alpha,
                        Pointer<const cuDoubleComplex> A, int lda,
                        Pointer<const cuDoubleComplex> B, int ldb,
                        Pointer<cuDoubleComplex> C, int ldc) {
  CheckCudaContext(current);
  return TO_ERROR(cublasZtrmm_v2(handle, side, uplo, trans, diag, m, n,
                                 ToCuda(alpha), ToCuda(A), lda, ToCuda(B), ldb,
                                 ToCuda(C), ldc));
}

// Following function is defined in cublas_stub.cc
// backward compartible wrapper for the cublasGemmEx
// to accomodate for the API change between cuBLAS v10 and v11.
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx_v10(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, /* host or device pointer */
    const void* A, cudaDataType Atype, int lda, const void* B,
    cudaDataType Btype, int ldb, const void* beta, /* host or device pointer */
    void* C, cudaDataType Ctype, int ldc, cudaDataType computeType,
    cublasGemmAlgo_t algo);

llvm::Error CublasGemmEx(CurrentContext current, cublasHandle_t handle,
                         cublasOperation_t transa, cublasOperation_t transb,
                         int m, int n, int k, Pointer<const void> alpha,
                         Pointer<const void> A, cudaDataType Atype, int lda,
                         Pointer<const void> B, cudaDataType Btype, int ldb,
                         Pointer<const void> beta, Pointer<void> C,
                         cudaDataType Ctype, int ldc, cudaDataType computeType,
                         cublasGemmAlgo_t algo) {
  CheckCudaContext(current);
  return TO_ERROR(cublasGemmEx_v10(handle, transa, transb, m, n, k,
                                   ToCuda(alpha), ToCuda(A), Atype, lda,
                                   ToCuda(B), Btype, ldb, ToCuda(beta),
                                   ToCuda(C), Ctype, ldc, computeType, algo));
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx_v10(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, /* host or device pointer */
    const void* Aarray[], cudaDataType Atype, int lda, const void* Barray[],
    cudaDataType Btype, int ldb, const void* beta, /* host or device pointer */
    void* Carray[], cudaDataType Ctype, int ldc, int batchCount,
    cudaDataType computeType, cublasGemmAlgo_t algo);

llvm::Error CublasGemmBatchedEx(
    CurrentContext current, cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, Pointer<const void> alpha,
    Pointer<const void*> Aarray, cudaDataType Atype, int lda,
    Pointer<const void*> Barray, cudaDataType Btype, int ldb,
    Pointer<const void> beta, Pointer<void*> Carray, cudaDataType Ctype,
    int ldc, int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo) {
  CheckCudaContext(current);
  return TO_ERROR(cublasGemmBatchedEx_v10(
      handle, transa, transb, m, n, k, ToCuda(alpha), ToCuda(Aarray), Atype,
      lda, ToCuda(Barray), Btype, ldb, ToCuda(beta), ToCuda(Carray), Ctype, ldc,
      batchCount, computeType, algo));
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx_v10(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, /* host or device pointer */
    const void* A, cudaDataType Atype, int lda, int64_t strideA, const void* B,
    cudaDataType Btype, int ldb, int64_t strideB,
    const void* beta, /* host or device pointer */
    void* C, cudaDataType Ctype, int ldc, int64_t strideC, int batchCount,
    cudaDataType computeType, cublasGemmAlgo_t algo);

llvm::Error CublasGemmStridedBatchedEx(
    CurrentContext current, cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, Pointer<const void> alpha,
    Pointer<const void> A, cudaDataType Atype, int lda, int64_t strideA,
    Pointer<const void> B, cudaDataType Btype, int ldb, int64_t strideB,
    Pointer<const void> beta, Pointer<void> C, cudaDataType Ctype, int ldc,
    int64_t strideC, int batchCount, cudaDataType computeType,
    cublasGemmAlgo_t algo) {
  CheckCudaContext(current);
  return TO_ERROR(cublasGemmStridedBatchedEx_v10(
      handle, transa, transb, m, n, k, ToCuda(alpha), ToCuda(A), Atype, lda,
      strideA, ToCuda(B), Btype, ldb, strideB, ToCuda(beta), ToCuda(C), Ctype,
      ldc, strideC, batchCount, computeType, algo));
}

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt
