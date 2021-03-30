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

//===- rocblas_wrapper.cc ---------------------------------------*- C++ -*-===//
//
// Thin wrapper around the rocBLAS API adding llvm::Error.
//
//===----------------------------------------------------------------------===//
#include "tfrt/gpu/stream/rocblas_wrapper.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "wrapper_detail.h"

#define RETURN_IF_ERROR(expr)                                  \
  while (rocblas_status _result = expr) {                      \
    return llvm::make_error<RocblasErrorInfo>(                 \
        RocblasErrorData{_result, #expr, CreateStackTrace()}); \
  }

#define TO_ERROR(expr)                                                    \
  [](rocblas_status _result) -> llvm::Error {                             \
    if (_result == rocblas_status_success) return llvm::Error::success(); \
    return llvm::make_error<RocblasErrorInfo>(                            \
        RocblasErrorData{_result, #expr, CreateStackTrace()});            \
  }(expr)

namespace tfrt {
namespace gpu {
namespace stream {

static llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     rocblas_status status) {
  switch (status) {
    case rocblas_status_success:
      return os << "rocblas_status_success";
    case rocblas_status_invalid_handle:
      return os << "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
      return os << "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return os << "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return os << "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
      return os << "rocblas_status_memory_error";
    case rocblas_status_internal_error:
      return os << "rocblas_status_internal_error";
    case rocblas_status_perf_degraded:
      return os << "rocblas_status_perf_degraded";
    case rocblas_status_size_query_mismatch:
      return os << "rocblas_status_size_query_mismatch";
    case rocblas_status_size_increased:
      return os << "rocblas_status_size_increased";
    case rocblas_status_size_unchanged:
      return os << "rocblas_status_size_unchanged";
    case rocblas_status_invalid_value:
      return os << "rocblas_status_invalid_value";
    case rocblas_status_continue:
      return os << "rocblas_status_continue";
    default:
      return os << llvm::formatv("rocblas_status({0})",
                                 static_cast<int>(status));
  }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const RocblasErrorData& data) {
  os << "'" << data.expr << "': " << data.result;
  if (data.stack_trace) os << ", stack trace:\n" << data.stack_trace;
  return os;
}

rocblas_status GetResult(const RocblasErrorInfo& info) {
  return info.get<RocblasErrorData>().result;
}

template <typename T>
static T* ToRocm(Pointer<T> ptr) {
  return ptr.raw(Platform::ROCm);
}

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

llvm::Error RocblasSnrm2(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const float> x, int incx,
                         Pointer<float> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_snrm2(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasDnrm2(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const double> x, int incx,
                         Pointer<double> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dnrm2(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasScnrm2(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_float_complex> x, int incx,
                          Pointer<float> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_scnrm2(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasDznrm2(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_double_complex> x, int incx,
                          Pointer<double> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dznrm2(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasSdot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<const float> x, int incx,
                        Pointer<const float> y, int incy,
                        Pointer<float> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_sdot(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                               ToRocm(result)));
}

llvm::Error RocblasDdot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<const double> x, int incx,
                        Pointer<const double> y, int incy,
                        Pointer<double> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ddot(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                               ToRocm(result)));
}

llvm::Error RocblasCdotu(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cdotu(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                                ToRocm(result)));
}

llvm::Error RocblasCdotc(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cdotc(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                                ToRocm(result)));
}

llvm::Error RocblasZdotu(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zdotu(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                                ToRocm(result)));
}

llvm::Error RocblasZdotc(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zdotc(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                                ToRocm(result)));
}

llvm::Error RocblasSscal(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const float> alpha, Pointer<float> x,
                         int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_sscal(handle, n, ToRocm(alpha), ToRocm(x), incx));
}

llvm::Error RocblasDscal(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const double> alpha, Pointer<double> x,
                         int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dscal(handle, n, ToRocm(alpha), ToRocm(x), incx));
}

llvm::Error RocblasCscal(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<rocblas_float_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cscal(handle, n, ToRocm(alpha), ToRocm(x), incx));
}

llvm::Error RocblasCsscal(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const float> alpha,
                          Pointer<rocblas_float_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_csscal(handle, n, ToRocm(alpha), ToRocm(x), incx));
}

llvm::Error RocblasZscal(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<rocblas_double_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zscal(handle, n, ToRocm(alpha), ToRocm(x), incx));
}

llvm::Error RocblasZdscal(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const double> alpha,
                          Pointer<rocblas_double_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zdscal(handle, n, ToRocm(alpha), ToRocm(x), incx));
}

llvm::Error RocblasSaxpy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const float> alpha, Pointer<const float> x,
                         int incx, Pointer<float> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_saxpy(handle, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy));
}

llvm::Error RocblasDaxpy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const double> alpha, Pointer<const double> x,
                         int incx, Pointer<double> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_daxpy(handle, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy));
}

llvm::Error RocblasCaxpy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<rocblas_float_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_caxpy(handle, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy));
}

llvm::Error RocblasZaxpy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<rocblas_double_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zaxpy(handle, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy));
}

llvm::Error RocblasScopy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const float> x, int incx, Pointer<float> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_scopy(handle, n, ToRocm(x), incx, ToRocm(y), incy));
}

llvm::Error RocblasDcopy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const double> x, int incx, Pointer<double> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dcopy(handle, n, ToRocm(x), incx, ToRocm(y), incy));
}

llvm::Error RocblasCcopy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<rocblas_float_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ccopy(handle, n, ToRocm(x), incx, ToRocm(y), incy));
}

llvm::Error RocblasZcopy(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<rocblas_double_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zcopy(handle, n, ToRocm(x), incx, ToRocm(y), incy));
}

llvm::Error RocblasSswap(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<float> x, int incx, Pointer<float> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_sswap(handle, n, ToRocm(x), incx, ToRocm(y), incy));
}

llvm::Error RocblasDswap(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<double> x, int incx, Pointer<double> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dswap(handle, n, ToRocm(x), incx, ToRocm(y), incy));
}

llvm::Error RocblasCswap(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<rocblas_float_complex> x, int incx,
                         Pointer<rocblas_float_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cswap(handle, n, ToRocm(x), incx, ToRocm(y), incy));
}

llvm::Error RocblasZswap(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<rocblas_double_complex> x, int incx,
                         Pointer<rocblas_double_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zswap(handle, n, ToRocm(x), incx, ToRocm(y), incy));
}

llvm::Error RocblasIsamax(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const float> x, int incx,
                          Pointer<int> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_isamax(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasIdamax(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const double> x, int incx,
                          Pointer<int> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_idamax(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasIcamax(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_float_complex> x, int incx,
                          Pointer<int> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_icamax(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasIzamax(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_double_complex> x, int incx,
                          Pointer<int> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_izamax(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasIsamin(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const float> x, int incx,
                          Pointer<int> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_isamin(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasIdamin(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const double> x, int incx,
                          Pointer<int> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_idamin(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasIcamin(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_float_complex> x, int incx,
                          Pointer<int> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_icamin(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasIzamin(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_double_complex> x, int incx,
                          Pointer<int> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_izamin(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasSasum(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const float> x, int incx,
                         Pointer<float> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_sasum(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasDasum(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<const double> x, int incx,
                         Pointer<double> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dasum(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasScasum(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_float_complex> x, int incx,
                          Pointer<float> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_scasum(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasDzasum(CurrentContext current, rocblas_handle handle, int n,
                          Pointer<const rocblas_double_complex> x, int incx,
                          Pointer<double> result) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dzasum(handle, n, ToRocm(x), incx, ToRocm(result)));
}

llvm::Error RocblasSrot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<float> x, int incx, Pointer<float> y, int incy,
                        Pointer<const float> c, Pointer<const float> s) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_srot(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                               ToRocm(c), ToRocm(s)));
}

llvm::Error RocblasDrot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<double> x, int incx, Pointer<double> y,
                        int incy, Pointer<const double> c,
                        Pointer<const double> s) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_drot(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                               ToRocm(c), ToRocm(s)));
}

llvm::Error RocblasCrot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<rocblas_float_complex> x, int incx,
                        Pointer<rocblas_float_complex> y, int incy,
                        Pointer<const float> c,
                        Pointer<const rocblas_float_complex> s) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_crot(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                               ToRocm(c), ToRocm(s)));
}

llvm::Error RocblasCsrot(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<rocblas_float_complex> x, int incx,
                         Pointer<rocblas_float_complex> y, int incy,
                         Pointer<const float> c, Pointer<const float> s) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_csrot(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                                ToRocm(c), ToRocm(s)));
}

llvm::Error RocblasZrot(CurrentContext current, rocblas_handle handle, int n,
                        Pointer<rocblas_double_complex> x, int incx,
                        Pointer<rocblas_double_complex> y, int incy,
                        Pointer<const double> c,
                        Pointer<const rocblas_double_complex> s) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zrot(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                               ToRocm(c), ToRocm(s)));
}

llvm::Error RocblasZdrot(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<rocblas_double_complex> x, int incx,
                         Pointer<rocblas_double_complex> y, int incy,
                         Pointer<const double> c, Pointer<const double> s) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zdrot(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                                ToRocm(c), ToRocm(s)));
}

llvm::Error RocblasSrotg(CurrentContext current, rocblas_handle handle,
                         Pointer<float> a, Pointer<float> b, Pointer<float> c,
                         Pointer<float> s) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_srotg(handle, ToRocm(a), ToRocm(b), ToRocm(c), ToRocm(s)));
}

llvm::Error RocblasDrotg(CurrentContext current, rocblas_handle handle,
                         Pointer<double> a, Pointer<double> b,
                         Pointer<double> c, Pointer<double> s) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_drotg(handle, ToRocm(a), ToRocm(b), ToRocm(c), ToRocm(s)));
}

llvm::Error RocblasCrotg(CurrentContext current, rocblas_handle handle,
                         Pointer<rocblas_float_complex> a,
                         Pointer<rocblas_float_complex> b, Pointer<float> c,
                         Pointer<rocblas_float_complex> s) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_crotg(handle, ToRocm(a), ToRocm(b), ToRocm(c), ToRocm(s)));
}

llvm::Error RocblasZrotg(CurrentContext current, rocblas_handle handle,
                         Pointer<rocblas_double_complex> a,
                         Pointer<rocblas_double_complex> b, Pointer<double> c,
                         Pointer<rocblas_double_complex> s) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_zrotg(handle, ToRocm(a), ToRocm(b), ToRocm(c), ToRocm(s)));
}

llvm::Error RocblasSrotm(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<float> x, int incx, Pointer<float> y, int incy,
                         Pointer<const float> param) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_srotm(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                                ToRocm(param)));
}

llvm::Error RocblasDrotm(CurrentContext current, rocblas_handle handle, int n,
                         Pointer<double> x, int incx, Pointer<double> y,
                         int incy, Pointer<const double> param) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_drotm(handle, n, ToRocm(x), incx, ToRocm(y), incy,
                                ToRocm(param)));
}

llvm::Error RocblasSrotmg(CurrentContext current, rocblas_handle handle,
                          Pointer<float> d1, Pointer<float> d2,
                          Pointer<float> x1, Pointer<const float> y1,
                          Pointer<float> param) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_srotmg(handle, ToRocm(d1), ToRocm(d2), ToRocm(x1),
                                 ToRocm(y1), ToRocm(param)));
}

llvm::Error RocblasDrotmg(CurrentContext current, rocblas_handle handle,
                          Pointer<double> d1, Pointer<double> d2,
                          Pointer<double> x1, Pointer<const double> y1,
                          Pointer<double> param) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_drotmg(handle, ToRocm(d1), ToRocm(d2), ToRocm(x1),
                                 ToRocm(y1), ToRocm(param)));
}

llvm::Error RocblasSgemv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n,
                         Pointer<const float> alpha, Pointer<const float> A,
                         int lda, Pointer<const float> x, int incx,
                         Pointer<const float> beta, Pointer<float> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_sgemv(handle, trans, m, n, ToRocm(alpha), ToRocm(A),
                                lda, ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasDgemv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n,
                         Pointer<const double> alpha, Pointer<const double> A,
                         int lda, Pointer<const double> x, int incx,
                         Pointer<const double> beta, Pointer<double> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dgemv(handle, trans, m, n, ToRocm(alpha), ToRocm(A),
                                lda, ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasCgemv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cgemv(handle, trans, m, n, ToRocm(alpha), ToRocm(A),
                                lda, ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasZgemv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zgemv(handle, trans, m, n, ToRocm(alpha), ToRocm(A),
                                lda, ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasSgbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n, int kl, int ku,
                         Pointer<const float> alpha, Pointer<const float> A,
                         int lda, Pointer<const float> x, int incx,
                         Pointer<const float> beta, Pointer<float> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_sgbmv(handle, trans, m, n, kl, ku, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(x), incx, ToRocm(beta),
                                ToRocm(y), incy));
}

llvm::Error RocblasDgbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n, int kl, int ku,
                         Pointer<const double> alpha, Pointer<const double> A,
                         int lda, Pointer<const double> x, int incx,
                         Pointer<const double> beta, Pointer<double> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dgbmv(handle, trans, m, n, kl, ku, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(x), incx, ToRocm(beta),
                                ToRocm(y), incy));
}

llvm::Error RocblasCgbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n, int kl, int ku,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cgbmv(handle, trans, m, n, kl, ku, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(x), incx, ToRocm(beta),
                                ToRocm(y), incy));
}

llvm::Error RocblasZgbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_operation trans, int m, int n, int kl, int ku,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zgbmv(handle, trans, m, n, kl, ku, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(x), incx, ToRocm(beta),
                                ToRocm(y), incy));
}

llvm::Error RocblasStrmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const float> A,
                         int lda, Pointer<float> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_strmv(handle, uplo, trans, diag, n, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasDtrmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const double> A,
                         int lda, Pointer<double> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dtrmv(handle, uplo, trans, diag, n, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasCtrmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ctrmv(handle, uplo, trans, diag, n, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasZtrmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ztrmv(handle, uplo, trans, diag, n, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasStbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const float> A, int lda, Pointer<float> x,
                         int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_stbmv(handle, uplo, trans, diag, n, k, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasDtbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const double> A, int lda, Pointer<double> x,
                         int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dtbmv(handle, uplo, trans, diag, n, k, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasCtbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ctbmv(handle, uplo, trans, diag, n, k, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasZtbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ztbmv(handle, uplo, trans, diag, n, k, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasStpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const float> AP,
                         Pointer<float> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_stpmv(handle, uplo, trans, diag, n, ToRocm(AP), ToRocm(x), incx));
}

llvm::Error RocblasDtpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const double> AP,
                         Pointer<double> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_dtpmv(handle, uplo, trans, diag, n, ToRocm(AP), ToRocm(x), incx));
}

llvm::Error RocblasCtpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_float_complex> AP,
                         Pointer<rocblas_float_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_ctpmv(handle, uplo, trans, diag, n, ToRocm(AP), ToRocm(x), incx));
}

llvm::Error RocblasZtpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_double_complex> AP,
                         Pointer<rocblas_double_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_ztpmv(handle, uplo, trans, diag, n, ToRocm(AP), ToRocm(x), incx));
}

llvm::Error RocblasStrsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const float> A,
                         int lda, Pointer<float> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_strsv(handle, uplo, trans, diag, n, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasDtrsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const double> A,
                         int lda, Pointer<double> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dtrsv(handle, uplo, trans, diag, n, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasCtrsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ctrsv(handle, uplo, trans, diag, n, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasZtrsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ztrsv(handle, uplo, trans, diag, n, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasStpsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const float> AP,
                         Pointer<float> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_stpsv(handle, uplo, trans, diag, n, ToRocm(AP), ToRocm(x), incx));
}

llvm::Error RocblasDtpsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, Pointer<const double> AP,
                         Pointer<double> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_dtpsv(handle, uplo, trans, diag, n, ToRocm(AP), ToRocm(x), incx));
}

llvm::Error RocblasCtpsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_float_complex> AP,
                         Pointer<rocblas_float_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_ctpsv(handle, uplo, trans, diag, n, ToRocm(AP), ToRocm(x), incx));
}

llvm::Error RocblasZtpsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n,
                         Pointer<const rocblas_double_complex> AP,
                         Pointer<rocblas_double_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(
      rocblas_ztpsv(handle, uplo, trans, diag, n, ToRocm(AP), ToRocm(x), incx));
}

llvm::Error RocblasStbsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const float> A, int lda, Pointer<float> x,
                         int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_stbsv(handle, uplo, trans, diag, n, k, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasDtbsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const double> A, int lda, Pointer<double> x,
                         int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dtbsv(handle, uplo, trans, diag, n, k, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasCtbsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ctbsv(handle, uplo, trans, diag, n, k, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasZtbsv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans,
                         rocblas_diagonal diag, int n, int k,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> x, int incx) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ztbsv(handle, uplo, trans, diag, n, k, ToRocm(A), lda,
                                ToRocm(x), incx));
}

llvm::Error RocblasSsymv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const float> alpha,
                         Pointer<const float> A, int lda,
                         Pointer<const float> x, int incx,
                         Pointer<const float> beta, Pointer<float> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ssymv(handle, uplo, n, ToRocm(alpha), ToRocm(A), lda,
                                ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasDsymv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const double> alpha,
                         Pointer<const double> A, int lda,
                         Pointer<const double> x, int incx,
                         Pointer<const double> beta, Pointer<double> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dsymv(handle, uplo, n, ToRocm(alpha), ToRocm(A), lda,
                                ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasCsymv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_csymv(handle, uplo, n, ToRocm(alpha), ToRocm(A), lda,
                                ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasZsymv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zsymv(handle, uplo, n, ToRocm(alpha), ToRocm(A), lda,
                                ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasChemv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_chemv(handle, uplo, n, ToRocm(alpha), ToRocm(A), lda,
                                ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasZhemv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zhemv(handle, uplo, n, ToRocm(alpha), ToRocm(A), lda,
                                ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasSsbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, int k,
                         Pointer<const float> alpha, Pointer<const float> A,
                         int lda, Pointer<const float> x, int incx,
                         Pointer<const float> beta, Pointer<float> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ssbmv(handle, uplo, n, k, ToRocm(alpha), ToRocm(A),
                                lda, ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasDsbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, int k,
                         Pointer<const double> alpha, Pointer<const double> A,
                         int lda, Pointer<const double> x, int incx,
                         Pointer<const double> beta, Pointer<double> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dsbmv(handle, uplo, n, k, ToRocm(alpha), ToRocm(A),
                                lda, ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasChbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, int k,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_chbmv(handle, uplo, n, k, ToRocm(alpha), ToRocm(A),
                                lda, ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasZhbmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, int k,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zhbmv(handle, uplo, n, k, ToRocm(alpha), ToRocm(A),
                                lda, ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasSspmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const float> alpha,
                         Pointer<const float> AP, Pointer<const float> x,
                         int incx, Pointer<const float> beta, Pointer<float> y,
                         int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_sspmv(handle, uplo, n, ToRocm(alpha), ToRocm(AP),
                                ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasDspmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const double> alpha,
                         Pointer<const double> AP, Pointer<const double> x,
                         int incx, Pointer<const double> beta,
                         Pointer<double> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dspmv(handle, uplo, n, ToRocm(alpha), ToRocm(AP),
                                ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasChpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> AP,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_chpmv(handle, uplo, n, ToRocm(alpha), ToRocm(AP),
                                ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasZhpmv(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> AP,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> y, int incy) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zhpmv(handle, uplo, n, ToRocm(alpha), ToRocm(AP),
                                ToRocm(x), incx, ToRocm(beta), ToRocm(y),
                                incy));
}

llvm::Error RocblasSger(CurrentContext current, rocblas_handle handle, int m,
                        int n, Pointer<const float> alpha,
                        Pointer<const float> x, int incx,
                        Pointer<const float> y, int incy, Pointer<float> A,
                        int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_sger(handle, m, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasDger(CurrentContext current, rocblas_handle handle, int m,
                        int n, Pointer<const double> alpha,
                        Pointer<const double> x, int incx,
                        Pointer<const double> y, int incy, Pointer<double> A,
                        int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dger(handle, m, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasCgeru(CurrentContext current, rocblas_handle handle, int m,
                         int n, Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cgeru(handle, m, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasCgerc(CurrentContext current, rocblas_handle handle, int m,
                         int n, Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cgerc(handle, m, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasZgeru(CurrentContext current, rocblas_handle handle, int m,
                         int n, Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zgeru(handle, m, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasZgerc(CurrentContext current, rocblas_handle handle, int m,
                         int n, Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zgerc(handle, m, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasSsyr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const float> alpha,
                        Pointer<const float> x, int incx, Pointer<float> A,
                        int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ssyr(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(A), lda));
}

llvm::Error RocblasDsyr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const double> alpha,
                        Pointer<const double> x, int incx, Pointer<double> A,
                        int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dsyr(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(A), lda));
}

llvm::Error RocblasCsyr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n,
                        Pointer<const rocblas_float_complex> alpha,
                        Pointer<const rocblas_float_complex> x, int incx,
                        Pointer<rocblas_float_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_csyr(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(A), lda));
}

llvm::Error RocblasZsyr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n,
                        Pointer<const rocblas_double_complex> alpha,
                        Pointer<const rocblas_double_complex> x, int incx,
                        Pointer<rocblas_double_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zsyr(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(A), lda));
}

llvm::Error RocblasCher(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const float> alpha,
                        Pointer<const rocblas_float_complex> x, int incx,
                        Pointer<rocblas_float_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cher(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(A), lda));
}

llvm::Error RocblasZher(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const double> alpha,
                        Pointer<const rocblas_double_complex> x, int incx,
                        Pointer<rocblas_double_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zher(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(A), lda));
}

llvm::Error RocblasSspr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const float> alpha,
                        Pointer<const float> x, int incx, Pointer<float> AP) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_sspr(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(AP)));
}

llvm::Error RocblasDspr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const double> alpha,
                        Pointer<const double> x, int incx, Pointer<double> AP) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dspr(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(AP)));
}

llvm::Error RocblasChpr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const float> alpha,
                        Pointer<const rocblas_float_complex> x, int incx,
                        Pointer<rocblas_float_complex> AP) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_chpr(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(AP)));
}

llvm::Error RocblasZhpr(CurrentContext current, rocblas_handle handle,
                        rocblas_fill uplo, int n, Pointer<const double> alpha,
                        Pointer<const rocblas_double_complex> x, int incx,
                        Pointer<rocblas_double_complex> AP) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zhpr(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                               ToRocm(AP)));
}

llvm::Error RocblasSsyr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const float> alpha,
                         Pointer<const float> x, int incx,
                         Pointer<const float> y, int incy, Pointer<float> A,
                         int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ssyr2(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasDsyr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const double> alpha,
                         Pointer<const double> x, int incx,
                         Pointer<const double> y, int incy, Pointer<double> A,
                         int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dsyr2(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasCsyr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_csyr2(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasZsyr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zsyr2(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasCher2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cher2(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasZher2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> A, int lda) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zher2(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(A), lda));
}

llvm::Error RocblasSspr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const float> alpha,
                         Pointer<const float> x, int incx,
                         Pointer<const float> y, int incy, Pointer<float> AP) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_sspr2(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(AP)));
}

llvm::Error RocblasDspr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n, Pointer<const double> alpha,
                         Pointer<const double> x, int incx,
                         Pointer<const double> y, int incy,
                         Pointer<double> AP) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dspr2(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(AP)));
}

llvm::Error RocblasChpr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> x, int incx,
                         Pointer<const rocblas_float_complex> y, int incy,
                         Pointer<rocblas_float_complex> AP) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_chpr2(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(AP)));
}

llvm::Error RocblasZhpr2(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> x, int incx,
                         Pointer<const rocblas_double_complex> y, int incy,
                         Pointer<rocblas_double_complex> AP) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zhpr2(handle, uplo, n, ToRocm(alpha), ToRocm(x), incx,
                                ToRocm(y), incy, ToRocm(AP)));
}

llvm::Error RocblasHgemm(CurrentContext current, rocblas_handle handle,
                         rocblas_operation transa, rocblas_operation transb,
                         int m, int n, int k, Pointer<const rocblas_half> alpha,
                         Pointer<const rocblas_half> A, int lda,
                         Pointer<const rocblas_half> B, int ldb,
                         Pointer<const rocblas_half> beta,
                         Pointer<rocblas_half> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_hgemm(handle, transa, transb, m, n, k, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                ToRocm(C), ldc));
}

llvm::Error RocblasSgemm(CurrentContext current, rocblas_handle handle,
                         rocblas_operation transa, rocblas_operation transb,
                         int m, int n, int k, Pointer<const float> alpha,
                         Pointer<const float> A, int lda,
                         Pointer<const float> B, int ldb,
                         Pointer<const float> beta, Pointer<float> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_sgemm(handle, transa, transb, m, n, k, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                ToRocm(C), ldc));
}

llvm::Error RocblasDgemm(CurrentContext current, rocblas_handle handle,
                         rocblas_operation transa, rocblas_operation transb,
                         int m, int n, int k, Pointer<const double> alpha,
                         Pointer<const double> A, int lda,
                         Pointer<const double> B, int ldb,
                         Pointer<const double> beta, Pointer<double> C,
                         int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dgemm(handle, transa, transb, m, n, k, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                ToRocm(C), ldc));
}

llvm::Error RocblasCgemm(CurrentContext current, rocblas_handle handle,
                         rocblas_operation transa, rocblas_operation transb,
                         int m, int n, int k,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> B, int ldb,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cgemm(handle, transa, transb, m, n, k, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                ToRocm(C), ldc));
}

llvm::Error RocblasZgemm(CurrentContext current, rocblas_handle handle,
                         rocblas_operation transa, rocblas_operation transb,
                         int m, int n, int k,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> B, int ldb,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zgemm(handle, transa, transb, m, n, k, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                ToRocm(C), ldc));
}

llvm::Error RocblasSsyrk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const float> alpha,
                         Pointer<const float> A, int lda,
                         Pointer<const float> beta, Pointer<float> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ssyrk(handle, uplo, trans, n, k, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(beta), ToRocm(C), ldc));
}

llvm::Error RocblasDsyrk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const double> alpha,
                         Pointer<const double> A, int lda,
                         Pointer<const double> beta, Pointer<double> C,
                         int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dsyrk(handle, uplo, trans, n, k, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(beta), ToRocm(C), ldc));
}

llvm::Error RocblasCsyrk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_csyrk(handle, uplo, trans, n, k, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(beta), ToRocm(C), ldc));
}

llvm::Error RocblasZsyrk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zsyrk(handle, uplo, trans, n, k, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(beta), ToRocm(C), ldc));
}

llvm::Error RocblasCherk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const float> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const float> beta,
                         Pointer<rocblas_float_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cherk(handle, uplo, trans, n, k, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(beta), ToRocm(C), ldc));
}

llvm::Error RocblasZherk(CurrentContext current, rocblas_handle handle,
                         rocblas_fill uplo, rocblas_operation trans, int n,
                         int k, Pointer<const double> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const double> beta,
                         Pointer<rocblas_double_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zherk(handle, uplo, trans, n, k, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(beta), ToRocm(C), ldc));
}

llvm::Error RocblasSsyr2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const float> alpha,
                          Pointer<const float> A, int lda,
                          Pointer<const float> B, int ldb,
                          Pointer<const float> beta, Pointer<float> C,
                          int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ssyr2k(handle, uplo, trans, n, k, ToRocm(alpha),
                                 ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                 ToRocm(C), ldc));
}

llvm::Error RocblasDsyr2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const double> alpha,
                          Pointer<const double> A, int lda,
                          Pointer<const double> B, int ldb,
                          Pointer<const double> beta, Pointer<double> C,
                          int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dsyr2k(handle, uplo, trans, n, k, ToRocm(alpha),
                                 ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                 ToRocm(C), ldc));
}

llvm::Error RocblasCsyr2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const rocblas_float_complex> alpha,
                          Pointer<const rocblas_float_complex> A, int lda,
                          Pointer<const rocblas_float_complex> B, int ldb,
                          Pointer<const rocblas_float_complex> beta,
                          Pointer<rocblas_float_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_csyr2k(handle, uplo, trans, n, k, ToRocm(alpha),
                                 ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                 ToRocm(C), ldc));
}

llvm::Error RocblasZsyr2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const rocblas_double_complex> alpha,
                          Pointer<const rocblas_double_complex> A, int lda,
                          Pointer<const rocblas_double_complex> B, int ldb,
                          Pointer<const rocblas_double_complex> beta,
                          Pointer<rocblas_double_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zsyr2k(handle, uplo, trans, n, k, ToRocm(alpha),
                                 ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                 ToRocm(C), ldc));
}

llvm::Error RocblasCher2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const rocblas_float_complex> alpha,
                          Pointer<const rocblas_float_complex> A, int lda,
                          Pointer<const rocblas_float_complex> B, int ldb,
                          Pointer<const float> beta,
                          Pointer<rocblas_float_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_cher2k(handle, uplo, trans, n, k, ToRocm(alpha),
                                 ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                 ToRocm(C), ldc));
}

llvm::Error RocblasZher2k(CurrentContext current, rocblas_handle handle,
                          rocblas_fill uplo, rocblas_operation trans, int n,
                          int k, Pointer<const rocblas_double_complex> alpha,
                          Pointer<const rocblas_double_complex> A, int lda,
                          Pointer<const rocblas_double_complex> B, int ldb,
                          Pointer<const double> beta,
                          Pointer<rocblas_double_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zher2k(handle, uplo, trans, n, k, ToRocm(alpha),
                                 ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                 ToRocm(C), ldc));
}

llvm::Error RocblasSsymm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const float> alpha, Pointer<const float> A,
                         int lda, Pointer<const float> B, int ldb,
                         Pointer<const float> beta, Pointer<float> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ssymm(handle, side, uplo, m, n, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                ToRocm(C), ldc));
}

llvm::Error RocblasDsymm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const double> alpha, Pointer<const double> A,
                         int lda, Pointer<const double> B, int ldb,
                         Pointer<const double> beta, Pointer<double> C,
                         int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dsymm(handle, side, uplo, m, n, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                ToRocm(C), ldc));
}

llvm::Error RocblasCsymm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> B, int ldb,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_csymm(handle, side, uplo, m, n, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                ToRocm(C), ldc));
}

llvm::Error RocblasZsymm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> B, int ldb,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zsymm(handle, side, uplo, m, n, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                ToRocm(C), ldc));
}

llvm::Error RocblasChemm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<const rocblas_float_complex> B, int ldb,
                         Pointer<const rocblas_float_complex> beta,
                         Pointer<rocblas_float_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_chemm(handle, side, uplo, m, n, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                ToRocm(C), ldc));
}

llvm::Error RocblasZhemm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo, int m, int n,
                         Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<const rocblas_double_complex> B, int ldb,
                         Pointer<const rocblas_double_complex> beta,
                         Pointer<rocblas_double_complex> C, int ldc) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_zhemm(handle, side, uplo, m, n, ToRocm(alpha),
                                ToRocm(A), lda, ToRocm(B), ldb, ToRocm(beta),
                                ToRocm(C), ldc));
}

llvm::Error RocblasStrsm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const float> alpha,
                         Pointer<const float> A, int lda, Pointer<float> B,
                         int ldb) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_strsm(handle, side, uplo, trans, diag, m, n,
                                ToRocm(alpha), ToRocm(A), lda, ToRocm(B), ldb));
}

llvm::Error RocblasDtrsm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const double> alpha,
                         Pointer<const double> A, int lda, Pointer<double> B,
                         int ldb) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dtrsm(handle, side, uplo, trans, diag, m, n,
                                ToRocm(alpha), ToRocm(A), lda, ToRocm(B), ldb));
}

llvm::Error RocblasCtrsm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> B, int ldb) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ctrsm(handle, side, uplo, trans, diag, m, n,
                                ToRocm(alpha), ToRocm(A), lda, ToRocm(B), ldb));
}

llvm::Error RocblasZtrsm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> B, int ldb) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ztrsm(handle, side, uplo, trans, diag, m, n,
                                ToRocm(alpha), ToRocm(A), lda, ToRocm(B), ldb));
}

llvm::Error RocblasStrmm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const float> alpha,
                         Pointer<const float> A, int lda, Pointer<float> B,
                         int ldb) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_strmm(handle, side, uplo, trans, diag, m, n,
                                ToRocm(alpha), ToRocm(A), lda, ToRocm(B), ldb));
}

llvm::Error RocblasDtrmm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const double> alpha,
                         Pointer<const double> A, int lda, Pointer<double> B,
                         int ldb) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_dtrmm(handle, side, uplo, trans, diag, m, n,
                                ToRocm(alpha), ToRocm(A), lda, ToRocm(B), ldb));
}

llvm::Error RocblasCtrmm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const rocblas_float_complex> alpha,
                         Pointer<const rocblas_float_complex> A, int lda,
                         Pointer<rocblas_float_complex> B, int ldb) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ctrmm(handle, side, uplo, trans, diag, m, n,
                                ToRocm(alpha), ToRocm(A), lda, ToRocm(B), ldb));
}

llvm::Error RocblasZtrmm(CurrentContext current, rocblas_handle handle,
                         rocblas_side side, rocblas_fill uplo,
                         rocblas_operation trans, rocblas_diagonal diag, int m,
                         int n, Pointer<const rocblas_double_complex> alpha,
                         Pointer<const rocblas_double_complex> A, int lda,
                         Pointer<rocblas_double_complex> B, int ldb) {
  CheckHipContext(current);
  return TO_ERROR(rocblas_ztrmm(handle, side, uplo, trans, diag, m, n,
                                ToRocm(alpha), ToRocm(A), lda, ToRocm(B), ldb));
}

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt
