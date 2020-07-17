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

//===- cusolver_wrapper.cc --------------------------------------*- C++ -*-===//
//
// Thin wrapper around the cuSOLVER API adding llvm::Error.
//
//===----------------------------------------------------------------------===//
#include "tfrt/gpu/stream/cusolver_wrapper.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "wrapper_detail.h"

#define RETURN_IF_ERROR(expr)                                   \
  while (cusolverStatus_t _result = expr) {                     \
    return llvm::make_error<CusolverErrorInfo>(                 \
        CusolverErrorData{_result, #expr, CreateStackTrace()}); \
  }

#define TO_ERROR(expr)                                                     \
  [](cusolverStatus_t _result) -> llvm::Error {                            \
    if (_result == CUSOLVER_STATUS_SUCCESS) return llvm::Error::success(); \
    return llvm::make_error<CusolverErrorInfo>(                            \
        CusolverErrorData{_result, #expr, CreateStackTrace()});            \
  }(expr)

namespace tfrt {
namespace gpu {
namespace stream {

static llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     cusolverStatus_t status) {
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:
      return os << "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return os << "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return os << "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return os << "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return os << "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return os << "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return os << "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return os << "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return os << "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return os << "CUSOLVER_STATUS_NOT_SUPPORTED";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return os << "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return os << "CUSOLVER_STATUS_INVALID_LICENSE";
    default:
      return os << llvm::formatv("cusolverStatus_t({0})",
                                 static_cast<int>(status));
  }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const CusolverErrorData& data) {
  os << "'" << data.expr << "': " << data.result;
  if (data.stack_trace) os << ", stack trace:\n" << data.stack_trace;
  return os;
}

cusolverStatus_t GetResult(const CusolverErrorInfo& info) {
  return info.get<CusolverErrorData>().result;
}

template <typename T>
static T* ToCuda(Pointer<T> ptr) {
  return ptr.raw(Platform::CUDA);
}

llvm::Expected<OwningSolverDnHandle> CusolverDnCreate(CurrentContext current) {
  CheckCudaContext(current);
  cusolverDnHandle_t handle = nullptr;
  RETURN_IF_ERROR(cusolverDnCreate(&handle));
  return OwningSolverDnHandle(handle);
}

llvm::Error CusolverDnDestroy(cusolverDnHandle_t handle) {
  return TO_ERROR(cusolverDnDestroy(handle));
}

llvm::Error CusolverDnSetStream(cusolverDnHandle_t handle,
                                cudaStream_t stream) {
  return TO_ERROR(cusolverDnSetStream(handle, stream));
}

llvm::Expected<Stream> CusolverDnGetStream(cusolverDnHandle_t handle) {
  cudaStream_t stream = nullptr;
  RETURN_IF_ERROR(cusolverDnGetStream(handle, &stream));
  return Stream(stream);
}

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt
