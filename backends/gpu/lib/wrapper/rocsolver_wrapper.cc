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
#include "tfrt/gpu/wrapper/rocsolver_wrapper.h"

#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

template llvm::raw_ostream& internal::operator<<(
    llvm::raw_ostream&, const ErrorData<rocblas_status>&);

llvm::Expected<OwningSolverHandle> RocsolverCreate() {
  rocblas_handle handle = nullptr;
  RETURN_IF_ERROR(rocblas_create_handle(&handle));
  return OwningSolverHandle(handle);
}

llvm::Error RocsolverDestroy(rocblas_handle handle) {
  return TO_ERROR(rocblas_destroy_handle(handle));
}

llvm::Error RocsolverSetStream(rocblas_handle handle, hipStream_t stream) {
  return TO_ERROR(rocblas_set_stream(handle, stream));
}

llvm::Expected<Stream> RocsolverGetStream(rocblas_handle handle) {
  hipStream_t stream = nullptr;
  RETURN_IF_ERROR(rocblas_get_stream(handle, &stream));
  return Stream(stream);
}

llvm::Error RocsolverPotrf(CurrentContext current, rocblas_handle handle,
                           rocblas_fill fillMode, int n, Pointer<float> A,
                           int heightA, Pointer<int> devInfo) {
  CheckHipContext(current);
  return TO_ERROR(rocsolver_spotrf(handle, fillMode, n, ToRocm(A), heightA,
                                   ToRocm(devInfo)));
}

llvm::Error RocsolverPotrf(CurrentContext current, rocblas_handle handle,
                           rocblas_fill fillMode, int n, Pointer<double> A,
                           int heightA, Pointer<int> devInfo) {
  CheckHipContext(current);
  return TO_ERROR(rocsolver_dpotrf(handle, fillMode, n, ToRocm(A), heightA,
                                   ToRocm(devInfo)));
}

llvm::Error RocsolverPotrf(CurrentContext current, rocblas_handle handle,
                           rocblas_fill fillMode, int n,
                           Pointer<rocblas_float_complex> A, int heightA,
                           Pointer<int> devInfo) {
  CheckHipContext(current);
  return TO_ERROR(rocsolver_cpotrf(handle, fillMode, n, ToRocm(A), heightA,
                                   ToRocm(devInfo)));
}

llvm::Error RocsolverPotrf(CurrentContext current, rocblas_handle handle,
                           rocblas_fill fillMode, int n,
                           Pointer<rocblas_double_complex> A, int heightA,
                           Pointer<int> devInfo) {
  CheckHipContext(current);
  return TO_ERROR(rocsolver_zpotrf(handle, fillMode, n, ToRocm(A), heightA,
                                   ToRocm(devInfo)));
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
