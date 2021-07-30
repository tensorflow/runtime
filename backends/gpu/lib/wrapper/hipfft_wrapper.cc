// Copyright 2021 The TensorFlow Runtime Authors
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

// Thin wrapper around the cuFFT API that adapts the library for TFRT
// conventions.
#include "tfrt/gpu/wrapper/hipfft_wrapper.h"

#include <cstddef>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::Expected<LibraryVersion> HipfftGetVersion() {
  LibraryVersion version;
  RETURN_IF_ERROR(hipfftGetProperty(HIPFFT_MAJOR_VERSION, &version.major));
  RETURN_IF_ERROR(hipfftGetProperty(HIPFFT_MINOR_VERSION, &version.minor));
  RETURN_IF_ERROR(hipfftGetProperty(HIPFFT_PATCH_LEVEL, &version.patch));
  return version;
}

llvm::Error HipfftDestroy(hipfftHandle plan) {
  return TO_ERROR(hipfftDestroy(plan));
}

llvm::Error HipfftSetStream(hipfftHandle plan, hipStream_t stream) {
  return TO_ERROR(hipfftSetStream(plan, stream));
}

llvm::Expected<OwningFftHandle> HipfftPlan1d(int nx, hipfftType type,
                                             int batch) {
  hipfftHandle plan;
  RETURN_IF_ERROR(hipfftPlan1d(&plan, nx, type, batch));
  return OwningFftHandle(plan);
}

llvm::Expected<OwningFftHandle> HipfftPlan2d(int nx, int ny, hipfftType type) {
  hipfftHandle plan;
  RETURN_IF_ERROR(hipfftPlan2d(&plan, nx, ny, type));
  return OwningFftHandle(plan);
}

llvm::Expected<OwningFftHandle> HipfftPlan3d(int nx, int ny, int nz,
                                             hipfftType type) {
  hipfftHandle plan;
  RETURN_IF_ERROR(hipfftPlan3d(&plan, nx, ny, nz, type));
  return OwningFftHandle(plan);
}

// llvm::Expected<OwningFftHandle> HipfftPlanMany(
//     hipfftType type, int batch, const CufftManyOptions<int>& options) {
//   if (auto err = ValidateOptions(options)) return std::move(err);

//   cufftHandle plan;
//   RETURN_IF_ERROR(cufftPlanMany(
//       &plan, options.rank, ToCufft(options.dims),
//       ToCufft(options.input_embed), options.input_stride, options.input_dist,
//       ToCufft(options.output_embed), options.output_stride,
//       options.output_dist, type, batch));
//   return OwningFftHandle(plan);
// }

llvm::Expected<size_t> HipfftGetSize(hipfftHandle plan) {
  size_t work_size;
  RETURN_IF_ERROR(hipfftGetSize(plan, &work_size));
  return work_size;
}

llvm::Error HipfftSetWorkArea(hipfftHandle plan, Pointer<void> work_area) {
  return TO_ERROR(hipfftSetWorkArea(plan, ToRocm(work_area)));
}

static int ToRocm(FftDirection direction) {
  int fft_dir = 0;  // currently 0 is an invalid vallue
  switch (direction) {
    case FftDirection::kForward:
      fft_dir = HIPFFT_FORWARD;
      break;
    case FftDirection::kInverse:
      fft_dir = HIPFFT_BACKWARD;
      break;
  }
  return fft_dir;
}

llvm::Error HipfftExecC2C(hipfftHandle plan, hipfftComplex* input_data,
                          hipfftComplex* output_data, FftDirection direction) {
  return TO_ERROR(
      hipfftExecC2C(plan, input_data, output_data, ToRocm(direction)));
}

llvm::Error HipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex* input_data,
                          hipfftDoubleComplex* output_data,
                          FftDirection direction) {
  return TO_ERROR(
      hipfftExecZ2Z(plan, input_data, output_data, ToRocm(direction)));
}

llvm::Error HipfftExecR2C(hipfftHandle plan, hipfftReal* input_data,
                          hipfftComplex* output_data) {
  return TO_ERROR(hipfftExecR2C(plan, input_data, output_data));
}

llvm::Error HipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal* input_data,
                          hipfftDoubleComplex* output_data) {
  return TO_ERROR(hipfftExecD2Z(plan, input_data, output_data));
}

llvm::Error HipfftExecC2R(hipfftHandle plan, hipfftComplex* input_data,
                          hipfftReal* output_data) {
  return TO_ERROR(hipfftExecC2R(plan, input_data, output_data));
}

llvm::Error HipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex* input_data,
                          hipfftDoubleReal* output_data) {
  return TO_ERROR(hipfftExecZ2D(plan, input_data, output_data));
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
