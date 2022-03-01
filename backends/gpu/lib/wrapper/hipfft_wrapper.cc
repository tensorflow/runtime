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
#include "tfrt/gpu/wrapper/fft_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

// The hipFFT library does not enforce const correctness. We try to improve
// usability by providing some level of const correctness in the wrapper, but
// this means that we need to const cast some input pointers.
//
// This wrapper is intended to enhance readability, avoid repetition of the
// above warning, and assist in debugging if we do run into a "not-really-const"
// input issue.
// cf the coreesponding function in cufft_wrapper.cc
template <typename T>
T* ToHipfft(llvm::ArrayRef<T> t) {
  if (t.empty()) return nullptr;
  return const_cast<T*>(t.data());
}

llvm::Expected<LibraryVersion> HipfftGetVersion() {
  LibraryVersion version;
  RETURN_IF_ERROR(hipfftGetProperty(HIPFFT_MAJOR_VERSION, &version.major));
  RETURN_IF_ERROR(hipfftGetProperty(HIPFFT_MINOR_VERSION, &version.minor));
  RETURN_IF_ERROR(hipfftGetProperty(HIPFFT_PATCH_LEVEL, &version.patch));
  return version;
}

llvm::Expected<OwningFftHandle> HipfftCreate() {
  hipfftHandle plan;
  RETURN_IF_ERROR(hipfftCreate(&plan));
  return OwningFftHandle(plan);
}

llvm::Error HipfftDestroy(hipfftHandle plan) {
  return TO_ERROR(hipfftDestroy(plan));
}

llvm::Error HipfftSetStream(hipfftHandle plan, hipStream_t stream) {
  return TO_ERROR(hipfftSetStream(plan, stream));
}

llvm::Expected<size_t> HipfftMakePlanMany(hipfftHandle plan, int rank,
                                          llvm::ArrayRef<int64_t> n,
                                          llvm::ArrayRef<int64_t> inembed,
                                          int64_t istride, int64_t idist,
                                          llvm::ArrayRef<int64_t> onembed,
                                          int64_t ostride, int64_t odist,
                                          hipfftType type, int64_t batch) {
  // NOLINTNEXTLINE(google-runtime-int)
  static_assert(sizeof(int64_t) == sizeof(long long),
                "hipFFT uses long long for 64-bit values, but there is a size "
                "mismatch between long long and int64_t");
  size_t work_size;
  return hipfftMakePlanMany64(
      // NOLINTNEXTLINE(google-runtime-int)
      plan, rank, reinterpret_cast<long long*>(ToHipfft(n)),
      // NOLINTNEXTLINE(google-runtime-int)
      reinterpret_cast<long long*>(ToHipfft(inembed)), istride, idist,
      // NOLINTNEXTLINE(google-runtime-int)
      reinterpret_cast<long long*>(ToHipfft(onembed)), ostride, odist, type,
      batch, &work_size);
}

llvm::Expected<size_t> HipfftGetSize(hipfftHandle plan) {
  size_t work_size;
  RETURN_IF_ERROR(hipfftGetSize(plan, &work_size));
  return work_size;
}

llvm::Error HipfftSetWorkArea(hipfftHandle plan, Pointer<void> work_area) {
  return TO_ERROR(hipfftSetWorkArea(plan, ToRocm(work_area)));
}

static int ToRocm(FftDirection direction) {
  switch (direction) {
    case FftDirection::kForward:
      return HIPFFT_FORWARD;
    case FftDirection::kInverse:
      return HIPFFT_BACKWARD;
    default:
      return 0;  // 0 is an invalid value.
  }
}

llvm::Error HipfftExecC2C(hipfftHandle plan, Pointer<hipfftComplex> input_data,
                          Pointer<hipfftComplex> output_data,
                          FftDirection direction) {
  return TO_ERROR(hipfftExecC2C(plan, input_data.raw(Platform::ROCm),
                                output_data.raw(Platform::ROCm),
                                ToRocm(direction)));
}

llvm::Error HipfftExecZ2Z(hipfftHandle plan,
                          Pointer<hipfftDoubleComplex> input_data,
                          Pointer<hipfftDoubleComplex> output_data,
                          FftDirection direction) {
  return TO_ERROR(hipfftExecZ2Z(plan, input_data.raw(Platform::ROCm),
                                output_data.raw(Platform::ROCm),
                                ToRocm(direction)));
}

llvm::Error HipfftExecR2C(hipfftHandle plan, Pointer<hipfftReal> input_data,
                          Pointer<hipfftComplex> output_data) {
  return TO_ERROR(hipfftExecR2C(plan, input_data.raw(Platform::ROCm),
                                output_data.raw(Platform::ROCm)));
}

llvm::Error HipfftExecD2Z(hipfftHandle plan,
                          Pointer<hipfftDoubleReal> input_data,
                          Pointer<hipfftDoubleComplex> output_data) {
  return TO_ERROR(hipfftExecD2Z(plan, input_data.raw(Platform::ROCm),
                                output_data.raw(Platform::ROCm)));
}

llvm::Error HipfftExecC2R(hipfftHandle plan, Pointer<hipfftComplex> input_data,
                          Pointer<hipfftReal> output_data) {
  return TO_ERROR(hipfftExecC2R(plan, input_data.raw(Platform::ROCm),
                                output_data.raw(Platform::ROCm)));
}

llvm::Error HipfftExecZ2D(hipfftHandle plan,
                          Pointer<hipfftDoubleComplex> input_data,
                          Pointer<hipfftDoubleReal> output_data) {
  return TO_ERROR(hipfftExecZ2D(plan, input_data.raw(Platform::ROCm),
                                output_data.raw(Platform::ROCm)));
}

llvm::Error HipfftExec(hipfftHandle plan, Pointer<void> raw_input,
                       Pointer<void> raw_output, FftType type) {
  switch (type) {
    case FftType::kC2CForward:
      return HipfftExecC2C(plan, static_cast<Pointer<hipfftComplex>>(raw_input),
                           static_cast<Pointer<hipfftComplex>>(raw_output),
                           FftDirection::kForward);
    case FftType::kC2CInverse:
      return HipfftExecC2C(plan, static_cast<Pointer<hipfftComplex>>(raw_input),
                           static_cast<Pointer<hipfftComplex>>(raw_output),
                           FftDirection::kInverse);
    case FftType::kZ2ZForward:
      return HipfftExecZ2Z(
          plan, static_cast<Pointer<hipfftDoubleComplex>>(raw_input),
          static_cast<Pointer<hipfftDoubleComplex>>(raw_output),
          FftDirection::kForward);
    case FftType::kZ2ZInverse:
      return HipfftExecZ2Z(
          plan, static_cast<Pointer<hipfftDoubleComplex>>(raw_input),
          static_cast<Pointer<hipfftDoubleComplex>>(raw_output),
          FftDirection::kInverse);
    case FftType::kR2C:
      return HipfftExecR2C(plan, static_cast<Pointer<hipfftReal>>(raw_input),
                           static_cast<Pointer<hipfftComplex>>(raw_output));
    case FftType::kD2Z:
      return HipfftExecD2Z(
          plan, static_cast<Pointer<hipfftDoubleReal>>(raw_input),
          static_cast<Pointer<hipfftDoubleComplex>>(raw_output));
    case FftType::kC2R:
      return HipfftExecC2R(plan, static_cast<Pointer<hipfftComplex>>(raw_input),
                           static_cast<Pointer<hipfftReal>>(raw_output));
    case FftType::kZ2D:
      return HipfftExecZ2D(plan,
                           static_cast<Pointer<hipfftDoubleComplex>>(raw_input),
                           static_cast<Pointer<hipfftDoubleReal>>(raw_output));
    default:
      return llvm::createStringError(std::errc::invalid_argument,
                                     "invalid FFT type");
  }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, hipfftResult_t result) {
  switch (result) {
    case HIPFFT_SUCCESS:
      return os << "HIPFFT_SUCCESS";
    case HIPFFT_INVALID_PLAN:
      return os << "HIPFFT_INVALID_PLAN";
    case HIPFFT_ALLOC_FAILED:
      return os << "HIPFFT_ALLOC_FAILED";
    case HIPFFT_INVALID_TYPE:
      return os << "HIPFFT_INVALID_TYPE";
    case HIPFFT_INVALID_VALUE:
      return os << "HIPFFT_INVALID_VALUE";
    case HIPFFT_INTERNAL_ERROR:
      return os << "HIPFFT_INTERNAL_ERROR";
    case HIPFFT_EXEC_FAILED:
      return os << "HIPFFT_EXEC_FAILED";
    case HIPFFT_SETUP_FAILED:
      return os << "HIPFFT_SETUP_FAILED";
    case HIPFFT_INVALID_SIZE:
      return os << "HIPFFT_INVALID_SIZE";
    case HIPFFT_UNALIGNED_DATA:
      return os << "HIPFFT_UNALIGNED_DATA";
    case HIPFFT_INCOMPLETE_PARAMETER_LIST:
      return os << "HIPFFT_INCOMPLETE_PARAMETER_LIST";
    case HIPFFT_INVALID_DEVICE:
      return os << "HIPFFT_INVALID_DEVICE";
    case HIPFFT_PARSE_ERROR:
      return os << "HIPFFT_PARSE_ERROR";
    case HIPFFT_NO_WORKSPACE:
      return os << "HIPFFT_NO_WORKSPACE";
    case HIPFFT_NOT_IMPLEMENTED:
      return os << "HIPFFT_NOT_IMPLEMENTED";
    case HIPFFT_NOT_SUPPORTED:
      return os << "HIPFFT_NOT_SUPPORTED";
    default:
      return os << llvm::formatv("hipfftResult_t({0})",
                                 static_cast<int>(result));
  }
}

llvm::Expected<hipfftType> FftTypeToHipfftType(FftType type) {
  switch (type) {
    case FftType::kZ2ZForward:
    case FftType::kZ2ZInverse:
      return HIPFFT_Z2Z;
    case FftType::kR2C:
      return HIPFFT_R2C;
    case FftType::kC2R:
      return HIPFFT_C2R;
    case FftType::kC2CForward:
    case FftType::kC2CInverse:
      return HIPFFT_C2C;
    case FftType::kD2Z:
      return HIPFFT_D2Z;
    case FftType::kZ2D:
      return HIPFFT_Z2D;
    default:
      return MakeStringError("invalid FFT type");
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
