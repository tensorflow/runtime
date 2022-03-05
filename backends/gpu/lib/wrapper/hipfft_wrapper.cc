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

// Thin wrapper around the hipFFT API that adapts the library for TFRT
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

llvm::Expected<LibraryVersion> HipfftGetVersion() {
  LibraryVersion version;
  RETURN_IF_ERROR(hipfftGetProperty(HIPFFT_MAJOR_VERSION, &version.major));
  RETURN_IF_ERROR(hipfftGetProperty(HIPFFT_MINOR_VERSION, &version.minor));
  RETURN_IF_ERROR(hipfftGetProperty(HIPFFT_PATCH_LEVEL, &version.patch));
  return version;
}

llvm::Expected<OwningFftHandle> HipfftCreate(CurrentContext current) {
  CheckHipContext(current);
  hipfftHandle handle;
  RETURN_IF_ERROR(hipfftCreate(&handle));
  return OwningFftHandle(handle);
}

llvm::Error HipfftDestroy(hipfftHandle handle) {
  return TO_ERROR(hipfftDestroy(handle));
}

llvm::Error HipfftSetStream(hipfftHandle handle, hipStream_t stream) {
  return TO_ERROR(hipfftSetStream(handle, stream));
}

// The hipFFT library does not enforce const correctness. We try to improve
// usability by providing some level of const correctness in the wrapper, but
// this means that we need to const cast some input pointers.
//
// This wrapper is intended to enhance readability, avoid repetition of the
// above warning, and assist in debugging if we do run into a "not-really-const"
// input issue.
// cf the coreesponding function in hipfft_wrapper.cc
template <typename T>
static T* ToHipfft(llvm::ArrayRef<T> array_ref) {
  if (array_ref.empty()) return nullptr;
  return const_cast<T*>(array_ref.data());
}

llvm::Expected<size_t> HipfftMakePlanMany(
    hipfftHandle handle, hipfftType type, int64_t batch,
    llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<int64_t> input_embed,
    int64_t input_stride, int64_t input_dist,
    llvm::ArrayRef<int64_t> output_embed, int64_t output_stride,
    int64_t output_dist) {
  // NOLINTNEXTLINE(google-runtime-int)
  static_assert(sizeof(int64_t) == sizeof(long long),
                "hipFFT uses long long for 64-bit values, but there is a size "
                "mismatch between long long and int64_t");
  int64_t rank = dims.size();
  if (!input_embed.empty() && input_embed.size() != rank)
    return MakeStringError("Expected input_embed to be empty or of size rank");
  if (!output_embed.empty() && output_embed.size() != rank)
    return MakeStringError("Expected output_embed to be empty or of size rank");
  size_t workspace_size_bytes;
  return hipfftMakePlanMany64(
      // NOLINTNEXTLINE(google-runtime-int)
      handle, rank, reinterpret_cast<long long*>(ToHipfft(dims)),
      // NOLINTNEXTLINE(google-runtime-int)
      reinterpret_cast<long long*>(ToHipfft(input_embed)), input_stride,
      input_dist,
      // NOLINTNEXTLINE(google-runtime-int)
      reinterpret_cast<long long*>(ToHipfft(output_embed)), output_stride,
      output_dist, type, batch, &workspace_size_bytes);
}

llvm::Expected<size_t> HipfftGetSize(hipfftHandle handle) {
  size_t work_size;
  RETURN_IF_ERROR(hipfftGetSize(handle, &work_size));
  return work_size;
}

llvm::Error HipfftDisableAutoAllocation(hipfftHandle handle) {
  return TO_ERROR(hipfftSetAutoAllocation(handle, 0));
}

llvm::Error HipfftEnableAutoAllocation(hipfftHandle handle) {
  return TO_ERROR(hipfftSetAutoAllocation(handle, 1));
}

llvm::Error HipfftSetWorkArea(hipfftHandle handle, Pointer<void> work_area) {
  return TO_ERROR(hipfftSetWorkArea(handle, ToRocm(work_area)));
}

llvm::Error HipfftExec(CurrentContext current, hipfftHandle handle,
                       Pointer<const void> input, Pointer<void> output,
                       hipfftType type, hipfftDirection direction) {
  CheckHipContext(current);
  void* input_ptr = const_cast<void*>(input.raw(Platform::ROCm));
  void* output_ptr = output.raw(Platform::ROCm);

  switch (type) {
    case HIPFFT_C2C:
      return TO_ERROR(
          hipfftExecC2C(handle, static_cast<hipfftComplex*>(input_ptr),
                        static_cast<hipfftComplex*>(output_ptr), direction));
    case HIPFFT_Z2Z:
      return TO_ERROR(hipfftExecZ2Z(
          handle, static_cast<hipfftDoubleComplex*>(input_ptr),
          static_cast<hipfftDoubleComplex*>(output_ptr), direction));
    case HIPFFT_R2C:
      return TO_ERROR(hipfftExecR2C(handle, static_cast<hipfftReal*>(input_ptr),
                                    static_cast<hipfftComplex*>(output_ptr)));
    case HIPFFT_D2Z:
      return TO_ERROR(
          hipfftExecD2Z(handle, static_cast<hipfftDoubleReal*>(input_ptr),
                        static_cast<hipfftDoubleComplex*>(output_ptr)));
    case HIPFFT_C2R:
      return TO_ERROR(hipfftExecC2R(handle,
                                    static_cast<hipfftComplex*>(input_ptr),
                                    static_cast<hipfftReal*>(output_ptr)));
    case HIPFFT_Z2D:
      return TO_ERROR(
          hipfftExecZ2D(handle, static_cast<hipfftDoubleComplex*>(input_ptr),
                        static_cast<hipfftDoubleReal*>(output_ptr)));
    default:
      return MakeStringError("invalid hipfftType: ", type);
  }
}

llvm::raw_ostream& Print(llvm::raw_ostream& os, hipfftResult_t result) {
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

Expected<hipfftType> Parse(llvm::StringRef name, hipfftType) {
  if (name == "HIPFFT_R2C") return HIPFFT_R2C;
  if (name == "HIPFFT_C2R") return HIPFFT_C2R;
  if (name == "HIPFFT_C2C") return HIPFFT_C2C;
  if (name == "HIPFFT_D2Z") return HIPFFT_D2Z;
  if (name == "HIPFFT_Z2D") return HIPFFT_Z2D;
  if (name == "HIPFFT_Z2Z") return HIPFFT_Z2Z;
  return MakeStringError("Unknown hipfftType: ", name);
}

raw_ostream& Print(raw_ostream& os, hipfftType value) {
  switch (value) {
    case HIPFFT_R2C:
      return os << "HIPFFT_R2C";
    case HIPFFT_C2R:
      return os << "HIPFFT_C2R";
    case HIPFFT_C2C:
      return os << "HIPFFT_C2C";
    case HIPFFT_D2Z:
      return os << "HIPFFT_D2Z";
    case HIPFFT_Z2D:
      return os << "HIPFFT_Z2D";
    case HIPFFT_Z2Z:
      return os << "HIPFFT_Z2Z";
    default:
      return os << llvm::formatv("hipfftType({0})", static_cast<int>(value));
  }
}

Expected<hipfftDirection> Parse(llvm::StringRef name, hipfftDirection) {
  if (name == "HIPFFT_FORWARD") return hipfftDirection(HIPFFT_FORWARD);
  if (name == "HIPFFT_INVERSE") return hipfftDirection(HIPFFT_INVERSE);
  return MakeStringError("Unknown hipfftDirection: ", name);
}

raw_ostream& Print(raw_ostream& os, hipfftDirection value) {
  switch (static_cast<int>(value)) {
    case HIPFFT_FORWARD:
      return os << "HIPFFT_FORWARD";
    case HIPFFT_INVERSE:
      return os << "HIPFFT_INVERSE";
    default:
      return os << llvm::formatv("hipfftDirection({0})",
                                 static_cast<int>(value));
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
