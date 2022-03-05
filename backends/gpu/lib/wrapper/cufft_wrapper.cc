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

// Thin wrapper around the cuFFT API that adapts the library for TFRT
// conventions.
#include "tfrt/gpu/wrapper/cufft_wrapper.h"

#include <cstddef>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/gpu/wrapper/fft_wrapper.h"
#include "tfrt/support/error_util.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::Expected<LibraryVersion> CufftGetVersion() {
  LibraryVersion version;
  RETURN_IF_ERROR(
      cufftGetProperty(libraryPropertyType::MAJOR_VERSION, &version.major));
  RETURN_IF_ERROR(
      cufftGetProperty(libraryPropertyType::MINOR_VERSION, &version.minor));
  RETURN_IF_ERROR(
      cufftGetProperty(libraryPropertyType::PATCH_LEVEL, &version.patch));
  return version;
}

llvm::Expected<OwningFftHandle> CufftCreate(CurrentContext current) {
  CheckCudaContext(current);
  cufftHandle handle;
  RETURN_IF_ERROR(cufftCreate(&handle));
  return OwningFftHandle(handle);
}

llvm::Error CufftDestroy(cufftHandle handle) {
  auto fft_error = TO_ERROR(cufftDestroy(handle));
  // Restores the current context after cufftDestroy has fiddled with it.
  auto ctx_error = TO_ERROR(cuCtxSetCurrent(kContextTls.cuda_ctx));
  return llvm::joinErrors(std::move(fft_error), std::move(ctx_error));
}

llvm::Error CufftSetStream(cufftHandle handle, cudaStream_t stream) {
  return TO_ERROR(cufftSetStream(handle, stream));
}

// The CuFFT library does not enforce const correctness. We try to improve
// usability by providing some level of const correctness in the wrapper, but
// this means that we need to const cast some input pointers. Since we dont'
// have the cuFFT sources, we can't guarantee that there isn't some bug/edge
// case where an input parameter is modified, but we can reasonably assume that
// parameters documented as input are effectively const.
//
// This wrapper is intended to enhance readability, avoid repetition of the
// above warning, and assist in debugging if we do run into a "not-really-const"
// input issue.
template <typename T>
static T* ToCufft(ArrayRef<T> array_ref) {
  if (array_ref.empty()) return nullptr;
  return const_cast<T*>(array_ref.data());
}

llvm::Expected<size_t> CufftMakePlanMany(
    cufftHandle handle, cufftType type, int64_t batch, ArrayRef<int64_t> dims,
    ArrayRef<int64_t> input_embed, int64_t input_stride, int64_t input_dist,
    ArrayRef<int64_t> output_embed, int64_t output_stride,
    int64_t output_dist) {
  // NOLINTNEXTLINE(google-runtime-int)
  static_assert(sizeof(int64_t) == sizeof(long long),
                "cuFFT uses long long for 64-bit values, but there is a size "
                "mismatch between long long and int64_t");
  int rank = dims.size();
  if (!input_embed.empty() && input_embed.size() != rank)
    return MakeStringError("Expected input_embed to be empty or of size rank");
  if (!output_embed.empty() && output_embed.size() != rank)
    return MakeStringError("Expected output_embed to be empty or of size rank");
  size_t workspace_size_bytes;
  RETURN_IF_ERROR(cufftMakePlanMany64(
      handle, rank, reinterpret_cast<long long*>(ToCufft(dims)),
      reinterpret_cast<long long*>(ToCufft(input_embed)), input_stride,
      input_dist, reinterpret_cast<long long*>(ToCufft(output_embed)),
      output_stride, output_dist, type, batch, &workspace_size_bytes));
  return workspace_size_bytes;
}

llvm::Expected<size_t> CufftEstimateMany(
    cufftType type, int batch, int rank, ArrayRef<int> dims,
    ArrayRef<int> input_embed, int input_stride, ArrayRef<int> output_embed,
    int output_stride, int input_dist, int output_dist) {
  size_t work_size;
  RETURN_IF_ERROR(cufftEstimateMany(rank, ToCufft(dims), ToCufft(input_embed),
                                    input_stride, input_dist,
                                    ToCufft(output_embed), output_stride,
                                    output_dist, type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftGetSizeMany(
    cufftHandle handle, cufftType type, int batch, int rank,
    ArrayRef<int64_t> dims, ArrayRef<int64_t> input_embed, int64_t input_stride,
    ArrayRef<int64_t> output_embed, int64_t output_stride, int64_t input_dist,
    int64_t output_dist) {
  size_t work_size;
  RETURN_IF_ERROR(cufftGetSizeMany64(
      handle, rank, reinterpret_cast<long long*>(ToCufft(dims)),
      reinterpret_cast<long long*>(ToCufft(input_embed)), input_stride,
      input_dist, reinterpret_cast<long long*>(ToCufft(output_embed)),
      output_stride, output_dist, type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftGetSize(cufftHandle handle) {
  size_t work_size;
  RETURN_IF_ERROR(cufftGetSize(handle, &work_size));
  return work_size;
}

llvm::Error CufftDisableAutoAllocation(cufftHandle handle) {
  return TO_ERROR(cufftSetAutoAllocation(handle, 0));
}

llvm::Error CufftEnableAutoAllocation(cufftHandle handle) {
  return TO_ERROR(cufftSetAutoAllocation(handle, 1));
}

llvm::Error CufftSetWorkArea(cufftHandle handle, Pointer<void> work_area) {
  return TO_ERROR(cufftSetWorkArea(handle, ToCuda(work_area)));
}

llvm::Error CufftExec(CurrentContext current, cufftHandle handle,
                      Pointer<const void> input, Pointer<void> output,
                      cufftType type, cufftDirection direction) {
  CheckCudaContext(current);

  void* input_ptr = const_cast<void*>(input.raw(Platform::CUDA));
  void* output_ptr = output.raw(Platform::CUDA);

  switch (type) {
    case CUFFT_C2C:
      return TO_ERROR(
          cufftExecC2C(handle, static_cast<cufftComplex*>(input_ptr),
                       static_cast<cufftComplex*>(output_ptr), direction));
    case CUFFT_Z2Z:
      return TO_ERROR(cufftExecZ2Z(
          handle, static_cast<cufftDoubleComplex*>(input_ptr),
          static_cast<cufftDoubleComplex*>(output_ptr), direction));
    case CUFFT_R2C:
      return TO_ERROR(cufftExecR2C(handle, static_cast<cufftReal*>(input_ptr),
                                   static_cast<cufftComplex*>(output_ptr)));
    case CUFFT_D2Z:
      return TO_ERROR(
          cufftExecD2Z(handle, static_cast<cufftDoubleReal*>(input_ptr),
                       static_cast<cufftDoubleComplex*>(output_ptr)));
    case CUFFT_C2R:
      return TO_ERROR(cufftExecC2R(handle,
                                   static_cast<cufftComplex*>(input_ptr),
                                   static_cast<cufftReal*>(output_ptr)));
    case CUFFT_Z2D:
      return TO_ERROR(cufftExecZ2D(handle,
                                   static_cast<cufftDoubleComplex*>(input_ptr),
                                   static_cast<cufftDoubleReal*>(output_ptr)));
    default:
      return MakeStringError("invalid cufftType: ", type);
  }
}

llvm::raw_ostream& Print(llvm::raw_ostream& os, cufftResult result) {
  switch (result) {
    case CUFFT_SUCCESS:
      return os << "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
      return os << "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
      return os << "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
      return os << "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return os << "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
      return os << "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
      return os << "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
      return os << "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
      return os << "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
      return os << "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return os << "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
      return os << "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
      return os << "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
      return os << "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
      return os << "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR:
      return os << "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED:
      return os << "CUFFT_NOT_SUPPORTED";
    default:
      return os << llvm::formatv("cufftResult({0})", static_cast<int>(result));
  }
}

Expected<cufftType> Parse(llvm::StringRef name, cufftType) {
  if (name == "CUFFT_R2C") return CUFFT_R2C;
  if (name == "CUFFT_C2R") return CUFFT_C2R;
  if (name == "CUFFT_C2C") return CUFFT_C2C;
  if (name == "CUFFT_D2Z") return CUFFT_D2Z;
  if (name == "CUFFT_Z2D") return CUFFT_Z2D;
  if (name == "CUFFT_Z2Z") return CUFFT_Z2Z;
  return MakeStringError("Unknown cufftType: ", name);
}

raw_ostream& Print(raw_ostream& os, cufftType value) {
  switch (value) {
    case CUFFT_R2C:
      return os << "CUFFT_R2C";
    case CUFFT_C2R:
      return os << "CUFFT_C2R";
    case CUFFT_C2C:
      return os << "CUFFT_C2C";
    case CUFFT_D2Z:
      return os << "CUFFT_D2Z";
    case CUFFT_Z2D:
      return os << "CUFFT_Z2D";
    case CUFFT_Z2Z:
      return os << "CUFFT_Z2Z";
    default:
      return os << llvm::formatv("cufftType({0})", static_cast<int>(value));
  }
}

Expected<cufftDirection> Parse(llvm::StringRef name, cufftDirection) {
  if (name == "CUFFT_FORWARD") return cufftDirection(CUFFT_FORWARD);
  if (name == "CUFFT_INVERSE") return cufftDirection(CUFFT_INVERSE);
  return MakeStringError("Unknown cufftDirection: ", name);
}

raw_ostream& Print(raw_ostream& os, cufftDirection value) {
  switch (static_cast<int>(value)) {
    case CUFFT_FORWARD:
      return os << "CUFFT_FORWARD";
    case CUFFT_INVERSE:
      return os << "CUFFT_INVERSE";
    default:
      return os << llvm::formatv("cufftDirection({0})",
                                 static_cast<int>(value));
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
