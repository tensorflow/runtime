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
#include "tfrt/gpu/wrapper/fft_wrapper.h"
#include "tfrt/support/error_util.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

namespace {

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
T* ToCufft(llvm::ArrayRef<T> t) {
  if (t.empty()) return nullptr;
  return const_cast<T*>(t.data());
}

struct CufftDataType {
  enum class Domain { kReal, kComplex };
  enum class Precision { kI8, kU8, kI32, kU32, kF16, kF32, kF64 };
  Domain domain;
  Precision precision;
};

llvm::Error ValidateOptions(int rank, llvm::ArrayRef<int64_t> dims,
                            llvm::ArrayRef<int64_t> input_embed,
                            llvm::ArrayRef<int64_t> output_embed) {
  if (dims.size() != rank) {
    return MakeStringError("Mismatch between size of dims and rank, %d vs %d",
                           dims.size(), rank);
  }
  if (!input_embed.empty() && input_embed.size() != rank) {
    return MakeStringError(
        "Mismatch between size of input_embed and rank, %d vs %d", dims.size(),
        rank);
  }
  if (!output_embed.empty() && output_embed.size() != rank) {
    return MakeStringError(
        "Mismatch between size of output_embed and rank, %d vs %d", dims.size(),
        rank);
  }
  return llvm::Error::success();
}

static CufftDataType FromCudaDataType(cudaDataType type) {
  using D = CufftDataType::Domain;
  using P = CufftDataType::Precision;
  CufftDataType data_type;

  switch (type) {
    case CUDA_R_16F:
      data_type.domain = D::kReal;
      data_type.precision = P::kF16;
      return data_type;
    case CUDA_C_16F:
      data_type.domain = D::kComplex;
      data_type.precision = P::kF16;
      return data_type;
    case CUDA_R_32F:
      data_type.domain = D::kReal;
      data_type.precision = P::kF32;
      return data_type;
    case CUDA_C_32F:
      data_type.domain = D::kComplex;
      data_type.precision = P::kF32;
      return data_type;
    case CUDA_R_64F:
      data_type.domain = D::kReal;
      data_type.precision = P::kF64;
      return data_type;
    case CUDA_C_64F:
      data_type.domain = D::kComplex;
      data_type.precision = P::kF64;
      return data_type;
    case CUDA_R_8I:
      data_type.domain = D::kReal;
      data_type.precision = P::kI8;
      return data_type;
    case CUDA_C_8I:
      data_type.domain = D::kComplex;
      data_type.precision = P::kI8;
      return data_type;
    case CUDA_R_8U:
      data_type.domain = D::kReal;
      data_type.precision = P::kU8;
      return data_type;
    case CUDA_C_8U:
      data_type.domain = D::kComplex;
      data_type.precision = P::kU8;
      return data_type;
    case CUDA_R_32I:
      data_type.domain = D::kReal;
      data_type.precision = P::kI32;
      return data_type;
    case CUDA_C_32I:
      data_type.domain = D::kComplex;
      data_type.precision = P::kI32;
      return data_type;
    case CUDA_R_32U:
      data_type.domain = D::kReal;
      data_type.precision = P::kU32;
      return data_type;
    case CUDA_C_32U:
      data_type.domain = D::kComplex;
      data_type.precision = P::kU32;
      return data_type;
    default:
      llvm_unreachable(StrCat("Unrecognized cudaDataType: ", type).c_str());
  }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cudaDataType type) {
  switch (type) {
    case CUDA_C_16F:
      return os << "CUDA_C_16F";
    case CUDA_R_32F:
      return os << "CUDA_R_32F";
    case CUDA_C_32F:
      return os << "CUDA_C_32F";
    case CUDA_R_64F:
      return os << "CUDA_R_64F";
    case CUDA_C_64F:
      return os << "CUDA_C_64F";
    case CUDA_R_8I:
      return os << "CUDA_R_8I";
    case CUDA_C_8I:
      return os << "CUDA_C_8I";
    case CUDA_R_8U:
      return os << "CUDA_R_8U";
    case CUDA_C_8U:
      return os << "CUDA_C_8U";
#if CUDA_VERSION >= 11000
    case CUDA_R_16F:
      return os << "CUDA_R_16F";
    case CUDA_R_16BF:
      return os << "CUDA_R_16BF";
    case CUDA_C_16BF:
      return os << "CUDA_C_16BF";
    case CUDA_R_4I:
      return os << "xCUDA_R_4I";
    case CUDA_C_4I:
      return os << "CUDA_C_4I";
    case CUDA_R_4U:
      return os << "CUDA_R_4U";
    case CUDA_C_4U:
      return os << "CUDA_C_4U";
    case CUDA_R_16I:
      return os << "CUDA_R_16I";
    case CUDA_C_16I:
      return os << "CUDA_C_16I";
    case CUDA_R_16U:
      return os << "CUDA_R_16U";
    case CUDA_C_16U:
      return os << "CUDA_C_16U";
    case CUDA_R_32I:
      return os << "CUDA_R_32I";
    case CUDA_C_32I:
      return os << "CUDA_C_32I";
    case CUDA_R_32U:
      return os << "CUDA_R_32U";
    case CUDA_C_32U:
      return os << "CUDA_C_32U";
    case CUDA_R_64I:
      return os << "CUDA_R_64I";
    case CUDA_C_64I:
      return os << "CUDA_C_64I";
    case CUDA_R_64U:
      return os << "CUDA_R_64U";
    case CUDA_C_64U:
      return os << "CUDA_C_64U";
#endif
    default:
      return os << llvm::formatv("cudaDataType({0})", static_cast<int>(type));
  }
}

}  // namespace

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cufftResult result) {
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

llvm::Expected<OwningFftHandle> CufftCreate() {
  cufftHandle plan;
  RETURN_IF_ERROR(cufftCreate(&plan));
  return OwningFftHandle(plan);
}

llvm::Error CufftDestroy(cufftHandle plan) {
  return TO_ERROR(cufftDestroy(plan));
}

llvm::Error CufftSetStream(cufftHandle plan, cudaStream_t stream) {
  return TO_ERROR(cufftSetStream(plan, stream));
}

llvm::Expected<size_t> CufftMakePlanMany(
    cufftHandle plan, cufftType type, int64_t batch, int rank,
    llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<int64_t> input_embed,
    int64_t input_stride, llvm::ArrayRef<int64_t> output_embed,
    int64_t output_stride, int64_t input_dist, int64_t output_dist) {
  // NOLINTNEXTLINE(google-runtime-int)
  static_assert(sizeof(int64_t) == sizeof(long long),
                "cuFFT uses long long for 64-bit values, but there is a size "
                "mismatch between long long and int64_t");
  size_t work_size;
  RETURN_IF_ERROR(cufftMakePlanMany64(
      plan, rank, reinterpret_cast<long long*>(ToCufft(dims)),
      reinterpret_cast<long long*>(ToCufft(input_embed)), input_stride,
      input_dist, reinterpret_cast<long long*>(ToCufft(output_embed)),
      output_stride, output_dist, type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftEstimateMany(cufftType type, int batch, int rank,
                                         llvm::ArrayRef<int> dims,
                                         llvm::ArrayRef<int> input_embed,
                                         int input_stride,
                                         llvm::ArrayRef<int> output_embed,
                                         int output_stride, int input_dist,
                                         int output_dist) {
  size_t work_size;
  RETURN_IF_ERROR(cufftEstimateMany(rank, ToCufft(dims), ToCufft(input_embed),
                                    input_stride, input_dist,
                                    ToCufft(output_embed), output_stride,
                                    output_dist, type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftGetSizeMany(
    cufftHandle plan, cufftType type, int batch, int rank,
    llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<int64_t> input_embed,
    int64_t input_stride, llvm::ArrayRef<int64_t> output_embed,
    int64_t output_stride, int64_t input_dist, int64_t output_dist) {
  size_t work_size;
  RETURN_IF_ERROR(cufftGetSizeMany64(
      plan, rank, reinterpret_cast<long long*>(ToCufft(dims)),
      reinterpret_cast<long long*>(ToCufft(input_embed)), input_stride,
      input_dist, reinterpret_cast<long long*>(ToCufft(output_embed)),
      output_stride, output_dist, type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftGetSize(cufftHandle plan) {
  size_t work_size;
  RETURN_IF_ERROR(cufftGetSize(plan, &work_size));
  return work_size;
}

llvm::Error CufftDisableAutoAllocation(cufftHandle plan) {
  return TO_ERROR(cufftSetAutoAllocation(plan, 0));
}

llvm::Error CufftEnableAutoAllocation(cufftHandle plan) {
  return TO_ERROR(cufftSetAutoAllocation(plan, 1));
}

llvm::Error CufftSetWorkArea(cufftHandle plan, Pointer<void> work_area) {
  return TO_ERROR(cufftSetWorkArea(plan, ToCuda(work_area)));
}

static int ToCuda(FftDirection direction) {
  switch (direction) {
    case FftDirection::kForward:
      return CUFFT_FORWARD;
    case FftDirection::kInverse:
      return CUFFT_INVERSE;
    default:
      return 0;  // 0 is an invalid value.
  }
}

llvm::Error CufftExecC2C(cufftHandle plan, Pointer<cufftComplex> input_data,
                         Pointer<cufftComplex> output_data,
                         FftDirection direction) {
  return TO_ERROR(cufftExecC2C(plan, input_data.raw(Platform::CUDA),
                               output_data.raw(Platform::CUDA),
                               ToCuda(direction)));
}

llvm::Error CufftExecZ2Z(cufftHandle plan,
                         Pointer<cufftDoubleComplex> input_data,
                         Pointer<cufftDoubleComplex> output_data,
                         FftDirection direction) {
  return TO_ERROR(cufftExecZ2Z(plan, input_data.raw(Platform::CUDA),
                               output_data.raw(Platform::CUDA),
                               ToCuda(direction)));
}

llvm::Error CufftExecR2C(cufftHandle plan, Pointer<cufftReal> input_data,
                         Pointer<cufftComplex> output_data) {
  return TO_ERROR(cufftExecR2C(plan, input_data.raw(Platform::CUDA),
                               output_data.raw(Platform::CUDA)));
}

llvm::Error CufftExecD2Z(cufftHandle plan, Pointer<cufftDoubleReal> input_data,
                         Pointer<cufftDoubleComplex> output_data) {
  return TO_ERROR(cufftExecD2Z(plan, input_data.raw(Platform::CUDA),
                               output_data.raw(Platform::CUDA)));
}

llvm::Error CufftExecC2R(cufftHandle plan, Pointer<cufftComplex> input_data,
                         Pointer<cufftReal> output_data) {
  return TO_ERROR(cufftExecC2R(plan, input_data.raw(Platform::CUDA),
                               output_data.raw(Platform::CUDA)));
}

llvm::Error CufftExecZ2D(cufftHandle plan,
                         Pointer<cufftDoubleComplex> input_data,
                         Pointer<cufftDoubleReal> output_data) {
  return TO_ERROR(cufftExecZ2D(plan, input_data.raw(Platform::CUDA),
                               output_data.raw(Platform::CUDA)));
}

llvm::Error CufftExec(cufftHandle plan, Pointer<void> raw_input,
                      Pointer<void> raw_output, FftType type) {
  switch (type) {
    case FftType::kC2CForward:
      return CufftExecC2C(plan, static_cast<Pointer<cufftComplex>>(raw_input),
                          static_cast<Pointer<cufftComplex>>(raw_output),
                          FftDirection::kForward);
    case FftType::kC2CInverse:
      return CufftExecC2C(plan, static_cast<Pointer<cufftComplex>>(raw_input),
                          static_cast<Pointer<cufftComplex>>(raw_output),
                          FftDirection::kInverse);
    case FftType::kZ2ZForward:
      return CufftExecZ2Z(plan,
                          static_cast<Pointer<cufftDoubleComplex>>(raw_input),
                          static_cast<Pointer<cufftDoubleComplex>>(raw_output),
                          FftDirection::kForward);
    case FftType::kZ2ZInverse:
      return CufftExecZ2Z(plan,
                          static_cast<Pointer<cufftDoubleComplex>>(raw_input),
                          static_cast<Pointer<cufftDoubleComplex>>(raw_output),
                          FftDirection::kInverse);
    case FftType::kR2C:
      return CufftExecR2C(plan, static_cast<Pointer<cufftReal>>(raw_input),
                          static_cast<Pointer<cufftComplex>>(raw_output));
    case FftType::kD2Z:
      return CufftExecD2Z(plan,
                          static_cast<Pointer<cufftDoubleReal>>(raw_input),
                          static_cast<Pointer<cufftDoubleComplex>>(raw_output));
    case FftType::kC2R:
      return CufftExecC2R(plan, static_cast<Pointer<cufftComplex>>(raw_input),
                          static_cast<Pointer<cufftReal>>(raw_output));
    case FftType::kZ2D:
      return CufftExecZ2D(plan,
                          static_cast<Pointer<cufftDoubleComplex>>(raw_input),
                          static_cast<Pointer<cufftDoubleReal>>(raw_output));
    default:
      return llvm::createStringError(std::errc::invalid_argument,
                                     "invalid FFT type");
  }
}

llvm::Expected<cufftType> FftTypeToCufftType(FftType type) {
  switch (type) {
    case FftType::kZ2ZForward:
    case FftType::kZ2ZInverse:
      return CUFFT_Z2Z;
    case FftType::kR2C:
      return CUFFT_R2C;
    case FftType::kC2R:
      return CUFFT_C2R;
    case FftType::kC2CForward:
    case FftType::kC2CInverse:
      return CUFFT_C2C;
    case FftType::kD2Z:
      return CUFFT_D2Z;
    case FftType::kZ2D:
      return CUFFT_Z2D;
    default:
      return llvm::createStringError(std::errc::invalid_argument,
                                     "invalid FFT type");
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
