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

#include "llvm/Support/FormatVariadic.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

template llvm::raw_ostream& internal::operator<<(
    llvm::raw_ostream&, const ErrorData<cufftResult_t>&);

void internal::CufftHandleDeleter::operator()(cufftHandle handle) const {
  LogIfError(CufftDestroy(handle));
}

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
T* ToCufft(llvm::SmallVector<T, 3> t) {
  if (t.empty()) return nullptr;
  return const_cast<T*>(t.data());
}

struct CufftDataType {
  enum class Domain { kReal, kComplex };
  enum class Precision { kI8, kU8, kI32, kU32, kF16, kF32, kF64 };
  Domain domain;
  Precision precision;
};

template <typename IntT>
llvm::Error ValidateOptions(const CufftManyOptions<IntT>& options) {
  if (options.dims.size() != options.rank) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Mismatch between size of dims and options rank, %d vs %d",
        options.dims.size(), options.rank);
  }
  if (!options.input_embed.empty() &&
      options.input_embed.size() != options.rank) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Mismatch between size of input_embed and options rank, %d vs %d",
        options.dims.size(), options.rank);
  }
  if (!options.output_embed.empty() &&
      options.output_embed.size() != options.rank) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Mismatch between size of output_embed and options rank, %d vs %d",
        options.dims.size(), options.rank);
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

llvm::Error DataTypesCompatible(const CufftXtManyOptions& options) {
  CufftDataType input_type = FromCudaDataType(options.input_type);
  CufftDataType output_type = FromCudaDataType(options.output_type);
  CufftDataType execution_type = FromCudaDataType(options.execution_type);

  // All parameters must match precision.
  if (input_type.precision != output_type.precision ||
      input_type.precision != execution_type.precision) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Input, output, and execution types must have the same precision. "
        "input=%s, output=%s, execution=%s",
        options.input_type, options.output_type, options.execution_type);
  }

  // Execution type must be complex.
  if (execution_type.domain != CufftDataType::Domain::kComplex) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Execution type must be complex but saw %s",
                                   options.execution_type);
  }

  return llvm::Error::success();
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

llvm::Expected<OwningCufftHandle> CufftCreate() {
  cufftHandle plan;
  RETURN_IF_ERROR(cufftCreate(&plan));
  return OwningCufftHandle(plan);
}

llvm::Error CufftDestroy(cufftHandle plan) {
  return TO_ERROR(cufftDestroy(plan));
}

llvm::Error CufftSetStream(cufftHandle plan, cudaStream_t stream) {
  return TO_ERROR(cufftSetStream(plan, stream));
}

llvm::Expected<OwningCufftHandle> CufftPlan1d(int nx, cufftType type,
                                              int batch) {
  cufftHandle plan;
  RETURN_IF_ERROR(cufftPlan1d(&plan, nx, type, batch));
  return OwningCufftHandle(plan);
}

llvm::Expected<OwningCufftHandle> CufftPlan2d(int nx, int ny, cufftType type) {
  cufftHandle plan;
  RETURN_IF_ERROR(cufftPlan2d(&plan, nx, ny, type));
  return OwningCufftHandle(plan);
}

llvm::Expected<OwningCufftHandle> CufftPlan3d(int nx, int ny, int nz,
                                              cufftType type) {
  cufftHandle plan;
  RETURN_IF_ERROR(cufftPlan3d(&plan, nx, ny, nz, type));
  return OwningCufftHandle(plan);
}

llvm::Expected<OwningCufftHandle> CufftPlanMany(
    cufftType type, int batch, const CufftManyOptions<int>& options) {
  if (auto err = ValidateOptions(options)) return std::move(err);

  cufftHandle plan;
  RETURN_IF_ERROR(cufftPlanMany(
      &plan, options.rank, ToCufft(options.dims), ToCufft(options.input_embed),
      options.input_stride, options.input_dist, ToCufft(options.output_embed),
      options.output_stride, options.output_dist, type, batch));
  return OwningCufftHandle(plan);
}

llvm::Expected<size_t> CufftMakePlan1d(cufftHandle plan, int nx, cufftType type,
                                       int batch) {
  size_t work_size;
  RETURN_IF_ERROR(cufftMakePlan1d(plan, nx, type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftMakePlan2d(cufftHandle plan, int nx, int ny,
                                       cufftType type) {
  size_t work_size;
  RETURN_IF_ERROR(cufftMakePlan2d(plan, nx, ny, type, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz,
                                       cufftType type) {
  size_t work_size;
  RETURN_IF_ERROR(cufftMakePlan3d(plan, nx, ny, nz, type, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftMakePlanMany(cufftHandle plan, cufftType type,
                                         int batch,
                                         CufftManyOptions<int>& options) {
  size_t work_size;
  RETURN_IF_ERROR(cufftMakePlanMany(
      plan, options.rank, ToCufft(options.dims), ToCufft(options.input_embed),
      options.input_stride, options.input_dist, ToCufft(options.output_embed),
      options.output_stride, options.output_dist, type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftMakePlanMany(cufftHandle plan, cufftType type,
                                         int64_t batch,
                                         CufftManyOptions<int64_t>& options) {
  // NOLINTNEXTLINE(google-runtime-int)
  static_assert(sizeof(int64_t) == sizeof(long long),
                "cuFFT uses long long for 64-bit values, but there is a size "
                "mismatch between long long and int64_t");
  size_t work_size;
  RETURN_IF_ERROR(cufftMakePlanMany64(
      plan, options.rank, reinterpret_cast<long long*>(ToCufft(options.dims)),
      reinterpret_cast<long long*>(ToCufft(options.input_embed)),
      static_cast<long long>(options.input_stride),
      static_cast<long long>(options.input_dist),
      reinterpret_cast<long long*>(ToCufft(options.output_embed)),
      static_cast<long long>(options.output_stride),
      static_cast<long long>(options.output_dist), type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftXtMakePlanMany(cufftHandle plan, int64_t batch,
                                           CufftXtManyOptions& options) {
  if (auto err = DataTypesCompatible(options)) return std::move(err);
  size_t work_size;
  RETURN_IF_ERROR(cufftXtMakePlanMany(
      plan, options.rank, reinterpret_cast<long long*>(ToCufft(options.dims)),
      reinterpret_cast<long long*>(ToCufft(options.input_embed)),
      static_cast<long long>(options.input_stride),
      static_cast<long long>(options.input_dist), options.input_type,
      reinterpret_cast<long long*>(ToCufft(options.output_embed)),
      static_cast<long long>(options.output_stride),
      static_cast<long long>(options.output_dist), options.output_type, batch,
      &work_size, options.execution_type));
  return work_size;
}

llvm::Expected<size_t> CufftEstimate1d(int nx, cufftType type, int batch) {
  size_t work_size;
  RETURN_IF_ERROR(cufftEstimate1d(nx, type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftEstimate2d(int nx, int ny, cufftType type) {
  size_t work_size;
  RETURN_IF_ERROR(cufftEstimate2d(nx, ny, type, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftEstimate3d(int nx, int ny, int nz, cufftType type) {
  size_t work_size;
  RETURN_IF_ERROR(cufftEstimate3d(nx, ny, nz, type, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftEstimateMany(cufftType type, int batch,
                                         CufftManyOptions<int>& options) {
  size_t work_size;
  RETURN_IF_ERROR(cufftEstimateMany(
      options.rank, ToCufft(options.dims), ToCufft(options.input_embed),
      options.input_stride, options.input_dist, ToCufft(options.output_embed),
      options.output_stride, options.output_dist, type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftGetSize1d(cufftHandle plan, int nx, cufftType type,
                                      int batch) {
  size_t work_size;
  RETURN_IF_ERROR(cufftGetSize1d(plan, nx, type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftGetSize2d(cufftHandle plan, int nx, int ny,
                                      cufftType type) {
  size_t work_size;
  RETURN_IF_ERROR(cufftGetSize2d(plan, nx, ny, type, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftGetSize3d(cufftHandle plan, int nx, int ny, int nz,
                                      cufftType type) {
  size_t work_size;
  RETURN_IF_ERROR(cufftGetSize3d(plan, nx, ny, nz, type, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftGetSizeMany(cufftHandle plan, cufftType type,
                                        int batch,
                                        CufftManyOptions<int>& options) {
  size_t work_size;
  RETURN_IF_ERROR(cufftGetSizeMany(
      plan, options.rank, ToCufft(options.dims), ToCufft(options.input_embed),
      options.input_stride, options.input_dist, ToCufft(options.output_embed),
      options.output_stride, options.output_dist, type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftGetSizeMany(cufftHandle plan, cufftType type,
                                        int batch,
                                        CufftManyOptions<int64_t>& options) {
  size_t work_size;
  RETURN_IF_ERROR(cufftGetSizeMany64(
      plan, options.rank, reinterpret_cast<long long*>(ToCufft(options.dims)),
      reinterpret_cast<long long*>(ToCufft(options.input_embed)),
      static_cast<long long>(options.input_stride),
      static_cast<long long>(options.input_dist),
      reinterpret_cast<long long*>(ToCufft(options.output_embed)),
      static_cast<long long>(options.output_stride),
      static_cast<long long>(options.output_dist), type, batch, &work_size));
  return work_size;
}

llvm::Expected<size_t> CufftXtGetSizeMany(cufftHandle plan, cufftType type,
                                          int batch,
                                          CufftXtManyOptions& options) {
  size_t work_size;
  RETURN_IF_ERROR(cufftXtGetSizeMany(
      plan, options.rank, reinterpret_cast<long long*>(ToCufft(options.dims)),
      reinterpret_cast<long long*>(ToCufft(options.input_embed)),
      static_cast<long long>(options.input_stride),
      static_cast<long long>(options.input_dist), options.input_type,
      reinterpret_cast<long long*>(ToCufft(options.output_embed)),
      static_cast<long long>(options.output_stride),
      static_cast<long long>(options.output_dist), options.output_type, batch,
      &work_size, options.execution_type));
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

llvm::Error CufftXtSetWorkAreaPolicy(cufftHandle plan,
                                     cufftXtWorkAreaPolicy policy) {
  return TO_ERROR(cufftXtSetWorkAreaPolicy(plan, policy, nullptr));
}

static int ToCuda(FftDirection direction) {
  int fft_dir = 0;
  switch (direction) {
  case FftDirection::kForward: fft_dir = CUFFT_FORWARD; break;
  case FftDirection::kInverse:  fft_dir = CUFFT_INVERSE; break;
  }
  return fft_dir;
}

llvm::Error CufftExecC2C(cufftHandle plan, cufftComplex* input_data,
                         cufftComplex* output_data, FftDirection direction) {
  return TO_ERROR(
      cufftExecC2C(plan, input_data, output_data, ToCuda(direction)));
}

llvm::Error CufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* input_data,
                         cufftDoubleComplex* output_data,
                         FftDirection direction) {
  return TO_ERROR(
      cufftExecZ2Z(plan, input_data, output_data, ToCuda(direction)));
}

llvm::Error CufftExecR2C(cufftHandle plan, cufftReal* input_data,
                         cufftComplex* output_data) {
  return TO_ERROR(cufftExecR2C(plan, input_data, output_data));
}

llvm::Error CufftExecD2C(cufftHandle plan, cufftDoubleReal* input_data,
                         cufftDoubleComplex* output_data) {
  return TO_ERROR(cufftExecD2Z(plan, input_data, output_data));
}

llvm::Error CufftExecC2R(cufftHandle plan, cufftComplex* input_data,
                         cufftReal* output_data) {
  return TO_ERROR(cufftExecC2R(plan, input_data, output_data));
}

llvm::Error CufftExecZ2D(cufftHandle plan, cufftDoubleComplex* input_data,
                         cufftDoubleReal* output_data) {
  return TO_ERROR(cufftExecZ2D(plan, input_data, output_data));
}

llvm::Error CufftXtExec(cufftHandle plan, void* input, void* output,
                        FftDirection direction) {
  return TO_ERROR(
      cufftXtExec(plan, input, output, ToCuda(direction)));
}

llvm::Error CufftXtMemcpy(cufftHandle plan, void* dst, void* src,
                          cufftXtCopyType type) {
  return TO_ERROR(cufftXtMemcpy(plan, dst, src, type));
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
