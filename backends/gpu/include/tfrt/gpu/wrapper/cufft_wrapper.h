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
#ifndef TFRT_GPU_WRAPPER_CUFFT_WRAPPER_H_
#define TFRT_GPU_WRAPPER_CUFFT_WRAPPER_H_

#include "cufft.h"    // from @cuda_headers
#include "cufftXt.h"  // from @cuda_headers
#include "tfrt/gpu/wrapper/fft_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cufftResult result);

// Wraps a cufftHandle so that it can be used in a std::unique_ptr.
class CufftHandle {
 public:
  CufftHandle() = default;
  CufftHandle(std::nullptr_t) : CufftHandle() {}
  CufftHandle(cufftHandle handle) : handle_(handle) {}

  operator cufftHandle() const { return handle_; }
  operator bool() const { return handle_; }

  bool operator!=(std::nullptr_t) const { return *this; }
  CufftHandle& operator=(std::nullptr_t) { return *this = CufftHandle(); }

 private:
  cufftHandle handle_ = 0;
};

namespace internal {
// Helper to wrap resources and memory into RAII types.
struct CufftHandleDeleter {
  using pointer = CufftHandle;
  void operator()(cufftHandle handle) const;
};
}  // namespace internal

// RAII wrappers for resources. Instances own the underlying resource.
//
// They are implemented as std::unique_ptrs with custom deleters.
//
// Use get() and release() to access the non-owning handle, please use with
// appropriate care.
using OwningCufftHandle =
    internal::OwningResource<internal::CufftHandleDeleter>;

// CufftMany creates an FFT plan configuration for ranks 1, 2, or 3.
template <typename IntT>
struct CufftManyOptions {
  static_assert(std::is_integral<IntT>::value,
                "CufftManyOptions dimension must be an integral type.");
  using ValueType = IntT;

  // Dimension of the FFT plan. This rank should match the size of all ArrayRefs
  // in the options.
  int rank;

  llvm::SmallVector<ValueType, 3> dims;

  // Optional description for the layout for more complex FFT plans.

  // Storage dimensions of the input data. input_stride is ignored if
  // input_embed is empty.
  llvm::SmallVector<ValueType, 3> input_embed;
  ValueType input_stride;

  // Storage dimensions of the output data. output_stride is ignored if
  // output_embed is nullptr.
  llvm::SmallVector<ValueType, 3> output_embed;
  ValueType output_stride;

  // Distance between the first element of two consecutive signals in a batch.
  ValueType input_dist;

  // Distance between the first element of two consecutive signals in a batch.
  ValueType output_dist;
};

// Cufft can only configure plans up to 3D.
constexpr bool IsValidFftRank(int rank) {
  return rank == 1 || rank == 2 || rank == 3;
}

llvm::Expected<LibraryVersion> CufftGetVersion();

// Creates an opaque handle and allocates small data for the plan. Use
// CufftMakePlan* to do the plan generation. (See
// https://docs.nvidia.com/cuda/cufft/index.html#plan-extensible).
llvm::Expected<OwningCufftHandle> CufftCreate();

// Frees all GPU resources associated with the plan and destroys internal data
// structures.
llvm::Error CufftDestroy(cufftHandle plan);

// Sets the stream for execution of cuFFT functions. Note that these functions
// may consist of many kernel invocations.
llvm::Error CufftSetStream(cufftHandle plan, cudaStream_t stream);

// Creates FFT plans for the specific dimension, window dimensions, transform
// type. (See https://docs.nvidia.com/cuda/cufft/index.html#plan-basic)
llvm::Expected<OwningCufftHandle> CufftPlan1d(int nx, cufftType type,
                                              int batch);
llvm::Expected<OwningCufftHandle> CufftPlan2d(int nx, int ny, cufftType type);
llvm::Expected<OwningCufftHandle> CufftPlan3d(int nx, int ny, int nz,
                                              cufftType type);

// Only 'int' is supported for options packed to
// internal::CufftManyOptions<int> type.
llvm::Expected<OwningCufftHandle> CufftPlanMany(
    cufftType type, int batch, const CufftManyOptions<int>& options);

// Following a call to CufftCreate, makes a plan for the specified signal size
// and type. Work size contains the size(s) in bytes of the work areas for each
// GPU used.
llvm::Expected<size_t> CufftMakePlan1d(cufftHandle plan, int nx, cufftType type,
                                       int batch);
llvm::Expected<size_t> CufftMakePlan2d(cufftHandle plan, int nx, int ny,
                                       cufftType type);
llvm::Expected<size_t> CufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz,
                                       cufftType type);

llvm::Expected<size_t> CufftMakePlanMany(cufftHandle plan, cufftType type,
                                         int batch,
                                         CufftManyOptions<int>& options);

llvm::Expected<size_t> CufftMakePlanMany(cufftHandle plan, cufftType type,
                                         int64_t batch,
                                         CufftManyOptions<int64_t>& options);

struct CufftXtManyOptions : CufftManyOptions<int64_t> {
  cudaDataType input_type;
  cudaDataType output_type;
  cudaDataType execution_type;
};

llvm::Expected<size_t> CufftXtMakePlanMany(cufftHandle plan, int64_t batch,
                                           CufftXtManyOptions& options);

// Functions for getting estimated size of work area for temporary results
// during plan execution.
// See https://docs.nvidia.com/cuda/cufft/index.html#work-estimate.

llvm::Expected<size_t> CufftEstimate1d(int nx, cufftType type, int batch);
llvm::Expected<size_t> CufftEstimate2d(int nx, int ny, cufftType type);
llvm::Expected<size_t> CufftEstimate3d(int nx, int ny, int nz, cufftType type);

llvm::Expected<size_t> CufftEstimateMany(cufftType type, int batch,
                                         CufftManyOptions<int>& options);

// GetSize provides a more refined estimate than the Estimate functions, taking
// into account plan details. work_size must contain one element for each GPU in
// use.
llvm::Expected<size_t> CufftGetSize1d(cufftHandle plan, int nx, cufftType type,
                                      int batch);
llvm::Expected<size_t> CufftGetSize2d(cufftHandle plan, int nx, int ny,
                                      cufftType type);
llvm::Expected<size_t> CufftGetSize3d(cufftHandle plan, int nx, int ny, int nz,
                                      cufftType type);

llvm::Expected<size_t> CufftGetSizeMany(cufftHandle plan, cufftType type,
                                        int batch,
                                        CufftManyOptions<int>& option);

llvm::Expected<size_t> CufftGetSizeMany(cufftHandle plan, cufftType type,
                                        int batch,
                                        CufftManyOptions<int64_t>& options);

llvm::Expected<size_t> CufftXtGetSizeMany(cufftHandle plan, cufftType type,
                                          int batch,
                                          CufftXtManyOptions& options);

llvm::Expected<size_t> CufftGetSize(cufftHandle plan);

// Lower level memory management support.
// See https://docs.nvidia.com/cuda/cufft/index.html#unique_772799016.

llvm::Error CufftDisableAutoAllocation(cufftHandle plan);
llvm::Error CufftEnableAutoAllocation(cufftHandle plan);

llvm::Error CufftSetWorkArea(cufftHandle plan, Pointer<void> work_area);

// TODO(gkg): The nvidia API currently supports an unused work_size
// parameter. Expose this flag once there is functionality there.
llvm::Error CufftXtSetWorkAreaPolicy(cufftHandle plan,
                                     cufftXtWorkAreaPolicy policy);

llvm::Error CufftExecC2C(cufftHandle plan, cufftComplex* input_data,
                         cufftComplex* output_data, FftDirection direction);
llvm::Error CufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* input_data,
                         cufftDoubleComplex* output_data,
                         FftDirection direction);

llvm::Error CufftExecR2C(cufftHandle plan, cufftReal* input_data,
                         cufftComplex* output_data);
llvm::Error CufftExecD2Z(cufftHandle plan, cufftDoubleReal* input_data,
                         cufftDoubleComplex* output_data);

llvm::Error CufftExecC2R(cufftHandle plan, cufftComplex* input_data,
                         cufftReal* output_data);
llvm::Error CufftExecZ2D(cufftHandle plan, cufftDoubleComplex* input_data,
                         cufftDoubleReal* output_data);

llvm::Error CufftXtExec(cufftHandle plan, void* input, void* output,
                        FftDirection direction);

llvm::Error CufftXtMemcpy(cufftHandle plan, void* dst, void* src,
                          cufftXtCopyType type);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CUFFT_WRAPPER_H_
