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

// Cufft can only configure plans up to 3D.
constexpr bool IsValidFftRank(int rank) {
  return rank == 1 || rank == 2 || rank == 3;
}

llvm::Expected<cufftType> FftTypeToCufftType(FftType type);

llvm::Expected<LibraryVersion> CufftGetVersion();

// Creates an opaque handle and allocates small data for the plan. Use
// CufftMakePlan* to do the plan generation. (See
// https://docs.nvidia.com/cuda/cufft/index.html#plan-extensible).
llvm::Expected<OwningFftHandle> CufftCreate();

// Frees all GPU resources associated with the plan and destroys internal data
// structures.
llvm::Error CufftDestroy(cufftHandle plan);

// Sets the stream for execution of cuFFT functions. Note that these functions
// may consist of many kernel invocations.
llvm::Error CufftSetStream(cufftHandle plan, cudaStream_t stream);

// Following a call to CufftCreate, makes a plan for the specified signal size
// and type. Work size contains the size(s) in bytes of the work areas for each
// GPU used.
llvm::Expected<size_t> CufftMakePlanMany(
    cufftHandle plan, cufftType type, int64_t batch, int rank,
    llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<int64_t> input_embed,
    int64_t input_stride, llvm::ArrayRef<int64_t> output_embed,
    int64_t output_stride, int64_t input_dist, int64_t output_dist);

// Functions for getting estimated size of work area for temporary results
// during plan execution.
// See https://docs.nvidia.com/cuda/cufft/index.html#work-estimate.

llvm::Expected<size_t> CufftEstimateMany(cufftType type, int batch, int rank,
                                         llvm::SmallVector<int, 3> dims,
                                         llvm::SmallVector<int, 3> input_embed,
                                         int input_stride,
                                         llvm::SmallVector<int, 3> output_embed,
                                         int output_stride, int input_dist,
                                         int output_dist);
llvm::Expected<size_t> CufftGetSizeMany(
    cufftHandle plan, cufftType type, int batch, int rank,
    llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<int64_t> input_embed,
    int64_t input_stride, llvm::ArrayRef<int64_t> output_embed,
    int64_t output_stride, int64_t input_dist, int64_t output_dist);

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

llvm::Error CufftExecC2C(cufftHandle plan, Pointer<cufftComplex> input_data,
                         Pointer<cufftComplex> output_data,
                         FftDirection direction);
llvm::Error CufftExecZ2Z(cufftHandle plan,
                         Pointer<cufftDoubleComplex> input_data,
                         Pointer<cufftDoubleComplex> output_data,
                         FftDirection direction);

llvm::Error CufftExecR2C(cufftHandle plan, Pointer<cufftReal> input_data,
                         Pointer<cufftComplex> output_data);
llvm::Error CufftExecD2Z(cufftHandle plan, Pointer<cufftDoubleReal> input_data,
                         Pointer<cufftDoubleComplex> output_data);

llvm::Error CufftExecC2R(cufftHandle plan, Pointer<cufftComplex> input_data,
                         Pointer<cufftReal> output_data);
llvm::Error CufftExecZ2D(cufftHandle plan,
                         Pointer<cufftDoubleComplex> input_data,
                         Pointer<cufftDoubleReal> output_data);

llvm::Error CufftExec(cufftHandle plan, Pointer<void> raw_input,
                      Pointer<void> raw_output, FftType type);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CUFFT_WRAPPER_H_
