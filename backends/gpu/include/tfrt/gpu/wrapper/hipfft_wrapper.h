/*
 * Copyright 2021 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Thin wrapper around the hipfft API adding llvm::Error.
#ifndef TFRT_GPU_WRAPPER_HIPFFT_WRAPPER_H_
#define TFRT_GPU_WRAPPER_HIPFFT_WRAPPER_H_

#include "tfrt/gpu/wrapper/fft_wrapper.h"
#include "tfrt/gpu/wrapper/hipfft_stub.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::Expected<LibraryVersion> HipfftGetVersion();

// Frees all GPU resources associated with the plan and destroys internal data
// structures.
llvm::Error HipfftDestroy(hipfftHandle plan);

// Sets the stream for execution of hipFFT functions. Note that these functions
// may consist of many kernel invocations.
llvm::Error HipfftSetStream(hipfftHandle plan, hipStream_t stream);

// Creates FFT plans for the specific dimension, window dimensions, transform
// type.
llvm::Expected<OwningFftHandle> HipfftPlan1d(int nx, hipfftType type,
                                             int batch);
llvm::Expected<OwningFftHandle> HipfftPlan2d(int nx, int ny, hipfftType type);
llvm::Expected<OwningFftHandle> HipfftPlan3d(int nx, int ny, int nz,
                                             hipfftType type);

llvm::Expected<size_t> HipfftGetSize(hipfftHandle plan);

llvm::Error HipfftSetWorkArea(hipfftHandle plan, Pointer<void> work_area);

llvm::Error HipfftExecC2C(hipfftHandle plan, hipfftComplex* input_data,
                          hipfftComplex* output_data, FftDirection direction);

llvm::Error HipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex* input_data,
                          hipfftDoubleComplex* output_data,
                          FftDirection direction);

llvm::Error HipfftExecR2C(hipfftHandle plan, hipfftReal* input_data,
                          hipfftComplex* output_data);

llvm::Error HipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal* input_data,
                          hipfftDoubleComplex* output_data);

llvm::Error HipfftExecC2R(hipfftHandle plan, hipfftComplex* input_data,
                          hipfftReal* output_data);

llvm::Error HipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex* input_data,
                          hipfftDoubleReal* output_data);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_HIPFFT_WRAPPER_H_
