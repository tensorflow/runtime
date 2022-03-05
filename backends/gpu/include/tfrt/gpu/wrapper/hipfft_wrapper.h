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

raw_ostream& Print(raw_ostream& os, hipfftResult_t result);
raw_ostream& Print(raw_ostream& os, hipfftType value);
raw_ostream& Print(raw_ostream& os, hipfftDirection value);

Expected<hipfftType> Parse(llvm::StringRef name, hipfftType);
Expected<hipfftDirection> Parse(llvm::StringRef name, hipfftDirection);

namespace internal {
template <>
struct EnumPlatform<FftType, hipfftType> : RocmPlatformType {};
template <>
struct EnumPlatform<FftDirection, hipfftDirection> : RocmPlatformType {};

template <>
struct EnumStream<FftType, Platform::ROCm>
    : EnumStreamPtrs<hipfftType, Parse, Print> {};
template <>
struct EnumStream<FftDirection, Platform::ROCm>
    : EnumStreamPtrs<hipfftDirection, Parse, Print> {};
}  // namespace internal

llvm::Expected<LibraryVersion> HipfftGetVersion();

// Creates an opaque handle and allocates small data for the plan. Use
// HipfftMakePlanMany to do the plan generation.
llvm::Expected<OwningFftHandle> HipfftCreate(CurrentContext current);

// Frees all GPU resources associated with the handle and destroys internal data
// structures.
llvm::Error HipfftDestroy(hipfftHandle handle);

// Sets the stream for execution of hipFFT functions. Note that these functions
// may consist of many kernel invocations.
llvm::Error HipfftSetStream(hipfftHandle handle, hipStream_t stream);

llvm::Expected<size_t> HipfftMakePlanMany(
    hipfftHandle handle, hipfftType type, int64_t batch, ArrayRef<int64_t> dims,
    ArrayRef<int64_t> input_embed, int64_t input_stride, int64_t input_dist,
    ArrayRef<int64_t> output_embed, int64_t output_stride, int64_t output_dist);

llvm::Expected<size_t> HipfftGetSize(hipfftHandle handle);

llvm::Error HipfftDisableAutoAllocation(hipfftHandle handle);
llvm::Error HipfftEnableAutoAllocation(hipfftHandle handle);

llvm::Error HipfftSetWorkArea(hipfftHandle handle, Pointer<void> work_area);

llvm::Error HipfftExec(CurrentContext current, hipfftHandle handle,
                       Pointer<const void> raw_input, Pointer<void> raw_output,
                       hipfftType type, hipfftDirection direction);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_HIPFFT_WRAPPER_H_
