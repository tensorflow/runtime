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

// Thin abstraction layer for cuFFT and rocFFT.
#include "tfrt/gpu/wrapper/fft_wrapper.h"

#include <cstddef>

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/wrapper/cufft_wrapper.h"
#include "tfrt/gpu/wrapper/hipfft_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

void internal::FftHandleDeleter::operator()(FftHandle handle) const {
  LogIfError(FftDestroy(handle));
}

llvm::Expected<OwningFftHandle> FftCreate(CurrentContext current) {
  switch (current.platform()) {
    case Platform::CUDA:
      return CufftCreate(current);
    case Platform::ROCm:
      return HipfftCreate(current);
    default:
      return InvalidPlatform(current.platform());
  }
}

llvm::Error FftDestroy(FftHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CufftDestroy(handle);
    case Platform::ROCm:
      return HipfftDestroy(handle);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error FftSetStream(FftHandle handle, Stream stream) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CufftSetStream(handle, stream);
    case Platform::ROCm:
      return HipfftSetStream(handle, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error FftDisableAutoAllocation(FftHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CufftDisableAutoAllocation(handle);
    case Platform::ROCm:
      return HipfftDisableAutoAllocation(handle);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error FftEnableAutoAllocation(FftHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CufftEnableAutoAllocation(handle);
    case Platform::ROCm:
      return HipfftEnableAutoAllocation(handle);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<size_t> FftGetWorkspaceSize(FftHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CufftGetSize(handle);
    case Platform::ROCm:
      return HipfftGetSize(handle);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error FftSetWorkspace(FftHandle handle, Pointer<void> workspace,
                            size_t size_bytes) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CufftSetWorkArea(handle, workspace);
    case Platform::ROCm:
      return HipfftSetWorkArea(handle, workspace);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<size_t> FftMakePlanMany(
    FftHandle handle, FftType type, int64_t batch, llvm::ArrayRef<int64_t> dims,
    llvm::ArrayRef<int64_t> input_embed, int64_t input_stride,
    int64_t input_dist, llvm::ArrayRef<int64_t> output_embed,
    int64_t output_stride, int64_t output_dist) {
  switch (handle.platform()) {
    case Platform::CUDA:
      return CufftMakePlanMany(handle, type, batch, dims, input_embed,
                               input_stride, input_dist, output_embed,
                               output_stride, output_dist);
    case Platform::ROCm:
      return HipfftMakePlanMany(handle, type, batch, dims, input_embed,
                                input_stride, input_dist, output_embed,
                                output_stride, output_dist);
    default:
      return InvalidPlatform(handle.platform());
  }
}

llvm::Expected<size_t> FftMakePlanMany(FftHandle handle, FftType type,
                                       int64_t batch,
                                       llvm::ArrayRef<int64_t> dims) {
  return FftMakePlanMany(handle, type, batch, dims, {}, 0, 0, {}, 0, 0);
}

llvm::Error FftExec(CurrentContext current, FftHandle handle,
                    wrapper::Pointer<const void> input,
                    wrapper::Pointer<void> output, FftType type,
                    FftDirection direction) {
  Platform platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CufftExec(current, handle, input, output, type, direction);
    case Platform::ROCm:
      return HipfftExec(current, handle, input, output, type, direction);
    default:
      return InvalidPlatform(platform);
  }
}
}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
