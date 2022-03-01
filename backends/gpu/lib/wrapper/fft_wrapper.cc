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

llvm::Expected<OwningFftHandle> FftCreate(Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CufftCreate();
    case Platform::ROCm:
      return HipfftCreate();
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error FftDestroy(FftHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CufftDestroy(handle);
    case Platform::ROCm:
      return HipfftDestroy(hipfftHandle(handle));
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
      return HipfftSetStream(hipfftHandle(handle), stream);
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
      return HipfftGetSize(hipfftHandle(handle));
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
      return HipfftSetWorkArea(hipfftHandle(handle), workspace);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<size_t> FftMakePlanMany(
    FftHandle plan, FftType type, int64_t batch, int rank,
    llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<int64_t> input_embed,
    int64_t input_stride, llvm::ArrayRef<int64_t> output_embed,
    int64_t output_stride, int64_t input_dist, int64_t output_dist) {
  switch (plan.platform()) {
    case Platform::CUDA:
      return CufftMakePlanMany(
          plan, FftTypeToCufftType(type).get(), batch, rank, dims, input_embed,
          input_stride, output_embed, output_stride, input_dist, output_dist);
    case Platform::ROCm:
      return HipfftMakePlanMany(hipfftHandle(plan), rank, dims, input_embed,
                                input_stride, input_dist, output_embed,
                                output_stride, output_dist,
                                FftTypeToHipfftType(type).get(), batch);
    default:
      return InvalidPlatform(plan.platform());
  }
}

llvm::Error FftExec(FftHandle plan, wrapper::Pointer<void> input,
                    wrapper::Pointer<void> output, FftType type) {
  Platform platform = plan.platform();
  switch (platform) {
    case Platform::CUDA:
      return CufftExec(plan, input, output, type);
    case Platform::ROCm:
      return HipfftExec(hipfftHandle(plan), input, output, type);
    default:
      return InvalidPlatform(platform);
  }
}
}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
