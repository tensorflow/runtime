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

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/wrapper/cufft_wrapper.h"
#include "tfrt/gpu/wrapper/rocfft_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

void internal::FftHandleDeleter::operator()(FftHandle handle) const {
  LogIfError(FftDestroy(handle));
}

llvm::Error FftDestroy(FftHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CufftDestroy(handle);
    case Platform::ROCm:
      if (auto error = RocfftExecutionInfoDestroy(handle)) return error;
      return RocfftPlanDestroy(handle);
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
      return RocfftExecutionInfoSetStream(handle, stream);
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
      return RocfftPlanGetWorkBufferSize(handle);
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
      return RocfftExecutionInfoSetWorkBuffer(handle, workspace, size_bytes);
    default:
      return InvalidPlatform(platform);
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
