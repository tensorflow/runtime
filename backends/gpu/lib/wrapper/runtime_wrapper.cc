// Copyright 2021 The TensorFlow Runtime Authors
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

// Thin abstraction layer for CUDA and HIP runtime API.
#include "tfrt/gpu/wrapper/runtime_wrapper.h"

#include "tfrt/gpu/wrapper/cudart_wrapper.h"
#include "tfrt/gpu/wrapper/hip_wrapper.h"
#include "tfrt/support/logging.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::Error Free(std::nullptr_t, Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CudaFree(nullptr);
    case Platform::ROCm:
      return HipFree(nullptr);
    default:
      return llvm::Error::success();
  }
}

llvm::Expected<int> RuntimeGetVersion(Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CudaRuntimeGetVersion();
    case Platform::ROCm:
      return HipRuntimeGetVersion();
    default:
      return 0;
  }
}

llvm::Error GetLastError(CurrentContext current) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudaGetLastError(current);
    case Platform::ROCm:
      return HipGetLastError(current);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error PeekAtLastError(CurrentContext current) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudaPeekAtLastError(current);
    case Platform::ROCm:
      return HipPeekAtLastError(current);
    default:
      return InvalidPlatform(platform);
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
