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

// Thin abstraction layer for cuSOLVER and rocSOLVER.
#include "tfrt/gpu/stream/solver_wrapper.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/stream/cusolver_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace stream {

void internal::SolverDnHandleDeleter::operator()(SolverDnHandle handle) const {
  LogIfError(SolverDnDestroy(handle));
}

llvm::Expected<OwningSolverDnHandle> SolverDnCreate(CurrentContext current) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CusolverDnCreate(current);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error SolverDnDestroy(SolverDnHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CusolverDnDestroy(handle);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error SolverDnSetStream(SolverDnHandle handle, Stream stream) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CusolverDnSetStream(handle, stream);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<Stream> SolverDnGetStream(SolverDnHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CusolverDnGetStream(handle);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt
