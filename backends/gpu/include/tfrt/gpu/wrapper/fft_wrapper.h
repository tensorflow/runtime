/*
 * Copyright 2020 The TensorFlow Runtime Authors
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

// Thin abstraction layer for cuFFT and rocFFT.
#ifndef TFRT_GPU_WRAPPER_FFT_WRAPPER_H_
#define TFRT_GPU_WRAPPER_FFT_WRAPPER_H_

#include <cstddef>
#include <memory>

#include "tfrt/gpu/wrapper/wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

enum struct FftType {
  kZ2ZForward,
  kC2CForward,
  kZ2ZInverse,
  kC2CInverse,
  kZ2D,
  kC2R,
  kD2Z,
  kR2C
};

class FftHandle {
 public:
  FftHandle() = default;
  FftHandle(std::nullptr_t) {}  // NOLINT(google-explicit-constructor)
  // NOLINTNEXTLINE(google-explicit-constructor)
  FftHandle(cufftHandle handle) : platform_(Platform::CUDA) {
    union_.cuda_handle = handle;
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  FftHandle(hipfftHandle handle) : platform_(Platform::ROCm) {
    union_.hip_handle = handle;
  }
  FftHandle& operator=(const FftHandle&) = default;
  // Required for std::unique_ptr<FftHandle>.
  FftHandle& operator=(std::nullptr_t) {
    platform_ = Platform::NONE;
    return *this;
  }
  // Required for std::unique_ptr<FftHandle>.
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator bool() const { return platform() != Platform::NONE; }
  operator cufftHandle() const {  // NOLINT(google-explicit-constructor)
    assert(platform() == Platform::CUDA);
    return union_.cuda_handle;
  }
  operator hipfftHandle() const {  // NOLINT(google-explicit-constructor)
    assert(platform() == Platform::ROCm);
    return union_.hip_handle;
  }
  Platform platform() const { return platform_; }

  bool operator==(cufftHandle handle) const {
    return platform() == Platform::CUDA && union_.cuda_handle == handle;
  }
  bool operator!=(cufftHandle handle) const { return !(*this == handle); }
  bool operator==(hipfftHandle handle) const {
    return platform() == Platform::ROCm && union_.hip_handle == handle;
  }
  bool operator!=(hipfftHandle handle) const { return !(*this == handle); }
  bool operator==(std::nullptr_t) const { return platform() == Platform::NONE; }
  bool operator!=(std::nullptr_t) const { return !(*this == nullptr); }

  // For member access from std::unique_ptr.
  const FftHandle* operator->() const { return this; }

 private:
  Platform platform_ = Platform::NONE;
  union {
    cufftHandle cuda_handle;
    hipfftHandle hip_handle;
  } union_;
};

enum class FftDirection : int { kForward, kInverse };

namespace internal {
// Helper to wrap resources and memory into RAII types.
struct FftHandleDeleter {
  using pointer = FftHandle;
  void operator()(FftHandle handle) const;
};
}  // namespace internal

// RAII wrappers for resources. Instances own the underlying resource.
using OwningFftHandle = internal::OwningResource<internal::FftHandleDeleter>;

llvm::Expected<OwningFftHandle> FftCreate(Platform platform);
llvm::Error FftDestroy(FftHandle handle);
llvm::Error FftSetStream(FftHandle handle, Stream stream);
llvm::Expected<size_t> FftGetWorkspaceSize(FftHandle handle);
llvm::Error FftSetWorkspace(FftHandle handle, Pointer<void> workspace,
                            size_t size_bytes);
llvm::Expected<size_t> FftMakePlanMany(
    FftHandle handle, FftType type, int64_t batch, int rank,
    llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<int64_t> input_embed,
    int64_t input_stride, llvm::ArrayRef<int64_t> output_embed,
    int64_t output_stride, int64_t input_dist, int64_t output_dist);
llvm::Error FftExec(FftHandle handle, wrapper::Pointer<void> input,
                    wrapper::Pointer<void> output, FftType type);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_FFT_WRAPPER_H_
