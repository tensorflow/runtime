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

// This file declares some helpers that tfrt::gpu::GpuDevice uses to interact
// with Eigen.

// Enable use of gpu* macros, must be defined before including Eigen headers.
#define EIGEN_PERMANENTLY_ENABLE_GPU_HIP_CUDA_DEFINES

// Prevent Eigen from including cuda_runtime.h and include cuda_runtime_api.h
// instead. The former requires a CUDA compatible compiler and header files
// from cuda/include/crt that are not part of the @cuda_headers repository.
#include "cuda_runtime_api.h"  // from @cuda_headers
#define __CUDA_RUNTIME_H__

#include "eigen_support.h"

#define EIGEN_USE_GPU

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace tfrt {
namespace gpu {
namespace {
class EigenStreamInterface : public Eigen::StreamInterface {
 public:
  explicit EigenStreamInterface(wrapper::Stream stream)
      : stream_(static_cast<gpuStream_t>(stream)) {}

  // NB! gpuStream_t and gpuDeviceProp_t are globally typedef-ed to cudaStream_t
  // and cudaDeviceProp, or to hipStream_t and hipDeviceProp_t.  This means a
  // given binary can have Eigen kernels that work for CUDA or HIP, but not
  // both.

  const gpuStream_t& stream() const override { return stream_; }
  const gpuDeviceProp_t& deviceProperties() const override {
    return Eigen::GetGpuDeviceProperties(0);
  };

  // Unimplemented methods that are not yet necessary.
  void* allocate(size_t num_bytes) const override { abort(); };
  void deallocate(void* buffer) const override { abort(); };
  void* scratchpad() const override { abort(); };
  unsigned int* semaphore() const override { abort(); };

 private:
  gpuStream_t stream_;
};
}  // namespace

void internal::EigenStreamInterfaceDeleter::operator()(
    ::Eigen::StreamInterface* interface) const {
  delete interface;
}
void internal::EigenGpuDeviceDeleter::operator()(
    ::Eigen::GpuDevice* device) const {
  delete device;
}

OwningEigenStreamInterface CreateEigenStreamInterface(wrapper::Stream stream) {
  return OwningEigenStreamInterface(new EigenStreamInterface(stream));
}

OwningEigenGpuDevice CreateEigenGpuDevice(::Eigen::StreamInterface* interface) {
  return OwningEigenGpuDevice(new ::Eigen::GpuDevice(interface));
}

}  // namespace gpu
}  // namespace tfrt
