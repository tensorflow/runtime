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

// This file declares GpuDevice which holds GPU device specific information.

#ifndef TFRT_GPU_DEVICE_DEVICE_H_
#define TFRT_GPU_DEVICE_DEVICE_H_

#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/dnn_wrapper.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"
#include "tfrt/host_context/device.h"
#include "tfrt/support/forward_decls.h"

namespace Eigen {
class GpuDevice;
}

namespace tfrt {
namespace gpu {
class GpuAllocator;

class GpuDevice : public Device, public DeviceTraits<GpuDevice> {
 public:
  static const char* type_name() {
    static constexpr char kName[] = "gpu";
    return kName;
  }

  explicit GpuDevice(string_view name, int gpu_ordinal);

  llvm::Error Initialize();

  // The inputs to the GPU dispatch function are available for reading on this
  // stream.  The outputs from the dispatch must also be ready for reading on
  // this stream.
  wrapper::Stream stream() const;

  // Allocator for allocating GPU device memory.
  AsyncValueRef<gpu::GpuAllocator> allocator() const;

  // Eigen GPU device. Used to launch Eigen kernels.
  Eigen::GpuDevice* eigen_gpu_device() const;

  // GPU BLAS library handle. Used to launch BLAS routines.
  wrapper::BlasHandle blas_handle() const;

  // GPU DNN library handle. Used to launch convolutions etc.
  wrapper::DnnHandle dnn_handle() const;

  // Set the context of current thread and return it. See the documentation
  // for wrapper::CurrentContext for more details.
  llvm::Expected<wrapper::CurrentContext> SetCurrentContext() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_DEVICE_DEVICE_H_
