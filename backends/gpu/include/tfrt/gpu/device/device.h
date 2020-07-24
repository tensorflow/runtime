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

//===- device.h -------------------------------------------------*- C++ -*-===//
//
// This file declares GpuDevice which holds GPU device specific information.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_GPU_DEVICE_DEVICE_H_
#define TFRT_GPU_DEVICE_DEVICE_H_

#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/stream/blas_wrapper.h"
#include "tfrt/gpu/stream/dnn_wrapper.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/host_context/device.h"

namespace Eigen {
class GpuDevice;
}

namespace tfrt {
namespace gpu {
class GpuAllocator;
}  // namespace gpu

using gpu::stream::CurrentContext;

class GpuDevice : public Device {
 public:
  explicit GpuDevice(int gpu_ordinal);

  llvm::Error Initialize();

  // The inputs to the GPU dispatch function are available for reading on this
  // stream.  The outputs from the dispatch must also be ready for reading on
  // this stream.
  gpu::stream::Stream stream() const;

  // Allocator for allocating GPU device memory.
  gpu::GpuAllocator* allocator() const;

  // Eigen GPU device. Used to launch Eigen kernels.
  Eigen::GpuDevice* eigen_gpu_device() const;

  // GPU BLAS library handle. Used to launch BLAS routines.
  gpu::stream::BlasHandle blas_handle() const;

  // GPU DNN library handle. Used to launch convolutions etc.
  gpu::stream::DnnHandle dnn_handle() const;

  // Create a current context. It is usually called inside the dispatch
  // function.  See the documentation for gpu::stream::CurrentContext for more
  // details.
  gpu::stream::CurrentContext CreateContext() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace tfrt

#endif  // TFRT_GPU_DEVICE_DEVICE_H_
