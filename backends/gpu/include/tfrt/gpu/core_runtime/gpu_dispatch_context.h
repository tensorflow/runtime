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

// This file declares GpuDispatchContext which holds information needed by
// GPU dispatch functions.

#ifndef TFRT_GPU_CORE_RUNTIME_GPU_DISPATCH_CONTEXT_H_
#define TFRT_GPU_CORE_RUNTIME_GPU_DISPATCH_CONTEXT_H_

#include "tfrt/gpu/device/device.h"
#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/wrapper/blas_wrapper.h"
#include "tfrt/gpu/wrapper/dnn_wrapper.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"

namespace Eigen {
class GpuDevice;
}

namespace tfrt {
namespace gpu {
class GpuDispatchContext {
 public:
  explicit GpuDispatchContext(const GpuDevice* device)
      : device_(device),
        stream_(device->stream()),
        allocator_(device->allocator()),
        eigen_gpu_device_(device->eigen_gpu_device()),
        blas_handle_(device->blas_handle()),
        dnn_handle_(device->dnn_handle()),
        current_context_(std::move(device->CreateContext())) {}

  // The inputs to the GPU dispatch function are available for reading on this
  // stream.  The outputs from the dispatch must also be ready for reading on
  // this stream.
  wrapper::Stream stream() const { return stream_; }

  // Allocator for allocating GPU device memory.
  gpu::GpuCrtAllocator* allocator() const { return allocator_; }

  // Eigen GPU device. Used to launch Eigen kernels.
  Eigen::GpuDevice* eigen_gpu_device() const { return eigen_gpu_device_; }

  // GPU BLAS library handle. Used to launch BLAS routines.
  wrapper::BlasHandle blas_handle() const { return blas_handle_; }

  // GPU DNN library handle. Used to launch convolutions etc.
  wrapper::DnnHandle dnn_handle() const { return dnn_handle_; }

  // The GPU device sets the current context before calling into the dispatch
  // function.  See the documentation for wrapper::CurrentContext for more
  // details.
  wrapper::CurrentContext current_context() const { return current_context_; }

  const GpuDevice& device() const { return *device_; }

 private:
  const GpuDevice* device_;
  wrapper::Stream stream_;
  GpuCrtAllocator* allocator_;
  Eigen::GpuDevice* eigen_gpu_device_;
  wrapper::BlasHandle blas_handle_;
  wrapper::DnnHandle dnn_handle_;
  wrapper::CurrentContext current_context_;
};
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_CORE_RUNTIME_GPU_DISPATCH_CONTEXT_H_
