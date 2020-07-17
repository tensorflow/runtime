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

//===- gpu_dispatch_context.h -----------------------------------*- C++ -*-===//
//
// This file declares GpuDispatchContext which holds information needed by
// GPU dispatch functions.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_GPU_CORE_RUNTIME_GPU_DISPATCH_CONTEXT_H_
#define TFRT_GPU_CORE_RUNTIME_GPU_DISPATCH_CONTEXT_H_

#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/stream/blas_wrapper.h"
#include "tfrt/gpu/stream/dnn_wrapper.h"
#include "tfrt/gpu/stream/stream_wrapper.h"

namespace Eigen {
class GpuDevice;
}

namespace tfrt {
class GpuDispatchContext {
 public:
  explicit GpuDispatchContext(gpu::stream::Stream stream,
                              gpu::GpuAllocator* allocator,
                              Eigen::GpuDevice* eigen_gpu_device,
                              gpu::stream::BlasHandle blas_handle,
                              gpu::stream::DnnHandle dnn_handle,
                              gpu::stream::CurrentContext current_context)
      : stream_(stream),
        allocator_(allocator),
        eigen_gpu_device_(eigen_gpu_device),
        blas_handle_(blas_handle),
        dnn_handle_(dnn_handle),
        current_context_(std::move(current_context)) {}

  // The inputs to the GPU dispatch function are available for reading on this
  // stream.  The outputs from the dispatch must also be ready for reading on
  // this stream.
  gpu::stream::Stream stream() const { return stream_; }

  // Allocator for allocating GPU device memory.
  gpu::GpuAllocator* allocator() const { return allocator_; }

  // Eigen GPU device. Used to launch Eigen kernels.
  Eigen::GpuDevice* eigen_gpu_device() const { return eigen_gpu_device_; }

  // GPU BLAS library handle. Used to launch BLAS routines.
  gpu::stream::BlasHandle blas_handle() const { return blas_handle_; }

  // GPU DNN library handle. Used to launch convolutions etc.
  gpu::stream::DnnHandle dnn_handle() { return dnn_handle_; }

  // The GPU device sets the current context before calling into the dispatch
  // function.  See the documentation for gpu::stream::CurrentContext for more
  // details.
  gpu::stream::CurrentContext current_context() const {
    return current_context_;
  }

 private:
  gpu::stream::Stream stream_;
  gpu::GpuAllocator* allocator_;
  Eigen::GpuDevice* eigen_gpu_device_;
  gpu::stream::BlasHandle blas_handle_;
  gpu::stream::DnnHandle dnn_handle_;
  gpu::stream::CurrentContext current_context_;
};
}  // namespace tfrt

#endif  // TFRT_GPU_CORE_RUNTIME_GPU_DISPATCH_CONTEXT_H_
