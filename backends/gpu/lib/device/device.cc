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

//===- device.cc ----------------------------------------------------------===//
//
// This file implements GPU device.
//
//===----------------------------------------------------------------------===//
#include "tfrt/gpu/device/device.h"

#include "eigen_support.h"
#include "tfrt/gpu/device/gpu_config.h"
#include "tfrt/gpu/memory/bfc_gpu_allocator.h"
#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/stream/cublas_wrapper.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/string_util.h"

namespace tfrt {
using gpu::stream::CtxSetCurrent;  // TODO(csigg): remove once ADL is fixed.

class GpuDevice::Impl {
 public:
  explicit Impl(int gpu_ordinal) : gpu_ordinal_(gpu_ordinal) {}

  llvm::Error Initialize();

  int gpu_ordinal_;

  // TODO(sanjoy): we need to figure out how the lifetimes of these objects
  // interact with the lifetime of the GPU op handler.

  gpu::stream::Device device_;

  // NB! The declaration order here is important.  We want to destroy
  // owned_context_ last.
  //
  // If `owned_context_` is null, `context_` points to a non-owning context.
  // Otherwise, `context_` is set to be `owned_context_.get()`.
  gpu::stream::OwningContext owned_context_;
  gpu::stream::Context context_;
  // If `owned_stream` is null, `stream_` points to a non-owning stream.
  // Otherwise, `stream_` is set to be `owned_stream_.get()`.
  gpu::stream::OwningStream owned_stream_;
  gpu::stream::Stream stream_;
  gpu::stream::OwningBlasHandle blas_handle_;
  gpu::stream::OwningDnnHandle dnn_handle_;

  // NB! The declaration order here is important. The eigen_gpu_device_
  // references eigen_stream_interface_, which references stream_.
  gpu::OwningEigenStreamInterface eigen_stream_interface_;
  gpu::OwningEigenGpuDevice eigen_gpu_device_;

  std::unique_ptr<gpu::GpuAllocator> allocator_;
};

llvm::Error GpuDevice::Impl::Initialize() {
  // TODO(zhangqiaorjc): Generalize to multi-GPU.
  TFRT_ASSIGN_OR_RETURN(device_,
                        DeviceGet(gpu::stream::Platform::CUDA, gpu_ordinal_));
  llvm::Optional<gpu::stream::CurrentContext> current_context;

  // Use external GPU resources if they are available.
  if (auto gpu_resources = gpu::GetTfrtGpuResources(device_)) {
    // Set a non-owning context.
    context_ = gpu_resources->gpu_context;
    TFRT_ASSIGN_OR_RETURN(auto current, CtxSetCurrent(context_));
    current_context.emplace(current);

    allocator_ = std::unique_ptr<gpu::GpuAllocator>(
        gpu_resources->allocator_factory(context_));

    stream_ = gpu_resources->stream;
  } else {
    TFRT_ASSIGN_OR_RETURN(owned_context_, DevicePrimaryCtxRetain(device_));
    context_ = owned_context_.get();
    TFRT_ASSIGN_OR_RETURN(auto current, CtxSetCurrent(context_));
    current_context.emplace(current);

    TFRT_ASSIGN_OR_RETURN(
        owned_stream_,
        StreamCreate(current, gpu::stream::StreamFlags::DEFAULT));
    stream_ = owned_stream_.get();

    allocator_ =
        std::unique_ptr<gpu::GpuAllocator>(new gpu::BfcGpuAllocator(current));
  }

  eigen_stream_interface_ = gpu::CreateEigenStreamInterface(stream_);
  eigen_gpu_device_ = gpu::CreateEigenGpuDevice(eigen_stream_interface_.get());

  // TODO(iga): Only log errors during BLAS handle creation?
  TFRT_ASSIGN_OR_RETURN(blas_handle_, BlasCreate(*current_context));
  if (auto error = gpu::stream::BlasSetStream(blas_handle_.get(), stream_))
    return error;
  if (auto error = gpu::stream::CublasSetMathMode(
          static_cast<cublasHandle_t>(blas_handle_.get()),
          CUBLAS_TENSOR_OP_MATH))
    return error;

  TFRT_ASSIGN_OR_RETURN(dnn_handle_, gpu::stream::DnnCreate(*current_context));
  if (auto error = gpu::stream::DnnSetStream(dnn_handle_.get(), stream_))
    return error;

  return Error::success();
}

GpuDevice::GpuDevice(int gpu_ordinal)
    : Device(GetStaticDeviceType("gpu"), StrCat("GPU:", gpu_ordinal)),
      impl_(std::make_unique<Impl>(gpu_ordinal)) {}

llvm::Error GpuDevice::Initialize() { return impl_->Initialize(); }

gpu::stream::Stream GpuDevice::stream() const { return impl_->stream_; }

gpu::GpuAllocator* GpuDevice::allocator() const {
  return impl_->allocator_.get();
}

Eigen::GpuDevice* GpuDevice::eigen_gpu_device() const {
  return impl_->eigen_gpu_device_.get();
}

gpu::stream::BlasHandle GpuDevice::blas_handle() const {
  return impl_->blas_handle_.get();
}

gpu::stream::DnnHandle GpuDevice::dnn_handle() const {
  return impl_->dnn_handle_.get();
}

gpu::stream::CurrentContext GpuDevice::CreateContext() const {
  // FIXME(sanjoy): Add proper error handling.
  llvm::ExitOnError die_if_error;
  return die_if_error(CtxSetCurrent(impl_->context_));
}

}  // namespace tfrt
