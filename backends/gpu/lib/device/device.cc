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

// This file implements GPU device.
#include "tfrt/gpu/device/device.h"

#include <optional>

#include "eigen_support.h"
#include "tfrt/gpu/device/gpu_config.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/memory/bfc_gpu_allocator.h"
#include "tfrt/gpu/wrapper/cublas_wrapper.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/string_util.h"

namespace tfrt {
namespace gpu {

// A mere wrapper of a GpuAllocator.
class GpuAllocatorWrapper : public GpuAllocator {
 public:
  // `allocator` should not be nullptr.
  explicit GpuAllocatorWrapper(GpuAllocator* allocator);

  ~GpuAllocatorWrapper() override;

 private:
  Expected<GpuPointer> Allocate(size_t size, wrapper::Stream stream) override;
  Error Deallocate(GpuPointer pointer, wrapper::Stream stream) override;

  // The class does not own the underlying allocator.
  GpuAllocator* allocator_;
};

GpuAllocatorWrapper::GpuAllocatorWrapper(GpuAllocator* allocator)
    : allocator_(allocator) {
  assert(allocator_ != nullptr);
}

GpuAllocatorWrapper::~GpuAllocatorWrapper() = default;

Expected<GpuPointer> GpuAllocatorWrapper::Allocate(size_t size,
                                                   wrapper::Stream stream) {
  return GpuAllocator::Allocate(allocator_, size, stream);
}

Error GpuAllocatorWrapper::Deallocate(GpuPointer pointer,
                                      wrapper::Stream stream) {
  return GpuAllocator::Deallocate(allocator_, pointer, stream);
}

class GpuDevice::Impl {
 public:
  explicit Impl(int gpu_ordinal) : gpu_ordinal_(gpu_ordinal) {}

  llvm::Error Initialize();

  int gpu_ordinal_;

  // TODO(sanjoy): we need to figure out how the lifetimes of these objects
  // interact with the lifetime of the GPU op handler.

  wrapper::Device device_;

  // NB! The declaration order here is important.  We want to destroy
  // owned_context_ last.
  //
  // If `owned_context_` is null, `context_` points to a non-owning context.
  // Otherwise, `context_` is set to be `owned_context_.get()`.
  wrapper::OwningContext owned_context_;
  wrapper::Context context_;
  // If `owned_stream` is null, `stream_` points to a non-owning stream.
  // Otherwise, `stream_` is set to be `owned_stream_.get()`.
  wrapper::OwningStream owned_stream_;
  wrapper::Stream stream_;
  wrapper::OwningBlasHandle blas_handle_;
  wrapper::OwningDnnHandle dnn_handle_;

  // NB! The declaration order here is important. The eigen_gpu_device_
  // references eigen_stream_interface_, which references stream_.
  gpu::OwningEigenStreamInterface eigen_stream_interface_;
  gpu::OwningEigenGpuDevice eigen_gpu_device_;

  std::unique_ptr<gpu::GpuAllocator> allocator_;
};

llvm::Error GpuDevice::Impl::Initialize() {
  // TODO(zhangqiaorjc): Generalize to multi-GPU.
  TFRT_ASSIGN_OR_RETURN(device_,
                        DeviceGet(wrapper::Platform::CUDA, gpu_ordinal_));
  std::optional<wrapper::CurrentContext> current_context;

  // Use external GPU resources if they are available.
  if (auto gpu_resources = gpu::GetTfrtGpuResources(device_)) {
    // Set a non-owning context.
    context_ = gpu_resources->gpu_context;
    TFRT_ASSIGN_OR_RETURN(auto current, CtxSetCurrent(context_));
    current_context.emplace(current);

    allocator_ = gpu_resources->allocator_factory(context_);

    stream_ = gpu_resources->stream;
  } else {
    TFRT_ASSIGN_OR_RETURN(owned_context_, DevicePrimaryCtxRetain(device_));
    context_ = owned_context_.get();
    TFRT_ASSIGN_OR_RETURN(auto current, CtxSetCurrent(context_));
    current_context.emplace(current);

    TFRT_ASSIGN_OR_RETURN(owned_stream_, StreamCreateNonBlocking(current));
    stream_ = owned_stream_.get();

    allocator_ = std::make_unique<gpu::BfcGpuAllocator>(current);
  }

  eigen_stream_interface_ = gpu::CreateEigenStreamInterface(stream_);
  eigen_gpu_device_ = gpu::CreateEigenGpuDevice(eigen_stream_interface_.get());

  // TODO(iga): Only log errors during BLAS handle creation?
  TFRT_ASSIGN_OR_RETURN(blas_handle_, BlasCreate(*current_context));
  if (auto error = wrapper::BlasSetStream(blas_handle_.get(), stream_))
    return error;
  if (auto error = wrapper::CublasSetMathMode(
          static_cast<cublasHandle_t>(blas_handle_.get()),
          CUBLAS_TENSOR_OP_MATH))
    return error;

  TFRT_ASSIGN_OR_RETURN(dnn_handle_, wrapper::DnnCreate(*current_context));
  if (auto error = wrapper::DnnSetStream(dnn_handle_.get(), stream_))
    return error;

  return Error::success();
}

GpuDevice::GpuDevice(string_view name, int gpu_ordinal)
    : Device(kDeviceType, name), impl_(std::make_unique<Impl>(gpu_ordinal)) {}

llvm::Error GpuDevice::Initialize() { return impl_->Initialize(); }

wrapper::Stream GpuDevice::stream() const { return impl_->stream_; }

AsyncValueRef<gpu::GpuAllocator> GpuDevice::allocator() const {
  return MakeAvailableAsyncValueRef<GpuAllocatorWrapper>(
      impl_->allocator_.get());
}

Eigen::GpuDevice* GpuDevice::eigen_gpu_device() const {
  return impl_->eigen_gpu_device_.get();
}

wrapper::BlasHandle GpuDevice::blas_handle() const {
  return impl_->blas_handle_.get();
}

wrapper::DnnHandle GpuDevice::dnn_handle() const {
  return impl_->dnn_handle_.get();
}

llvm::Expected<wrapper::CurrentContext> GpuDevice::SetCurrentContext() const {
  return CtxSetCurrent(impl_->context_);
}

}  // namespace gpu
}  // namespace tfrt
