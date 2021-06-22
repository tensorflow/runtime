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

// This file implements the tfrt_gpu.dnn kernels.
#include <cstdint>
#include <memory>
#include <string>

#include "kernels_detail.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"
#include "tfrt/gpu/wrapper/miopen_wrapper.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"
#include "tfrt/tensor/tensor_type_registration.h"

namespace tfrt {
namespace gpu {

template <typename T>
static llvm::Expected<ArrayRef<T>> GetTensorData(const DenseHostTensor& t) {
  if (t.shape().GetRank() != 1) {
    return MakeStringError(
        "GetTensorData: input tensor is not a rank 1 tensor");
  }
  if (t.dtype() != GetDType<T>())
    return tfrt::MakeStringError(
        "GetTensorData: input tensor type mismatch with desired vector type.");
  return ArrayRef<T>(static_cast<const T*>(t.data()), t.NumElements());
}

static Expected<GpuDnnHandle> DnnCreate(Argument<GpuStream> stream) {
  auto current = wrapper::CtxSetCurrent(stream->context());
  if (!current) return current.takeError();
  auto handle = wrapper::DnnCreate(*current);
  if (!handle) return handle.takeError();
  if (auto error = wrapper::DnnSetStream(handle->get(), stream->get()))
    return std::move(error);
  return GpuDnnHandle(stream.ValueRef(), std::move(*handle));
}

static Expected<wrapper::OwningDnnPoolingDescriptor> DnnCreatePoolingDescriptor(
    const GpuContext& context, uint32_t mode, uint32_t nan_propagation,
    const DenseHostTensor& window_dimensions, const DenseHostTensor& paddings,
    const DenseHostTensor& strides) {
  auto current = wrapper::CtxSetCurrent(context.get());
  auto descriptor = wrapper::DnnCreatePoolingDescriptor(context->platform());
  if (!descriptor) return descriptor.takeError();
  if (window_dimensions.dtype() != tfrt::DType::I32)
    return MakeStringError(
        "DnnCreatePoolingDescriptor: window_dimensions is not an I32 tensor.");
  auto window_dimensions_data = GetTensorData<int>(window_dimensions);
  if (!window_dimensions_data)
    return MakeStringError(
        "DnnCreatePoolingDescriptor: window_dimensions is not a 1D tensor.");
  if (paddings.dtype() != tfrt::DType::I32)
    return MakeStringError(
        "DnnSetPoolingDescriptor: paddings is not an I32 tensor.");
  auto paddings_data = GetTensorData<int>(paddings);
  if (!paddings_data)
    return MakeStringError(
        "DnnSetPoolingDescriptor: paddings is not a 1D tensor.");
  if (strides.dtype() != tfrt::DType::I32)
    return MakeStringError(
        "DnnCreatePoolingDescriptor: strides is not an I32 tensor.");
  auto strides_data = GetTensorData<int>(strides);
  if (!strides_data)
    return MakeStringError(
        "DnnCreatePoolingDescriptor: strides is not a 1D tensor.");
  wrapper::DnnNanPropagation cuda_nan_propagation;
  if (current->platform() == wrapper::Platform::CUDA)
    cuda_nan_propagation = static_cast<cudnnNanPropagation_t>(nan_propagation);
  if (auto error = wrapper::DnnSetPoolingDescriptor(
          descriptor->get(), static_cast<wrapper::DnnPoolingMode>(mode),
          cuda_nan_propagation, window_dimensions_data.get(),
          paddings_data.get(), strides_data.get()))
    return std::move(error);
  return std::move(*descriptor);
}

static Expected<GpuDnnTensorDesc> DnnCreateTensorDescriptor(
    const DenseHostTensor& dimensions, const DenseHostTensor& strides,
    Attribute<int32_t> data_type_attr) {
  auto data_type = wrapper::DnnDataType::FromOpaqueValue(*data_type_attr);
  auto descriptor = wrapper::DnnCreateTensorDescriptor(data_type.platform());
  if (!descriptor) return descriptor.takeError();
  if (dimensions.dtype() != tfrt::DType::I32)
    return MakeStringError(
        "DnnCreateTensorDescriptor: dimensions is not an I32 tensor.");
  auto dimensions_data = GetTensorData<int>(dimensions);
  if (!dimensions_data)
    return MakeStringError(
        "DnnCreateTensorDescriptor: dimensions is not a 1D tensor.");
  if (strides.dtype() != tfrt::DType::I32)
    return MakeStringError(
        "DnnCreateTensorDescriptor: strides is not an I32 tensor.");
  auto strides_data = GetTensorData<int>(strides);
  if (!strides_data)
    return MakeStringError(
        "DnnCreateTensorDescriptor: strides is not a 1D tensor.");
  if (auto error = wrapper::DnnSetTensorDescriptor(descriptor->get(), data_type,
                                                   dimensions_data.get(),
                                                   strides_data.get()))
    return std::move(error);
  return std::move(*descriptor);
}

static Error DnnPoolingForward(
    const GpuDnnHandle& handle,
    const wrapper::OwningDnnPoolingDescriptor& pooling_desc, float alpha,
    const GpuDnnTensorDesc& x_desc, const GpuBuffer& x, float beta,
    const GpuDnnTensorDesc& y_desc, const GpuBuffer& y) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();
  wrapper::Pointer<const void> alpha_ptr(&alpha, handle->platform());
  wrapper::Pointer<const void> beta_ptr(&beta, handle->platform());

  return wrapper::DnnPoolingForward(*current, handle.get(), pooling_desc.get(),
                                    alpha_ptr, x_desc.get(), x.pointer(),
                                    beta_ptr, y_desc.get(), y.pointer());
}

static Error DnnPoolingBackward(
    const GpuDnnHandle& handle,
    const wrapper::OwningDnnPoolingDescriptor& pooling_desc, float alpha,
    const GpuDnnTensorDesc& y_desc, const GpuBuffer& y,
    const GpuDnnTensorDesc& dy_desc, const GpuBuffer& dy,
    const GpuDnnTensorDesc& x_desc, const GpuBuffer& x, float beta,
    const GpuDnnTensorDesc& dx_desc, const GpuBuffer& dx) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();
  wrapper::Pointer<const void> alpha_ptr(&alpha, handle->platform());
  wrapper::Pointer<const void> beta_ptr(&beta, handle->platform());

  return wrapper::DnnPoolingBackward(
      *current, handle.get(), pooling_desc.get(), alpha_ptr, y_desc.get(),
      y.pointer(), dy_desc.get(), dy.pointer(), x_desc.get(), x.pointer(),
      beta_ptr, dx_desc.get(), dx.pointer());
}

Error DnnConvolutionForward(
    const GpuDnnHandle& handle, const GpuDnnTensorDesc& x_desc,
    const GpuBuffer& x, const wrapper::OwningDnnFilterDescriptor& w_desc,
    const GpuBuffer& w,
    const wrapper::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const GpuBuffer& work_space, const GpuDnnTensorDesc& y_desc,
    const GpuBuffer& y) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();
  auto algo_dnn = wrapper::DnnConvFwdAlgo(algo, handle->platform());
  return wrapper::DnnConvolutionForward(
      *current, handle.get(), x_desc.get(), x.pointer(), w_desc.get(),
      w.pointer(), conv_desc.get(), algo_dnn, work_space.pointer(),
      work_space.size(), y_desc.get(), y.pointer());
}

Error DnnConvolutionBackwardData(
    const GpuDnnHandle& handle,
    const wrapper::OwningDnnFilterDescriptor& w_desc, const GpuBuffer& w,
    const GpuDnnTensorDesc& dy_desc, const GpuBuffer& dy,
    const wrapper::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const GpuBuffer& work_space, const GpuDnnTensorDesc& dx_desc,
    const GpuBuffer& dx) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();
  auto algo_dnn = wrapper::DnnConvBwdDataAlgo(algo, handle->platform());
  return wrapper::DnnConvolutionBackwardData(
      *current, handle.get(), w_desc.get(), w.pointer(), dy_desc.get(),
      dy.pointer(), conv_desc.get(), algo_dnn, work_space.pointer(),
      work_space.size(), dx_desc.get(), dx.pointer());
}

Error DnnConvolutionBackwardFilter(
    const GpuDnnHandle& handle, const GpuDnnTensorDesc& x_desc,
    const GpuBuffer& x, const GpuDnnTensorDesc& dy_desc, const GpuBuffer& dy,
    const wrapper::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const GpuBuffer& work_space,
    const wrapper::OwningDnnFilterDescriptor& dw_desc, const GpuBuffer& dw) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();
  auto algo_dnn = wrapper::DnnConvBwdWeightsAlgo(algo, handle->platform());
  return wrapper::DnnConvolutionBackwardFilter(
      *current, handle.get(), x_desc.get(), x.pointer(), dy_desc.get(),
      dy.pointer(), conv_desc.get(), algo_dnn, work_space.pointer(),
      work_space.size(), dw_desc.get(), dw.pointer());
}

// This is CUDA specific kernel, there is no ROCm counterpart.
Error CudnnConvolutionBiasActivationForward(
    const GpuDnnHandle& handle, const GpuBuffer& alpha1,
    const GpuDnnTensorDesc& x_desc, const GpuBuffer& x,
    const wrapper::OwningDnnFilterDescriptor& w_desc, const GpuBuffer& w,
    const wrapper::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const GpuBuffer& work_space, const GpuBuffer& alpha2,
    const GpuDnnTensorDesc& z_desc, const GpuBuffer& z,
    const GpuDnnTensorDesc& bias_desc, const GpuBuffer& bias,
    const wrapper::OwningDnnActivationDescriptor& activation_desc,
    const GpuDnnTensorDesc& y_desc, const GpuBuffer& y) {
  auto current = wrapper::CtxSetCurrent(handle.context());
  if (!current) return current.takeError();
  auto algo_dnn = static_cast<cudnnConvolutionFwdAlgo_t>(algo);
  return wrapper::CudnnConvolutionBiasActivationForward(
      *current, handle.get(), alpha1.pointer(), x_desc.get(), x.pointer(),
      w_desc.get(), w.pointer(), conv_desc.get(), algo_dnn,
      work_space.pointer(), work_space.size(), alpha2.pointer(), z_desc.get(),
      z.pointer(), bias_desc.get(), bias.pointer(), activation_desc.get(),
      y_desc.get(), y.pointer());
}

void RegisterGpuDnnKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_gpu.dnn.create", TFRT_KERNEL(DnnCreate));
  kernel_reg->AddKernel("tfrt_gpu.dnn.create_pooling_descriptor",
                        TFRT_KERNEL(DnnCreatePoolingDescriptor));
  kernel_reg->AddKernel("tfrt_gpu.dnn.create_tensor_descriptor",
                        TFRT_KERNEL(DnnCreateTensorDescriptor));
  kernel_reg->AddKernel("tfrt_gpu.dnn.pooling_forward",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(DnnPoolingForward));
  kernel_reg->AddKernel("tfrt_gpu.dnn.pooling_backward",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(DnnPoolingBackward));
  kernel_reg->AddKernel("tfrt_gpu.dnn.convolution_forward",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(DnnConvolutionForward));
  kernel_reg->AddKernel(
      "tfrt_gpu.dnn.convolution_backward_data",
      TFRT_KERNEL_WITH_CHAIN_RESULT(DnnConvolutionBackwardData));
  kernel_reg->AddKernel(
      "tfrt_gpu.dnn.convolution_backward_filter",
      TFRT_KERNEL_WITH_CHAIN_RESULT(DnnConvolutionBackwardFilter));
  kernel_reg->AddKernel(
      "tfrt_gpu.dnn.convolution_bias_activation_forward",
      TFRT_KERNEL_WITH_CHAIN_RESULT(CudnnConvolutionBiasActivationForward));
}

}  // namespace gpu
}  // namespace tfrt
