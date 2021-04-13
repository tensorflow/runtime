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

//===- dnn_kernels.cc - CUDA runtime interface ----------------------------===//
//
// This file defines the C++ functions that implement the kernels provided by
// the TFRT CUDA runtime.
#include <cstdint>
#include <memory>
#include <string>

#include "kernels.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/stream/cudnn_wrapper.h"
#include "tfrt/gpu/stream/dnn_wrapper.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"
#include "tfrt/tensor/tensor_type_registration.h"

namespace tfrt {
namespace cuda {

template <typename T>
llvm::Expected<ArrayRef<T>> GetTensorData(const DenseHostTensor& t) {
  if (t.shape().GetRank() != 1) {
    return MakeStringError(
        "GetTensorData: input tensor is not a rank 1 tensor");
  }
  if (t.dtype() != GetDType<T>())
    return tfrt::MakeStringError(
        "GetTensorData: input tensor type mismatch with desired vector type.");
  return ArrayRef<T>(static_cast<const T*>(t.data()), t.NumElements());
}

// Casting UI32 from MLIR to the proper DNN enumerator typ.
llvm::Expected<gpu::stream::DnnPoolingMode> IntToDnnPoolingMode(uint32_t mode) {
  switch (mode) {
    case 0:
      return gpu::stream::DnnPoolingMode::kPoolingMax;
    case 1:
      return gpu::stream::DnnPoolingMode::kPoolingAverageCountIncludePadding;
    case 2:
      return gpu::stream::DnnPoolingMode::kPoolingAverageCountExcludePadding;
    case 3:
      return gpu::stream::DnnPoolingMode::kPoolingMaxDeterministic;
    default:
      return MakeStringError("UI32 mode out of range for enum cast");
  }
}

llvm::Expected<gpu::stream::DnnNanPropagation> IntToDnnNanPropagation(
    uint32_t nan_propagation) {
  switch (nan_propagation) {
    case 0:
      return gpu::stream::DnnNanPropagation::kNotPropagateNan;
    case 1:
      return gpu::stream::DnnNanPropagation::kPropagateNan;
    default:
      return MakeStringError("UI32 nan_propagation out of range for enum cast");
  }
}

using ::tfrt::gpu::stream::Pointer;

static Expected<gpu::stream::OwningDnnHandle> DnnCreate(
    const gpu::GpuContext& context) {
  auto current = gpu::stream::CtxSetCurrent(context.get());
  if (!current) return current.takeError();
  return gpu::stream::DnnCreate(*current);
}

static Error DnnDestroy(Argument<gpu::stream::OwningDnnHandle> dnn_handle) {
  return gpu::stream::DnnDestroy(dnn_handle->release());
}

static Error DnnSetStream(const gpu::stream::OwningDnnHandle& dnn_handle,
                          const gpu::GpuStream& stream) {
  return gpu::stream::DnnSetStream(dnn_handle.get(), stream.get());
}

static Expected<gpu::stream::Stream> DnnGetStream(
    const gpu::stream::OwningDnnHandle& dnn_handle) {
  return gpu::stream::DnnGetStream(dnn_handle.get());
}

static Expected<gpu::stream::OwningDnnPoolingDescriptor>
DnnCreatePoolingDescriptor(const gpu::GpuContext& context, uint32_t mode,
                           uint32_t nan_propagation,
                           const DenseHostTensor& window_dimensions,
                           const DenseHostTensor& paddings,
                           const DenseHostTensor& strides) {
  auto current = gpu::stream::CtxSetCurrent(context.get());
  auto descriptor =
      gpu::stream::DnnCreatePoolingDescriptor(context->platform());
  if (!descriptor) return descriptor.takeError();
  if (window_dimensions.dtype().kind() != tfrt::DType::I32)
    return MakeStringError(
        "DnnCreatePoolingDescriptor: window_dimensions is not an I32 tensor.");
  auto window_dimensions_data = GetTensorData<int>(window_dimensions);
  if (!window_dimensions_data)
    return MakeStringError(
        "DnnCreatePoolingDescriptor: window_dimensions is not a 1D tensor.");
  if (paddings.dtype().kind() != tfrt::DType::I32)
    return MakeStringError(
        "DnnSetPoolingDescriptor: paddings is not an I32 tensor.");
  auto paddings_data = GetTensorData<int>(paddings);
  if (!paddings_data)
    return MakeStringError(
        "DnnSetPoolingDescriptor: paddings is not a 1D tensor.");
  if (strides.dtype().kind() != tfrt::DType::I32)
    return MakeStringError(
        "DnnCreatePoolingDescriptor: strides is not an I32 tensor.");
  auto strides_data = GetTensorData<int>(strides);
  if (!strides_data)
    return MakeStringError(
        "DnnCreatePoolingDescriptor: strides is not a 1D tensor.");
  if (auto error = gpu::stream::DnnSetPoolingDescriptor(
          *current, descriptor.get().get(), IntToDnnPoolingMode(mode).get(),
          IntToDnnNanPropagation(nan_propagation).get(),
          window_dimensions_data.get(), paddings_data.get(),
          strides_data.get()))
    return std::move(error);
  return std::move(*descriptor);
}

static Error DnnDestroyPoolingDescriptor(
    Argument<gpu::stream::OwningDnnPoolingDescriptor> descriptor) {
  return gpu::stream::DnnDestroyPoolingDescriptor(descriptor->release());
}

static Expected<gpu::stream::OwningDnnTensorDescriptor>
DnnCreateTensorDescriptor(const gpu::GpuContext& context, uint32_t data_type,
                          const DenseHostTensor& dimensions,
                          const DenseHostTensor& strides) {
  auto current = gpu::stream::CtxSetCurrent(context.get());
  auto descriptor = gpu::stream::DnnCreateTensorDescriptor(context->platform());
  if (!descriptor) return descriptor.takeError();
  if (dimensions.dtype().kind() != tfrt::DType::I32)
    return MakeStringError(
        "DnnCreateTensorDescriptor: dimensions is not an I32 tensor.");
  auto dimensions_data = GetTensorData<int>(dimensions);
  if (!dimensions_data)
    return MakeStringError(
        "DnnCreateTensorDescriptor: dimensions is not a 1D tensor.");
  if (strides.dtype().kind() != tfrt::DType::I32)
    return MakeStringError(
        "DnnCreateTensorDescriptor: strides is not an I32 tensor.");
  auto strides_data = GetTensorData<int>(strides);
  if (!strides_data)
    return MakeStringError(
        "DnnCreateTensorDescriptor: strides is not a 1D tensor.");
  gpu::stream::DnnDataType dnn_data_type(static_cast<int>(data_type),
                                         context->platform());
  if (auto error = gpu::stream::DnnSetTensorDescriptor(
          descriptor->get(), dnn_data_type, dimensions_data.get(),
          strides_data.get()))
    return std::move(error);
  return std::move(*descriptor);
}

static Error DnnDestroyTensorDescriptor(
    Argument<gpu::stream::OwningDnnTensorDescriptor> descriptor) {
  return gpu::stream::DnnDestroyTensorDescriptor(descriptor->release());
}

static Error DnnPoolingForward(
    const gpu::GpuContext& context, const gpu::stream::OwningDnnHandle& handle,
    const gpu::stream::OwningDnnPoolingDescriptor& pooling_desc, float alpha,
    const gpu::stream::OwningDnnTensorDescriptor& x_desc,
    const RCReference<gpu::GpuBuffer>& x, float beta,
    const gpu::stream::OwningDnnTensorDescriptor& y_desc,
    const RCReference<gpu::GpuBuffer>& y) {
  auto current = gpu::stream::CtxSetCurrent(context.get());
  if (!current) return current.takeError();
  Pointer<const void> alpha_ptr(&alpha, context->platform());
  Pointer<const void> beta_ptr(&beta, context->platform());

  return tfrt::gpu::stream::DnnPoolingForward(
      *current, handle.get(), pooling_desc.get(), alpha_ptr, x_desc.get(),
      x->pointer(), beta_ptr, y_desc.get(), y->pointer());
}

static Error DnnPoolingBackward(
    const gpu::GpuContext& context, const gpu::stream::OwningDnnHandle& handle,
    const gpu::stream::OwningDnnPoolingDescriptor& pooling_desc, float alpha,
    const gpu::stream::OwningDnnTensorDescriptor& y_desc,
    const RCReference<gpu::GpuBuffer>& y,
    const gpu::stream::OwningDnnTensorDescriptor& dy_desc,
    const RCReference<gpu::GpuBuffer>& dy,
    const gpu::stream::OwningDnnTensorDescriptor& x_desc,
    const RCReference<gpu::GpuBuffer>& x, float beta,
    const gpu::stream::OwningDnnTensorDescriptor& dx_desc,
    const RCReference<gpu::GpuBuffer>& dx) {
  auto current = gpu::stream::CtxSetCurrent(context.get());
  if (!current) return current.takeError();
  Pointer<const void> alpha_ptr(&alpha, context->platform());
  Pointer<const void> beta_ptr(&beta, context->platform());

  return gpu::stream::DnnPoolingBackward(
      *current, handle.get(), pooling_desc.get(), alpha_ptr, y_desc.get(),
      y->pointer(), dy_desc.get(), dy->pointer(), x_desc.get(), x->pointer(),
      beta_ptr, dx_desc.get(), dx->pointer());
}

Error DnnConvolutionForward(
    const gpu::GpuContext& context, const gpu::stream::OwningDnnHandle& handle,
    const gpu::stream::OwningDnnTensorDescriptor& x_desc,
    const RCReference<gpu::GpuBuffer>& x,
    const gpu::stream::OwningDnnFilterDescriptor& w_desc,
    const RCReference<gpu::GpuBuffer>& w,
    const gpu::stream::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const RCReference<gpu::GpuBuffer>& work_space,
    const gpu::stream::OwningDnnTensorDescriptor& y_desc,
    const RCReference<gpu::GpuBuffer>& y) {
  auto current = gpu::stream::CtxSetCurrent(context.get());
  if (!current) return current.takeError();
  auto algo_dnn = gpu::stream::DnnConvFwdAlgo(algo, context->platform());
  return gpu::stream::DnnConvolutionForward(
      *current, handle.get(), x_desc.get(), x->pointer(), w_desc.get(),
      w->pointer(), conv_desc.get(), algo_dnn, work_space->pointer(),
      work_space->size(), y_desc.get(), y->pointer());
}

Error DnnConvolutionBackwardData(
    const gpu::GpuContext& context, const gpu::stream::OwningDnnHandle& handle,
    const gpu::stream::OwningDnnFilterDescriptor& w_desc,
    const RCReference<gpu::GpuBuffer>& w,
    const gpu::stream::OwningDnnTensorDescriptor& dy_desc,
    const RCReference<gpu::GpuBuffer>& dy,
    const gpu::stream::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const RCReference<gpu::GpuBuffer>& work_space,
    const gpu::stream::OwningDnnTensorDescriptor& dx_desc,
    const RCReference<gpu::GpuBuffer>& dx) {
  auto current = gpu::stream::CtxSetCurrent(context.get());
  if (!current) return current.takeError();
  auto algo_dnn = gpu::stream::DnnConvBwdDataAlgo(algo, context->platform());
  return gpu::stream::DnnConvolutionBackwardData(
      *current, handle.get(), w_desc.get(), w->pointer(), dy_desc.get(),
      dy->pointer(), conv_desc.get(), algo_dnn, work_space->pointer(),
      work_space->size(), dx_desc.get(), dx->pointer());
}

Error DnnConvolutionBackwardFilter(
    const gpu::GpuContext& context, const gpu::stream::OwningDnnHandle& handle,
    const gpu::stream::OwningDnnTensorDescriptor& x_desc,
    const RCReference<gpu::GpuBuffer>& x,
    const gpu::stream::OwningDnnTensorDescriptor& dy_desc,
    const RCReference<gpu::GpuBuffer>& dy,
    const gpu::stream::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const RCReference<gpu::GpuBuffer>& work_space,
    const gpu::stream::OwningDnnFilterDescriptor& dw_desc,
    const RCReference<gpu::GpuBuffer>& dw) {
  auto current = gpu::stream::CtxSetCurrent(context.get());
  if (!current) return current.takeError();
  auto algo_dnn = gpu::stream::DnnConvBwdWeightsAlgo(algo, context->platform());
  return gpu::stream::DnnConvolutionBackwardFilter(
      *current, handle.get(), x_desc.get(), x->pointer(), dy_desc.get(),
      dy->pointer(), conv_desc.get(), algo_dnn, work_space->pointer(),
      work_space->size(), dw_desc.get(), dw->pointer());
}

// This is CUDA specific kernel, there is no ROCm counterpart.
Error CudnnConvolutionBiasActivationForward(
    const gpu::GpuContext& context, const gpu::stream::OwningDnnHandle& handle,
    const RCReference<gpu::GpuBuffer>& alpha1,
    const gpu::stream::OwningDnnTensorDescriptor& x_desc,
    const RCReference<gpu::GpuBuffer>& x,
    const gpu::stream::OwningDnnFilterDescriptor& w_desc,
    const RCReference<gpu::GpuBuffer>& w,
    const gpu::stream::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const RCReference<gpu::GpuBuffer>& work_space,
    const RCReference<gpu::GpuBuffer>& alpha2,
    const gpu::stream::OwningDnnTensorDescriptor& z_desc,
    const RCReference<gpu::GpuBuffer>& z,
    const gpu::stream::OwningDnnTensorDescriptor& bias_desc,
    const RCReference<gpu::GpuBuffer>& bias,
    const gpu::stream::OwningDnnActivationDescriptor& activation_desc,
    const gpu::stream::OwningDnnTensorDescriptor& y_desc,
    const RCReference<gpu::GpuBuffer>& y) {
  auto current = gpu::stream::CtxSetCurrent(context.get());
  if (!current) return current.takeError();
  auto algo_dnn = static_cast<cudnnConvolutionFwdAlgo_t>(algo);
  return gpu::stream::CudnnConvolutionBiasActivationForward(
      *current, handle.get(), alpha1->pointer(), x_desc.get(), x->pointer(),
      w_desc.get(), w->pointer(), conv_desc.get(), algo_dnn,
      work_space->pointer(), work_space->size(), alpha2->pointer(),
      z_desc.get(), z->pointer(), bias_desc.get(), bias->pointer(),
      activation_desc.get(), y_desc.get(), y->pointer());
}

#define TFRT_WITH_CHAIN_RESULT(sync_func) \
  internal::WithChainResult<decltype(&sync_func), &sync_func>::Invoke

void RegisterCudaDnnKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_cuda.dnn.create", TFRT_KERNEL(DnnCreate));
  kernel_reg->AddKernel("tfrt_cuda.dnn.destroy",
                        TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(DnnDestroy)));
  kernel_reg->AddKernel("tfrt_cuda.dnn.set_stream",
                        TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(DnnSetStream)));
  kernel_reg->AddKernel("tfrt_cuda.dnn.get_stream", TFRT_KERNEL(DnnGetStream));
  kernel_reg->AddKernel("tfrt_cuda.dnn.create_pooling_descriptor",
                        TFRT_KERNEL(DnnCreatePoolingDescriptor));
  kernel_reg->AddKernel(
      "tfrt_cuda.dnn.destroy_pooling_descriptor",
      TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(DnnDestroyPoolingDescriptor)));
  kernel_reg->AddKernel("tfrt_cuda.dnn.create_tensor_descriptor",
                        TFRT_KERNEL(DnnCreateTensorDescriptor));
  kernel_reg->AddKernel(
      "tfrt_cuda.dnn.destroy_tensor_descriptor",
      TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(DnnDestroyTensorDescriptor)));
  kernel_reg->AddKernel("tfrt_cuda.dnn.pooling_forward",
                        TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(DnnPoolingForward)));
  kernel_reg->AddKernel(
      "tfrt_cuda.dnn.pooling_backward",
      TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(DnnPoolingBackward)));
  kernel_reg->AddKernel(
      "tfrt_cuda.dnn.convolution_forward",
      TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(DnnConvolutionForward)));
  kernel_reg->AddKernel(
      "tfrt_cuda.dnn.convolution_backward_data",
      TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(DnnConvolutionBackwardData)));
  kernel_reg->AddKernel(
      "tfrt_cuda.dnn.convolution_backward_filter",
      TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(DnnConvolutionBackwardFilter)));
  kernel_reg->AddKernel("tfrt_cuda.dnn.convolution_bias_activation_forward",
                        TFRT_KERNEL(TFRT_WITH_CHAIN_RESULT(
                            CudnnConvolutionBiasActivationForward)));
}

}  // namespace cuda
}  // namespace tfrt
