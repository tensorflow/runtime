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

#include "tfrt/gpu/stream/cudnn_wrapper.h"
#include "tfrt/gpu/stream/dnn_wrapper.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"
#include "tfrt/tensor/tensor_type_registration.h"

namespace tfrt {
namespace cuda {

// Overloaded helpers for the REPORT_ERROR macro. Allows the macro
// to use either strings or llvm::Errors.
static void ReportErrorInternal(KernelErrorHandler error_handler,
                                string_view error_message, string_view file,
                                int line) {
  return error_handler.ReportError(file, ':', line, ' ', error_message);
}

static void ReportErrorInternal(KernelErrorHandler error_handler, Error error,
                                string_view file, int line) {
  llvm::handleAllErrors(std::move(error), [&](const llvm::ErrorInfoBase& info) {
    ReportErrorInternal(error_handler, info.message(), file, line);
  });
}

#define REPORT_ERROR(error_handler, error) \
  ReportErrorInternal(error_handler, error, __FILE__, __LINE__)

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

static void DnnCreate(Argument<gpu::stream::Context> context,
                      Argument<Chain> in_chain,
                      Result<gpu::stream::OwningDnnHandle> dnn_handle,
                      Result<Chain> out_chain, KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  auto handle = gpu::stream::DnnCreate(*current);
  if (!handle) return REPORT_ERROR(handler, handle.takeError());
  dnn_handle.Emplace(std::move(*handle));
  out_chain.Set(in_chain);
}

static void DnnDestroy(Argument<gpu::stream::OwningDnnHandle> dnn_handle,
                       Argument<Chain> in_chain, Result<Chain> out_chain,
                       KernelErrorHandler handler) {
  if (auto error = gpu::stream::DnnDestroy(dnn_handle->get()))
    return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

static void DnnSetStream(Argument<gpu::stream::OwningDnnHandle> dnn_handle,
                         Argument<gpu::stream::OwningStream> stream,
                         Argument<Chain> in_chain, Result<Chain> out_chain,
                         KernelErrorHandler handler) {
  if (auto error = gpu::stream::DnnSetStream(dnn_handle->get(), stream->get()))
    return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

static void DnnGetStream(Argument<gpu::stream::OwningDnnHandle> dnn_handle,
                         Argument<Chain> in_chain,
                         Result<gpu::stream::Stream> out_stream,
                         Result<Chain> out_chain, KernelErrorHandler handler) {
  auto stream = gpu::stream::DnnGetStream(dnn_handle->get());
  if (!stream) return REPORT_ERROR(handler, stream.takeError());
  out_stream.Emplace(std::move(*stream));
  out_chain.Set(in_chain);
}

static void DnnCreatePoolingDescriptor(
    Argument<gpu::stream::Context> context, Argument<uint32_t> mode,
    Argument<uint32_t> nan_propagation,
    const DenseHostTensor& window_dimensions, const DenseHostTensor& paddings,
    const DenseHostTensor& strides, Argument<Chain> in_chain,
    Result<gpu::stream::OwningDnnPoolingDescriptor> dnn_pooling_descriptor,
    Result<Chain> out_chain, KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  auto descriptor =
      gpu::stream::DnnCreatePoolingDescriptor(current->platform());
  if (!descriptor) return REPORT_ERROR(handler, descriptor.takeError());
  if (window_dimensions.dtype().kind() != tfrt::DType::I32)
    return REPORT_ERROR(
        handler,
        "DnnCreatePoolingDescriptor: window_dimensions is not an I32 tensor.");
  auto window_dimensions_data = GetTensorData<int>(window_dimensions);
  if (!window_dimensions_data)
    return REPORT_ERROR(
        handler,
        "DnnCreatePoolingDescriptor: window_dimensions is not a 1D tensor.");
  if (paddings.dtype().kind() != tfrt::DType::I32)
    return REPORT_ERROR(
        handler, "DnnSetPoolingDescriptor: paddings is not an I32 tensor.");
  auto paddings_data = GetTensorData<int>(paddings);
  if (!paddings_data)
    return REPORT_ERROR(
        handler, "DnnSetPoolingDescriptor: paddings is not a 1D tensor.");
  if (strides.dtype().kind() != tfrt::DType::I32)
    return REPORT_ERROR(
        handler, "DnnCreatePoolingDescriptor: strides is not an I32 tensor.");
  auto strides_data = GetTensorData<int>(strides);
  if (!strides_data)
    return REPORT_ERROR(
        handler, "DnnCreatePoolingDescriptor: strides is not a 1D tensor.");
  if (auto error = gpu::stream::DnnSetPoolingDescriptor(
          *current, descriptor.get().get(),
          IntToDnnPoolingMode(mode.get()).get(),
          IntToDnnNanPropagation(nan_propagation.get()).get(),
          window_dimensions_data.get(), paddings_data.get(),
          strides_data.get()))
    return REPORT_ERROR(handler, std::move(error));
  dnn_pooling_descriptor.Emplace(std::move(*descriptor));
  out_chain.Set(in_chain);
}

static void DnnDestroyPoolingDescriptor(
    Argument<gpu::stream::OwningDnnPoolingDescriptor> descriptor,
    Argument<Chain> in_chain, Result<Chain> out_chain,
    KernelErrorHandler handler) {
  if (auto error = gpu::stream::DnnDestroyPoolingDescriptor(descriptor->get()))
    return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

static void DnnCreateTensorDescriptor(
    Argument<gpu::stream::Context> context, Argument<uint32_t> data_type,
    const DenseHostTensor& dimensions, const DenseHostTensor& strides,
    Argument<Chain> in_chain,
    Result<gpu::stream::OwningDnnTensorDescriptor> dnn_tensor_descriptor,
    Result<Chain> out_chain, KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  auto descriptor = gpu::stream::DnnCreateTensorDescriptor(current->platform());
  if (!descriptor) return REPORT_ERROR(handler, descriptor.takeError());
  if (dimensions.dtype().kind() != tfrt::DType::I32)
    return REPORT_ERROR(
        handler, "DnnCreateTensorDescriptor: dimensions is not an I32 tensor.");
  auto dimensions_data = GetTensorData<int>(dimensions);
  if (!dimensions_data)
    return REPORT_ERROR(
        handler, "DnnCreateTensorDescriptor: dimensions is not a 1D tensor.");
  if (strides.dtype().kind() != tfrt::DType::I32)
    return REPORT_ERROR(
        handler, "DnnCreateTensorDescriptor: strides is not an I32 tensor.");
  auto strides_data = GetTensorData<int>(strides);
  if (!strides_data)
    return REPORT_ERROR(
        handler, "DnnCreateTensorDescriptor: strides is not a 1D tensor.");
  gpu::stream::DnnDataType dnn_data_type(data_type.get(),
                                         context.get().platform());
  if (auto error = gpu::stream::DnnSetTensorDescriptor(
          descriptor.get().get(), dnn_data_type, dimensions_data.get(),
          strides_data.get()))
    return REPORT_ERROR(handler, std::move(error));
  dnn_tensor_descriptor.Emplace(std::move(*descriptor));
  out_chain.Set(in_chain);
}

static void DnnDestroyTensorDescriptor(
    Argument<gpu::stream::OwningDnnTensorDescriptor> descriptor,
    Argument<Chain> in_chain, Result<Chain> out_chain,
    KernelErrorHandler handler) {
  if (auto error = gpu::stream::DnnDestroyTensorDescriptor(descriptor->get()))
    return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

static void DnnPoolingForward(
    Argument<gpu::stream::Context> context,
    Argument<gpu::stream::OwningDnnHandle> handle,
    Argument<gpu::stream::OwningDnnPoolingDescriptor> pooling_desc,
    Argument<float> alpha,
    Argument<gpu::stream::OwningDnnTensorDescriptor> x_desc,
    Argument<RCReference<gpu::GpuBuffer>> x, Argument<float> beta,
    Argument<const gpu::stream::OwningDnnTensorDescriptor> y_desc,
    Argument<RCReference<gpu::GpuBuffer>> y, Argument<Chain> in_chain,
    Result<Chain> out_chain, KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  Pointer<const void> alpha_ptr(&(*alpha), context->platform());
  Pointer<const void> beta_ptr(&(*beta), context->platform());

  if (auto error = tfrt::gpu::stream::DnnPoolingForward(
          *current, handle->get(), pooling_desc->get(), alpha_ptr,
          x_desc->get(), Pointer<const void>(x->get()->pointer()), beta_ptr,
          y_desc->get(), Pointer<void>(y->get()->pointer())))
    return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

static void DnnPoolingBackward(
    Argument<gpu::stream::Context> context,
    Argument<gpu::stream::OwningDnnHandle> handle,
    Argument<gpu::stream::OwningDnnPoolingDescriptor> pooling_desc,
    Argument<float> alpha,
    Argument<gpu::stream::OwningDnnTensorDescriptor> y_desc,
    Argument<RCReference<gpu::GpuBuffer>> y,
    Argument<const gpu::stream::OwningDnnTensorDescriptor> dy_desc,
    Argument<RCReference<gpu::GpuBuffer>> dy,
    Argument<gpu::stream::OwningDnnTensorDescriptor> x_desc,
    Argument<RCReference<gpu::GpuBuffer>> x, Argument<float> beta,
    Argument<const gpu::stream::OwningDnnTensorDescriptor> dx_desc,
    Argument<RCReference<gpu::GpuBuffer>> dx, Argument<Chain> in_chain,
    Result<Chain> out_chain, KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  Pointer<const void> alpha_ptr(&*alpha, context->platform());
  Pointer<const void> beta_ptr(&(*beta), context->platform());

  if (auto error = gpu::stream::DnnPoolingBackward(
          *current, handle->get(), pooling_desc->get(), alpha_ptr,
          y_desc->get(), Pointer<const void>(y->get()->pointer()),
          dy_desc->get(), Pointer<const void>(dy->get()->pointer()),
          x_desc->get(), Pointer<const void>(x->get()->pointer()), beta_ptr,
          dx_desc->get(), Pointer<void>(dx->get()->pointer())))
    return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

llvm::Expected<std::tuple<>> DnnConvolutionForward(
    gpu::stream::Context context, const gpu::stream::OwningDnnHandle& handle,
    const gpu::stream::OwningDnnTensorDescriptor& x_desc,
    const RCReference<gpu::GpuBuffer>& x,
    const gpu::stream::OwningDnnFilterDescriptor& w_desc,
    const RCReference<gpu::GpuBuffer>& w,
    const gpu::stream::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const RCReference<gpu::GpuBuffer>& work_space,
    const gpu::stream::OwningDnnTensorDescriptor& y_desc,
    const RCReference<gpu::GpuBuffer>& y) {
  auto current = gpu::stream::CtxSetCurrent(context);
  if (!current) return current.takeError();
  auto algo_dnn = gpu::stream::DnnConvFwdAlgo(algo, current->platform());
  return gpu::stream::DnnConvolutionForward(
      *current, handle.get(), x_desc.get(), x->pointer(), w_desc.get(),
      w->pointer(), conv_desc.get(), algo_dnn, work_space->pointer(),
      work_space->size(), y_desc.get(), y->pointer());
}

llvm::Expected<std::tuple<>> DnnConvolutionBackwardData(
    gpu::stream::Context context, const gpu::stream::OwningDnnHandle& handle,
    const gpu::stream::OwningDnnFilterDescriptor& w_desc,
    const RCReference<gpu::GpuBuffer>& w,
    const gpu::stream::OwningDnnTensorDescriptor& dy_desc,
    const RCReference<gpu::GpuBuffer>& dy,
    const gpu::stream::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const RCReference<gpu::GpuBuffer>& work_space,
    const gpu::stream::OwningDnnTensorDescriptor& dx_desc,
    const RCReference<gpu::GpuBuffer>& dx) {
  auto current = gpu::stream::CtxSetCurrent(context);
  if (!current) return current.takeError();
  auto algo_dnn = gpu::stream::DnnConvBwdDataAlgo(algo, current->platform());
  return gpu::stream::DnnConvolutionBackwardData(
      *current, handle.get(), w_desc.get(), w->pointer(), dy_desc.get(),
      dy->pointer(), conv_desc.get(), algo_dnn, work_space->pointer(),
      work_space->size(), dx_desc.get(), dx->pointer());
}

llvm::Expected<std::tuple<>> DnnConvolutionBackwardFilter(
    gpu::stream::Context context, const gpu::stream::OwningDnnHandle& handle,
    const gpu::stream::OwningDnnTensorDescriptor& x_desc,
    const RCReference<gpu::GpuBuffer>& x,
    const gpu::stream::OwningDnnTensorDescriptor& dy_desc,
    const RCReference<gpu::GpuBuffer>& dy,
    const gpu::stream::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const RCReference<gpu::GpuBuffer>& work_space,
    const gpu::stream::OwningDnnFilterDescriptor& dw_desc,
    const RCReference<gpu::GpuBuffer>& dw) {
  auto current = gpu::stream::CtxSetCurrent(context);
  if (!current) return current.takeError();
  auto algo_dnn = gpu::stream::DnnConvBwdWeightsAlgo(algo, current->platform());
  return gpu::stream::DnnConvolutionBackwardFilter(
      *current, handle.get(), x_desc.get(), x->pointer(), dy_desc.get(),
      dy->pointer(), conv_desc.get(), algo_dnn, work_space->pointer(),
      work_space->size(), dw_desc.get(), dw->pointer());
}

// This is CUDA specific kernel, there is no ROCm counterpart.
llvm::Expected<std::tuple<>> CudnnConvolutionBiasActivationForward(
    gpu::stream::Context context, const gpu::stream::OwningDnnHandle& handle,
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
  auto current = gpu::stream::CtxSetCurrent(context);
  if (!current) return current.takeError();
  auto algo_dnn = static_cast<cudnnConvolutionFwdAlgo_t>(algo);
  return gpu::stream::CudnnConvolutionBiasActivationForward(
      *current, handle.get(), alpha1->pointer(), x_desc.get(), x->pointer(),
      w_desc.get(), w->pointer(), conv_desc.get(), algo_dnn,
      work_space->pointer(), work_space->size(), alpha2->pointer(),
      z_desc.get(), z->pointer(), bias_desc.get(), bias->pointer(),
      activation_desc.get(), y_desc.get(), y->pointer());
}

void RegisterCudaDnnKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_cuda.dnn.create", TFRT_KERNEL(DnnCreate));
  kernel_reg->AddKernel("tfrt_cuda.dnn.destroy", TFRT_KERNEL(DnnDestroy));
  kernel_reg->AddKernel("tfrt_cuda.dnn.set_stream", TFRT_KERNEL(DnnSetStream));
  kernel_reg->AddKernel("tfrt_cuda.dnn.get_stream", TFRT_KERNEL(DnnGetStream));
  kernel_reg->AddKernel("tfrt_cuda.dnn.create_pooling_descriptor",
                        TFRT_KERNEL(DnnCreatePoolingDescriptor));
  kernel_reg->AddKernel("tfrt_cuda.dnn.destroy_pooling_descriptor",
                        TFRT_KERNEL(DnnDestroyPoolingDescriptor));
  kernel_reg->AddKernel("tfrt_cuda.dnn.create_tensor_descriptor",
                        TFRT_KERNEL(DnnCreateTensorDescriptor));
  kernel_reg->AddKernel("tfrt_cuda.dnn.destroy_tensor_descriptor",
                        TFRT_KERNEL(DnnDestroyTensorDescriptor));
  kernel_reg->AddKernel("tfrt_cuda.dnn.pooling_forward",
                        TFRT_KERNEL(DnnPoolingForward));
  kernel_reg->AddKernel("tfrt_cuda.dnn.pooling_backward",
                        TFRT_KERNEL(DnnPoolingBackward));
  kernel_reg->AddKernel("tfrt_cuda.dnn.convolution_forward",
                        TFRT_KERNEL(DnnConvolutionForward));
  kernel_reg->AddKernel("tfrt_cuda.dnn.convolution_backward_data",
                        TFRT_KERNEL(DnnConvolutionBackwardData));
  kernel_reg->AddKernel("tfrt_cuda.dnn.convolution_backward_filter",
                        TFRT_KERNEL(DnnConvolutionBackwardFilter));
  kernel_reg->AddKernel("tfrt_cuda.dnn.convolution_bias_activation_forward",
                        TFRT_KERNEL(CudnnConvolutionBiasActivationForward));
}

}  // namespace cuda
}  // namespace tfrt
