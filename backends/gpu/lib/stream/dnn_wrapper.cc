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

//===- dnn_wrapper.cc -------------------------------------------*- C++ -*-===//
//
// Thin abstraction layer for cuDNN and MIOpen.
//
//===----------------------------------------------------------------------===//
#include "tfrt/gpu/stream/dnn_wrapper.h"

#include "library_types.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/stream/cudnn_wrapper.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace stream {

// Convert DNN wrapper enums to cuDNN enums.
static cudnnDataType_t ToCuda(DnnDataType data_type) {
  switch (data_type) {
    case DnnDataType::kFloat:
      return CUDNN_DATA_FLOAT;
    case DnnDataType::kDouble:
      return CUDNN_DATA_DOUBLE;
    case DnnDataType::kHalf:
      return CUDNN_DATA_HALF;
    case DnnDataType::kInt8:
      return CUDNN_DATA_INT8;
    case DnnDataType::kInt32:
      return CUDNN_DATA_INT32;
    case DnnDataType::kInt8x4:
      return CUDNN_DATA_INT8x4;
    case DnnDataType::kUint8:
      return CUDNN_DATA_UINT8;
    case DnnDataType::kUint8x4:
      return CUDNN_DATA_UINT8x4;
    case DnnDataType::kInt8x32:
      return CUDNN_DATA_INT8x32;
  }
  llvm_unreachable(StrCat("Unrecognized DnnDataType: ", data_type).c_str());
}

static cudnnPoolingMode_t ToCuda(DnnPoolingMode mode) {
  switch (mode) {
    case DnnPoolingMode::kPoolingMax:
      return CUDNN_POOLING_MAX;
    case DnnPoolingMode::kPoolingAverageCountIncludePadding:
      return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    case DnnPoolingMode::kPoolingAverageCountExcludePadding:
      return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    case DnnPoolingMode::kPoolingMaxDeterministic:
      return CUDNN_POOLING_MAX_DETERMINISTIC;
  }
  llvm_unreachable(StrCat("Unrecognized DnnPoolingMode mode: ", mode).c_str());
}

static cudnnNanPropagation_t ToCuda(DnnNanPropagation nan) {
  switch (nan) {
    case DnnNanPropagation::kNotPropagateNan:
      return CUDNN_NOT_PROPAGATE_NAN;
    case DnnNanPropagation::kPropagateNan:
      return CUDNN_PROPAGATE_NAN;
  }
  llvm_unreachable(StrCat("Unrecognized DnnNanPropagation nan: ", nan).c_str());
}

static auto ToCuda(DnnTensorDescriptor desc) {
  return static_cast<cudnnTensorDescriptor_t>(desc);
}

static auto ToCuda(DnnConvolutionDescriptor desc) {
  return static_cast<cudnnConvolutionDescriptor_t>(desc);
}

static auto ToCuda(DnnFilterDescriptor desc) {
  return static_cast<cudnnFilterDescriptor_t>(desc);
}

static auto ToCuda(DnnActivationDescriptor desc) {
  return static_cast<cudnnActivationDescriptor_t>(desc);
}

static constexpr auto ToCuda(DnnBatchNormMode mode) {
  return static_cast<cudnnBatchNormMode_t>(mode);
}

// Assume that the tensor descriptor array has a small size of this constant
// than it is possible use more efficient llvm::SmallVector instead
// of std::vector
static const int kTensorDescriptorArraySize = 16;
//  Helper function to convert ArrayRef’s in Dnn wrapper to ArrayRef’s (vectors)
//  to be used with Cudnn
llvm::SmallVector<cudnnTensorDescriptor_t, kTensorDescriptorArraySize> ToCuda(
    llvm::ArrayRef<DnnTensorDescriptor> dnn_descriptors) {
  llvm::SmallVector<cudnnTensorDescriptor_t, kTensorDescriptorArraySize>
      cudnn_descriptors;
  cudnn_descriptors.reserve(dnn_descriptors.size());
  copy(dnn_descriptors, std::back_inserter(cudnn_descriptors));
  return cudnn_descriptors;
}

static constexpr auto ToDnn(cudnnDataType_t data_type) {
  return static_cast<DnnDataType>(data_type);
}

static DnnTensorDescriptorData ToDnn(CudnnTensorDescriptorData data) {
  DnnTensorDescriptorData dnn_data;
  dnn_data.data_type = ToDnn(data.data_type);
  dnn_data.dimensions = data.dimensions;
  dnn_data.strides = data.strides;
  return dnn_data;
}

void internal::DnnHandleDeleter::operator()(DnnHandle handle) const {
  LogIfError(DnnDestroy(handle));
}
void internal::DnnTensorDescriptorDeleter::operator()(
    DnnTensorDescriptor descriptor) const {
  LogIfError(DnnDestroyTensorDescriptor(descriptor));
}
void internal::DnnConvolutionDescriptorDeleter::operator()(
    DnnConvolutionDescriptor descriptor) const {
  LogIfError(DnnDestroyConvolutionDescriptor(descriptor));
}
void internal::DnnPoolingDescriptorDeleter::operator()(
    DnnPoolingDescriptor descriptor) const {
  LogIfError(DnnDestroyPoolingDescriptor(descriptor));
}
void internal::DnnActivationDescriptorDeleter::operator()(
    DnnActivationDescriptor descriptor) const {
  LogIfError(DnnDestroyActivationDescriptor(descriptor));
}
void internal::DnnFilterDescriptorDeleter::operator()(
    DnnFilterDescriptor descriptor) const {
  LogIfError(DnnDestroyFilterDescriptor(descriptor));
}
void internal::DnnDropoutDescriptorDeleter::operator()(
    DnnDropoutDescriptor descriptor) const {
  LogIfError(DnnDestroyDropoutDescriptor(descriptor));
}
void internal::DnnRnnDescriptorDeleter::operator()(
    DnnRnnDescriptor descriptor) const {
  LogIfError(DnnDestroyRnnDescriptor(descriptor));
}

llvm::Expected<DnnLibraryVersion> DnnGetVersion(Platform platform) {
  DnnLibraryVersion version;
  switch (platform) {
    case Platform::CUDA:
      if (auto data = CudnnGetProperty(libraryPropertyType::MAJOR_VERSION)) {
        version.major = *data;
      } else {
        return data.takeError();
      }
      if (auto data = CudnnGetProperty(libraryPropertyType::MINOR_VERSION)) {
        version.minor = *data;
      } else {
        return data.takeError();
      }
      if (auto data = CudnnGetProperty(libraryPropertyType::PATCH_LEVEL)) {
        version.patch = *data;
        return version;
      } else {
        return data.takeError();
      }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningDnnHandle> DnnCreate(CurrentContext current) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnCreate(current);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnDestroy(DnnHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnDestroy(handle);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnSetStream(DnnHandle handle, Stream stream) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnSetStream(handle, stream);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<Stream> DnnGetStream(DnnHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnGetStream(handle);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningDnnTensorDescriptor> DnnCreateTensorDescriptor(
    Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CudnnCreateTensorDescriptor();
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnDestroyTensorDescriptor(DnnTensorDescriptor descriptor) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnDestroyTensorDescriptor(descriptor);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<DnnTensorDescriptorData> DnnGetTensorDescriptor(
    DnnTensorDescriptor descriptor) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      if (auto data = CudnnGetTensorDescriptor(descriptor)) {
        return ToDnn(*data);
      } else {
        return data.takeError();
      }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnSetTensorDescriptor(DnnTensorDescriptor descriptor,
                                   DnnDataType data_type,
                                   llvm::ArrayRef<int> dimensions,
                                   llvm::ArrayRef<int> strides) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnSetTensorDescriptor(descriptor, ToCuda(data_type), dimensions,
                                      strides);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningDnnConvolutionDescriptor> DnnCreateConvolutionDescriptor(
    Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CudnnCreateConvolutionDescriptor();
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnDestroyConvolutionDescriptor(
    DnnConvolutionDescriptor descriptor) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnDestroyConvolutionDescriptor(descriptor);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningDnnPoolingDescriptor> DnnCreatePoolingDescriptor(
    Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CudnnCreatePoolingDescriptor();
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnDestroyPoolingDescriptor(DnnPoolingDescriptor descriptor) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnDestroyPoolingDescriptor(descriptor);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnDestroyActivationDescriptor(DnnActivationDescriptor descriptor) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnDestroyActivationDescriptor(descriptor);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningDnnActivationDescriptor> DnnCreateActivationDescriptor(
    Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CudnnCreateActivationDescriptor();
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnSetTensor(CurrentContext current, DnnHandle handle,
                         DnnTensorDescriptor y_desc, Pointer<void> y,
                         Pointer<const void> value_ptr) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnSetTensor(current, handle, ToCuda(y_desc), y, value_ptr);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnScaleTensor(CurrentContext current, DnnHandle handle,
                           DnnTensorDescriptor y_desc, Pointer<void> y,
                           Pointer<const void> alpha) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnScaleTensor(current, handle, ToCuda(y_desc), y, alpha);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningDnnFilterDescriptor> DnnCreateFilterDescriptor(
    Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      if (auto data = CudnnCreateFilterDescriptor()) {
        return data;
      } else {
        return data.takeError();
      }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningDnnDropoutDescriptor> DnnCreateDropoutDescriptor(
    Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      if (auto data = CudnnCreateDropoutDescriptor()) {
        return data;
      } else {
        return data.takeError();
      }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningDnnRnnDescriptor> DnnCreateRnnDescriptor(
    Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      if (auto data = CudnnCreateRnnDescriptor()) {
        return data;
      } else {
        return data.takeError();
      }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnDestroyFilterDescriptor(DnnFilterDescriptor descriptor) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnDestroyFilterDescriptor(descriptor);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnDestroyDropoutDescriptor(DnnDropoutDescriptor descriptor) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnDestroyDropoutDescriptor(descriptor);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnDestroyRnnDescriptor(DnnRnnDescriptor descriptor) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnDestroyRnnDescriptor(descriptor);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnSetPoolingDescriptor(CurrentContext current,
                                    DnnPoolingDescriptor descriptor,
                                    DnnPoolingMode mode,
                                    DnnNanPropagation nan_propagation,
                                    llvm::ArrayRef<int> window_dimensions,
                                    llvm::ArrayRef<int> paddings,
                                    llvm::ArrayRef<int> strides) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnSetPoolingDescriptor(descriptor, ToCuda(mode),
                                       ToCuda(nan_propagation),
                                       window_dimensions, paddings, strides);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnSetConvolutionGroupCount(DnnConvolutionDescriptor descriptor,
                                        int group_count) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnSetConvolutionGroupCount(ToCuda(descriptor), group_count);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<int> DnnGetConvolutionGroupCount(
    DnnConvolutionDescriptor descriptor) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      if (auto data = CudnnGetConvolutionGroupCount(ToCuda(descriptor))) {
        return *data;
      } else {
        return data.takeError();
      }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<llvm::SmallVector<int, kDnnDimMax()>>
DnnGetConvolutionForwardOutputDim(DnnConvolutionDescriptor conv_desc,
                                  DnnTensorDescriptor input_tensor_desc,
                                  DnnFilterDescriptor filter_desc) {
  return CudnnGetConvolutionForwardOutputDim(
      ToCuda(conv_desc), ToCuda(input_tensor_desc), ToCuda(filter_desc));
}

llvm::Error DnnConvolutionBackwardBias(
    CurrentContext current, DnnHandle handle, Pointer<const void> alpha,
    DnnTensorDescriptor dy_desc, Pointer<const void> dy,
    Pointer<const void> beta, DnnTensorDescriptor db_desc, Pointer<void> db) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return DnnConvolutionBackwardBias(current, handle, alpha, ToCuda(dy_desc),
                                        dy, beta, ToCuda(db_desc), db);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<int> DnnGetConvolutionBackwardDataAlgorithmMaxCount(
    DnnHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      if (auto data =
              CudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle)) {
        return *data;
      } else {
        return data.takeError();
      }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<llvm::SmallVector<int, kDnnDimMax()>>
DnnGetPoolingForwardOutputDim(const DnnPoolingDescriptor pooling_desc,
                              const DnnTensorDescriptor input_tensor_desc) {
  auto platform = pooling_desc.platform();
  switch (platform) {
    case Platform::CUDA:
      if (auto data = CudnnGetPoolingForwardOutputDim(
              pooling_desc, ToCuda(input_tensor_desc))) {
        return data;
      } else {
        return data.takeError();
      }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnPoolingForward(CurrentContext current, DnnHandle handle,
                              const DnnPoolingDescriptor pooling_desc,
                              Pointer<const void> alpha,
                              const DnnTensorDescriptor x_desc,
                              Pointer<const void> x, Pointer<const void> beta,
                              const DnnTensorDescriptor y_desc,
                              Pointer<void> y) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnPoolingForward(current, handle, pooling_desc, alpha, x_desc,
                                 x, beta, y_desc, y);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnPoolingBackward(
    CurrentContext current, DnnHandle handle,
    const DnnPoolingDescriptor pooling_desc, Pointer<const void> alpha,
    const DnnTensorDescriptor y_desc, Pointer<const void> y,
    const DnnTensorDescriptor dy_desc, Pointer<const void> dy,
    const DnnTensorDescriptor x_desc, Pointer<const void> x,
    Pointer<const void> beta, const DnnTensorDescriptor dx_desc,
    Pointer<void> dx) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnPoolingBackward(current, handle, pooling_desc, alpha, y_desc,
                                  y, dy_desc, dy, x_desc, x, beta, dx_desc, dx);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnActivationForward(CurrentContext current, DnnHandle handle,
                                 DnnActivationDescriptor activation_desc,
                                 Pointer<const void> alpha,
                                 const DnnTensorDescriptor x_desc,
                                 Pointer<const void> x,
                                 Pointer<const void> beta,
                                 const DnnTensorDescriptor y_desc,
                                 Pointer<void> y) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnActivationForward(current, handle, ToCuda(activation_desc),
                                    alpha, ToCuda(x_desc), x, beta,
                                    ToCuda(y_desc), y);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnActivationBackward(
    CurrentContext current, DnnHandle handle,
    DnnActivationDescriptor activation_desc, Pointer<const void> alpha,
    const DnnTensorDescriptor y_desc, Pointer<const void> y,
    const DnnTensorDescriptor dy_desc, Pointer<const void> dy,
    const DnnTensorDescriptor x_desc, Pointer<const void> x,
    Pointer<const void> beta, const DnnTensorDescriptor dx_desc,
    Pointer<void> dx) {
  auto platform = activation_desc.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnActivationBackward(
          current, handle, ToCuda(activation_desc), alpha, ToCuda(y_desc), y,
          ToCuda(dy_desc), dy, ToCuda(x_desc), x, beta, ToCuda(dx_desc), dx);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnBatchNormalizationForwardInference(
    CurrentContext current, DnnHandle handle, DnnBatchNormMode mode,
    Pointer<const void> alpha, Pointer<const void> beta,
    DnnTensorDescriptor x_desc, Pointer<const void> x,
    DnnTensorDescriptor y_desc, Pointer<void> y,
    DnnTensorDescriptor bn_scale_bias_mean_var_desc,
    Pointer<const void> bn_scale, Pointer<const void> bn_bias,
    Pointer<const void> estimated_mean, Pointer<const void> estimated_variance,
    double epsilon) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnBatchNormalizationForwardInference(
          current, handle, ToCuda(mode), alpha, beta, ToCuda(x_desc), x,
          ToCuda(y_desc), y, ToCuda(bn_scale_bias_mean_var_desc), bn_scale,
          bn_bias, estimated_mean, estimated_variance, epsilon);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<size_t> DnnDropoutGetStatesSize(DnnHandle handle) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA:
      if (auto data = CudnnDropoutGetStatesSize(handle)) {
        return *data;
      } else {
        return data.takeError();
      }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<size_t> DnnDropoutGetReserveSpaceSize(
    CurrentContext current, DnnTensorDescriptor xdesc) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      if (auto data = CudnnDropoutGetReserveSpaceSize(ToCuda(xdesc))) {
        return *data;
      } else {
        return data.takeError();
      }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnRnnForwardInference(
    CurrentContext current, DnnHandle handle, DnnRnnDescriptor rnn_descriptor,
    llvm::ArrayRef<DnnTensorDescriptor> input_descriptors,
    Pointer<const void> input_data, DnnTensorDescriptor hidden_input_descriptor,
    Pointer<const void> hidden_input_data,
    DnnTensorDescriptor cell_input_descriptor,
    Pointer<const void> cell_input_data, DnnFilterDescriptor filter_descriptor,
    Pointer<const void> filter_data,
    llvm::ArrayRef<DnnTensorDescriptor> output_descriptors,
    Pointer<void> output_data, DnnTensorDescriptor hidden_output_descriptor,
    Pointer<void> hidden_output_data,
    DnnTensorDescriptor cell_output_descriptor, Pointer<void> cell_output_data,
    Pointer<void> workspace, size_t workspace_size_bytes) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA: {
      return CudnnRnnForwardInference(
          current, handle, rnn_descriptor, ToCuda(input_descriptors),
          input_data, hidden_input_descriptor, hidden_input_data,
          cell_input_descriptor, cell_input_data, filter_descriptor,
          filter_data, ToCuda(output_descriptors), output_data,
          hidden_output_descriptor, hidden_output_data, cell_output_descriptor,
          cell_output_data, workspace, workspace_size_bytes);
    }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnRnnForwardTraining(
    CurrentContext current, DnnHandle handle, DnnRnnDescriptor rnn_descriptor,
    llvm::ArrayRef<DnnTensorDescriptor> input_descriptors,
    Pointer<const void> input_data, DnnTensorDescriptor hidden_input_descriptor,
    Pointer<const void> hidden_input_data,
    DnnTensorDescriptor cell_input_descriptor,
    Pointer<const void> cell_input_data, DnnFilterDescriptor filter_descriptor,
    Pointer<const void> filter_data,
    llvm::ArrayRef<DnnTensorDescriptor> output_descriptors,
    Pointer<void> output_data, DnnTensorDescriptor hidden_output_descriptor,
    Pointer<void> hidden_output_data,
    DnnTensorDescriptor cell_output_descriptor, Pointer<void> cell_output_data,
    Pointer<void> workspace, size_t workspace_size_bytes,
    Pointer<void> reserve_space, size_t reserve_space_size_in_bytes) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA: {
      return CudnnRnnForwardTraining(
          current, handle, rnn_descriptor, ToCuda(input_descriptors),
          input_data, hidden_input_descriptor, hidden_input_data,
          cell_input_descriptor, cell_input_data, filter_descriptor,
          filter_data, ToCuda(output_descriptors), output_data,
          hidden_output_descriptor, hidden_output_data, cell_output_descriptor,
          cell_output_data, workspace, workspace_size_bytes, reserve_space,
          reserve_space_size_in_bytes);
    }
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt
