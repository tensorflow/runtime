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

// Thin abstraction layer for cuDNN and MIOpen.
#include "tfrt/gpu/wrapper/dnn_wrapper.h"

#include "library_types.h"
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"
#include "tfrt/gpu/wrapper/miopen_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

mlir::TypeID GetDnnDataTypeId(DnnDataType data_type) {
  auto platform = data_type.platform();
  switch (platform) {
    case Platform::CUDA:
      return GetCudnnDataTypeId(data_type);
    case Platform::ROCm:
      return GetMiopenDataTypeId(data_type);
    default:
      return {};
  }
}

std::pair<int, int> GetTensorVectorizedSizeAndDim(DnnDataType data_type) {
  auto platform = data_type.platform();
  switch (platform) {
    case Platform::CUDA:
      return GetCudnnVectorizedSizeAndDim(data_type);
    case Platform::ROCm:
      return GetMiopenVectorizedSizeAndDim(data_type);
    default:
      return {};
  }
}

DnnDataType GetUnvectorizedDnnDataType(DnnDataType data_type) {
  auto platform = data_type.platform();
  switch (platform) {
    case Platform::CUDA:
      return GetUnvectorizedCudnnDataType(data_type);
    case Platform::ROCm:
      return GetUnvectorizedMiopenDataType(data_type);
    default:
      return {};
  }
}

DnnDataType GetConvAccumulatorType(DnnDataType data_type,
                                   bool fp32_computation_for_fp16) {
  auto platform = data_type.platform();
  switch (platform) {
    case Platform::CUDA:
      return GetCudnnConvAccumulatorType(data_type, fp32_computation_for_fp16);
    case Platform::ROCm:
      return GetMiopenConvAccumulatorType(data_type, fp32_computation_for_fp16);
    default:
      return {};
  }
}

DnnDataType GetConvActivationType(DnnDataType data_type,
                                  bool fp32_computation_for_fp16) {
  auto platform = data_type.platform();
  switch (platform) {
    case Platform::CUDA:
      return GetCudnnConvActivationType(data_type, fp32_computation_for_fp16);
    case Platform::ROCm:
      return GetMiopenConvActivationType(data_type, fp32_computation_for_fp16);
    default:
      return {};
  }
}

static cudnnPoolingMode_t ToCuda(DnnPoolingMode mode) {
  switch (mode) {
    case DnnPoolingMode::kPoolingMax:
      return CUDNN_POOLING_MAX;
    case DnnPoolingMode::kPoolingAverageCountIncludePadding:
      return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    case DnnPoolingMode::kPoolingAverageCountExcludePadding:
      return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  llvm_unreachable(
      StrCat("Unrecognized DnnPoolingMode mode: ", static_cast<int>(mode))
          .c_str());
}

static miopenPoolingMode_t ToRocm(DnnPoolingMode mode) {
  switch (mode) {
    case DnnPoolingMode::kPoolingMax:
      return miopenPoolingMax;
    case DnnPoolingMode::kPoolingAverageCountIncludePadding:
      return miopenPoolingAverage;
    case DnnPoolingMode::kPoolingAverageCountExcludePadding:
      return miopenPoolingAverageInclusive;
  }
  llvm_unreachable(
      StrCat("Unrecognized DnnPoolingMode mode: ", static_cast<int>(mode))
          .c_str());
}

static constexpr auto ToCuda(DnnBatchNormMode mode) {
  return static_cast<cudnnBatchNormMode_t>(mode);
}

//  Helper function to convert ArrayRef’s in Dnn wrapper to ArrayRef’s (vectors)
//  to be used with Cudnn
auto ToCuda(llvm::ArrayRef<DnnTensorDescriptor> dnn_descriptors) {
  llvm::SmallVector<cudnnTensorDescriptor_t, 16> cudnn_descriptors;
  cudnn_descriptors.reserve(dnn_descriptors.size());
  copy(dnn_descriptors, std::back_inserter(cudnn_descriptors));
  return cudnn_descriptors;
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

llvm::Expected<LibraryVersion> DnnGetVersion(Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return CudnnGetVersion();
    case Platform::ROCm:
      return MiopenGetVersion();
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
      return MiopenCreate(current);
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
      return MiopenDestroy(handle);
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
      return MiopenSetStream(handle, stream);
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
      return MiopenGetStream(handle);
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
      return MiopenCreateTensorDescriptor();
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
      return MiopenDestroyTensorDescriptor(descriptor);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<DnnTensorDescriptorData> DnnGetTensorDescriptor(
    DnnTensorDescriptor descriptor) {
  auto to_dnn = [](auto data) {
    DnnTensorDescriptorData dnn_data;
    dnn_data.data_type = data.data_type;
    dnn_data.dimensions = data.dimensions;
    dnn_data.strides = data.strides;
    return dnn_data;
  };
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA: {
      auto data = CudnnGetTensorDescriptor(descriptor);
      if (!data) return data.takeError();
      return to_dnn(*data);
    }
    case Platform::ROCm: {
      auto data = MiopenGetTensorDescriptor(descriptor);
      if (!data) return data.takeError();
      return to_dnn(*data);
    }
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
      return CudnnSetTensorDescriptor(descriptor, data_type, dimensions,
                                      strides);
    case Platform::ROCm:
      return MiopenSetTensorDescriptor(descriptor, data_type, dimensions,
                                       strides);
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
      return MiopenCreateConvolutionDescriptor();
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
      return MiopenDestroyConvolutionDescriptor(descriptor);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnSetConvolutionDescriptor(DnnConvolutionDescriptor descriptor,
                                        llvm::ArrayRef<int> pad,
                                        llvm::ArrayRef<int> filter_stride,
                                        llvm::ArrayRef<int> dilation,
                                        DnnConvolutionMode mode,
                                        DnnDataType compute_type) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnSetConvolutionDescriptor(descriptor, pad, filter_stride,
                                           dilation, mode, compute_type);
    case Platform::ROCm:
      return MiopenInitConvolutionDescriptor(descriptor, pad, filter_stride,
                                             dilation, mode);
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
      return MiopenCreatePoolingDescriptor();
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
      return MiopenDestroyPoolingDescriptor(descriptor);
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

llvm::Error DnnSetActivationDescriptor(DnnActivationDescriptor descriptor,
                                       DnnActivationMode mode,
                                       bool nan_progapation,
                                       double coefficient) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnSetActivationDescriptor(
          descriptor, mode,
          nan_progapation ? CUDNN_PROPAGATE_NAN : CUDNN_NOT_PROPAGATE_NAN,
          coefficient);
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
      return CudnnSetTensor(current, handle, y_desc, y, value_ptr);
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
      return CudnnScaleTensor(current, handle, y_desc, y, alpha);
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
      return CudnnCreateFilterDescriptor();
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnSetFilterDescriptor(DnnFilterDescriptor descriptor,
                                   DnnDataType data_type, int32_t format,
                                   llvm::ArrayRef<int> dimensions) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnSetFilterDescriptor(descriptor, data_type,
                                      static_cast<cudnnTensorFormat_t>(format),
                                      dimensions);
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
      return CudnnCreateDropoutDescriptor();
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
      return CudnnCreateRnnDescriptor();
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

llvm::Error DnnSetPoolingDescriptor(DnnPoolingDescriptor descriptor,
                                    DnnPoolingMode mode,
                                    DnnNanPropagation nan_propagation,
                                    llvm::ArrayRef<int> window_dimensions,
                                    llvm::ArrayRef<int> paddings,
                                    llvm::ArrayRef<int> strides) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnSetPoolingDescriptor(descriptor, ToCuda(mode),
                                       nan_propagation, window_dimensions,
                                       paddings, strides);
    case Platform::ROCm:
      assert(nan_propagation.platform() == Platform::NONE);
      return MiopenSetPoolingDescriptor(descriptor, ToRocm(mode),
                                        window_dimensions, paddings, strides);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnSetConvolutionGroupCount(DnnConvolutionDescriptor descriptor,
                                        int group_count) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnSetConvolutionGroupCount(descriptor, group_count);
    case Platform::ROCm:
      return MiopenSetConvolutionGroupCount(descriptor, group_count);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<int> DnnGetConvolutionGroupCount(
    DnnConvolutionDescriptor descriptor) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnGetConvolutionGroupCount(descriptor);
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
  auto platform = conv_desc.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnGetConvolutionForwardOutputDim(conv_desc, input_tensor_desc,
                                                 filter_desc);
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnConvolutionForward(
    CurrentContext current, DnnHandle handle, DnnDataType scale_type,
    DnnTensorDescriptor x_desc, Pointer<const void> x,
    DnnFilterDescriptor w_desc, Pointer<const void> w,
    DnnConvolutionDescriptor conv_desc, DnnConvFwdAlgo algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes,
    DnnTensorDescriptor y_desc, Pointer<void> y) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA: {
      if (GetDnnDataTypeId(scale_type) == mlir::TypeID::get<double>()) {
        double alpha = 1.0;
        double beta = 0.0;
        return CudnnConvolutionForward(
            current, handle, &alpha, x_desc, x, w_desc, w, conv_desc, algo,
            work_space, work_space_size_in_bytes, &beta, y_desc, y);
      } else {
        float alpha = 1.0;
        float beta = 0.0;
        return CudnnConvolutionForward(
            current, handle, &alpha, x_desc, x, w_desc, w, conv_desc, algo,
            work_space, work_space_size_in_bytes, &beta, y_desc, y);
      }
    }
    case Platform::ROCm:
      return MiopenConvolutionForwardImmediate(
          current, handle, w_desc, w, x_desc, x, conv_desc, y_desc, y,
          work_space, work_space_size_in_bytes, algo);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnConvolutionBackwardData(
    CurrentContext current, DnnHandle handle, DnnDataType scale_type,
    DnnFilterDescriptor w_desc, Pointer<const void> w,
    DnnTensorDescriptor dy_desc, Pointer<const void> dy,
    DnnConvolutionDescriptor conv_desc, DnnConvBwdDataAlgo algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes,
    DnnTensorDescriptor dx_desc, Pointer<void> dx) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA: {
      if (GetDnnDataTypeId(scale_type) == mlir::TypeID::get<double>()) {
        double alpha = 1.0;
        double beta = 0.0;
        return CudnnConvolutionBackwardData(
            current, handle, &alpha, w_desc, w, dy_desc, dy, conv_desc, algo,
            work_space, work_space_size_in_bytes, &beta, dx_desc, dx);
      } else {
        float alpha = 1.0;
        float beta = 0.0;
        return CudnnConvolutionBackwardData(
            current, handle, &alpha, w_desc, w, dy_desc, dy, conv_desc, algo,
            work_space, work_space_size_in_bytes, &beta, dx_desc, dx);
      }
    }
    case Platform::ROCm:
      return MiopenConvolutionBackwardDataImmediate(
          current, handle, dy_desc, dy, w_desc, w, conv_desc, dx_desc, dx,
          work_space, work_space_size_in_bytes, algo);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnConvolutionBackwardFilter(
    CurrentContext current, DnnHandle handle, DnnDataType scale_type,
    DnnTensorDescriptor x_desc, Pointer<const void> x,
    DnnTensorDescriptor dy_desc, Pointer<const void> dy,
    DnnConvolutionDescriptor conv_desc, DnnConvBwdFilterAlgo algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes,
    DnnFilterDescriptor dw_desc, Pointer<void> dw) {
  auto platform = handle.platform();
  switch (platform) {
    case Platform::CUDA: {
      if (GetDnnDataTypeId(scale_type) == mlir::TypeID::get<double>()) {
        double alpha = 1.0;
        double beta = 0.0;
        return CudnnConvolutionBackwardFilter(
            current, handle, &alpha, x_desc, x, dy_desc, dy, conv_desc, algo,
            work_space, work_space_size_in_bytes, &beta, dw_desc, dw);
      } else {
        float alpha = 1.0;
        float beta = 0.0;
        return CudnnConvolutionBackwardFilter(
            current, handle, &alpha, x_desc, x, dy_desc, dy, conv_desc, algo,
            work_space, work_space_size_in_bytes, &beta, dw_desc, dw);
      }
    }
    case Platform::ROCm:
      return MiopenConvolutionBackwardWeightsImmediate(
          current, handle, dy_desc, dy, x_desc, x, conv_desc, dw_desc, dw,
          work_space, work_space_size_in_bytes, algo);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnConvolutionBackwardBias(
    CurrentContext current, DnnHandle handle, Pointer<const void> alpha,
    DnnTensorDescriptor dy_desc, Pointer<const void> dy,
    Pointer<const void> beta, DnnTensorDescriptor db_desc, Pointer<void> db) {
  auto platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return DnnConvolutionBackwardBias(current, handle, alpha, dy_desc, dy,
                                        beta, db_desc, db);
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
      return CudnnGetPoolingForwardOutputDim(pooling_desc, input_tensor_desc);
    case Platform::ROCm:
      return MiopenGetPoolingForwardOutputDim(pooling_desc, input_tensor_desc);
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
      return MiopenPoolingForward(current, handle, pooling_desc, alpha, x_desc,
                                  x, beta, y_desc, y, /*do_backward=*/false,
                                  /*workspace=*/{}, /*workspace_size_bytes=*/0);
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
      return CudnnActivationForward(current, handle, activation_desc, alpha,
                                    x_desc, x, beta, y_desc, y);
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
      return CudnnActivationBackward(current, handle, activation_desc, alpha,
                                     y_desc, y, dy_desc, dy, x_desc, x, beta,
                                     dx_desc, dx);
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
          current, handle, ToCuda(mode), alpha, beta, x_desc, x, y_desc, y,
          bn_scale_bias_mean_var_desc, bn_scale, bn_bias, estimated_mean,
          estimated_variance, epsilon);
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
      if (auto data = CudnnDropoutGetReserveSpaceSize(xdesc)) {
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

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
