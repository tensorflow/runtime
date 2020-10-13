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

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/stream/cudnn_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace stream {

// Convert DNN wrapper enums to cuDNN enums.
static constexpr auto ToCuda(DnnDataType data_type) {
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

static constexpr cudnnPoolingMode_t ToCuda(DnnPoolingMode mode) {
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

static constexpr auto ToCuda(DnnNanPropagation nan) {
  switch (nan) {
    case DnnNanPropagation::kNotPropagateNan:
      return CUDNN_NOT_PROPAGATE_NAN;
    case DnnNanPropagation::kPropagateNan:
      return CUDNN_PROPAGATE_NAN;
  }
  llvm_unreachable(StrCat("Unrecognized DnnNanPropagation nan: ", nan).c_str());
}

/*
static constexpr auto ToCuda(DnnTensorFormat format) {
  return static_cast<cudnnTensorFormat_t>(format);
}
static constexpr auto ToCuda(DnnRnnInputMode mode) {
  return static_cast<cudnnRNNInputMode_t>(mode);
}
static constexpr auto ToCuda(DnnDirectionMode mode) {
  return static_cast<cudnnDirectionMode_t>(mode);
}
static constexpr auto ToCuda(DnnRnnMode mode) {
  return static_cast<cudnnRNNMode_t>(mode);
}
*/

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
void internal::DnnPersistentRnnPlanDeleter::operator()(
    DnnPersistentRnnPlan plan) const {
  LogIfError(DnnDestroyPersistentRnnPlan(plan));
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

llvm::Expected<OwningPersistentRnnPlan> DnnCreatePersistentRnnPlan(
    DnnRnnDescriptor descriptor, int batch_size, DnnDataType data_type) {
  auto platform = descriptor.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnCreatePersistentRnnPlan(descriptor, batch_size,
                                          ToCuda(data_type));
    case Platform::ROCm:
      return UnsupportedPlatform(platform);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error DnnDestroyPersistentRnnPlan(DnnPersistentRnnPlan plan) {
  auto platform = plan.platform();
  switch (platform) {
    case Platform::CUDA:
      return CudnnDestroyPersistentRnnPlan(plan);
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
