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

// Thin abstraction layer for cuDNN and MIOpen.
#ifndef TFRT_GPU_WRAPPER_DNN_WRAPPER_H_
#define TFRT_GPU_WRAPPER_DNN_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "mlir/Support/TypeID.h"
#include "tfrt/gpu/wrapper/wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

constexpr int kDnnDimMax() { return 8; }

// Platform-discriminated enums.
struct DnnDataTypeTag;
using DnnDataType = Enum<DnnDataTypeTag>;
struct DnnConvolutionModeTag;
using DnnConvolutionMode = Enum<DnnConvolutionModeTag>;
struct DnnActivationModeTag;
using DnnActivationMode = Enum<DnnActivationModeTag>;
struct DnnMathTypeTag;
using DnnMathType = Enum<DnnMathTypeTag>;
struct DnnConvFwdAlgoTag {
  using type = uint64_t;
};
using DnnConvFwdAlgo = Enum<DnnConvFwdAlgoTag>;
struct DnnConvBwdDataAlgoTag {
  using type = uint64_t;
};
using DnnConvBwdDataAlgo = Enum<DnnConvBwdDataAlgoTag>;
struct DnnConvBwdFilterAlgoTag {
  using type = uint64_t;
};
using DnnConvBwdFilterAlgo = Enum<DnnConvBwdFilterAlgoTag>;
struct DnnNanPropagationTag;
using DnnNanPropagation = Enum<DnnNanPropagationTag>;  // cuDNN only.

enum class DnnRnnInputMode {  // Matches miopenRNNInputMode_t
  kLinear,
  kSkip,
};

enum class DnnDirectionMode {  // Matches miopenRNNDirectionMode_t
  kUnidirectional,
  kBidirectional,
};

enum class DnnRnnMode {  // Matches miopenRNNMode_t
  kRnnRelu,
  kRnnTanh,
  kLstm,
  kGru,
};

enum class DnnOpTensorOp {  // Matches miopenTensorOp_t
  kOpTensorAdd,
  kOpTensorMul,
  kOpTensorMin,
  kOpTensorMax,
};

enum class DnnIndicesType {  // Values do not match miopenIndexType_t!
  kDnn32BitIndices,
  kDnn64BitIndices,
  kDnn16BitIndices,
  kDnn8BitIndices,
};

enum class DnnSoftmaxAlgorithm {  // Matches miopenSoftmaxAlgorithm_t
  kSoftmaxFast,
  kSoftmaxAccurate,
  kSoftmaxLog,
};

enum class DnnSoftmaxMode {  // Matches miopenSoftmaxMode_t
  kSoftmaxModeInstance,
  kSoftmaxModeChannel,
};

enum class DnnPoolingMode {  // Values do not match miopenPoolingMode_t!
  kPoolingMax,
  kPoolingAverageCountIncludePadding,
  kPoolingAverageCountExcludePadding,
};

enum class DnnBatchNormMode {  // Matches miopenBatchNormMode_t
  // bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice,
  // normalized over N slice)
  kBatchnormPerActivation,
  // bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized
  // over Nx1xHxW subtensors)
  kBatchnormSpatial,
};

// Non-owning handles of GPU resources.
using DnnHandle = Resource<cudnnHandle_t, miopenHandle_t>;
using DnnTensorDescriptor =
    Resource<cudnnTensorDescriptor_t, miopenTensorDescriptor_t>;
using DnnConvolutionDescriptor =
    Resource<cudnnConvolutionDescriptor_t, miopenConvolutionDescriptor_t>;
using DnnPoolingDescriptor =
    Resource<cudnnPoolingDescriptor_t, miopenPoolingDescriptor_t>;
using DnnActivationDescriptor =
    Resource<cudnnActivationDescriptor_t, miopenActivationDescriptor_t>;
using DnnFilterDescriptor =
    Resource<cudnnFilterDescriptor_t, miopenTensorDescriptor_t>;
using DnnDropoutDescriptor =
    Resource<cudnnDropoutDescriptor_t, miopenDropoutDescriptor_t>;
using DnnRnnDescriptor = Resource<cudnnRNNDescriptor_t, miopenRNNDescriptor_t>;

namespace internal {
// Helper to wrap resources and memory into RAII types.
struct DnnHandleDeleter {
  using pointer = DnnHandle;
  void operator()(DnnHandle handle) const;
};
struct DnnTensorDescriptorDeleter {
  using pointer = DnnTensorDescriptor;
  void operator()(DnnTensorDescriptor descriptor) const;
};
struct DnnConvolutionDescriptorDeleter {
  using pointer = DnnConvolutionDescriptor;
  void operator()(DnnConvolutionDescriptor descriptor) const;
};
struct DnnPoolingDescriptorDeleter {
  using pointer = DnnPoolingDescriptor;
  void operator()(DnnPoolingDescriptor descriptor) const;
};
struct DnnActivationDescriptorDeleter {
  using pointer = DnnActivationDescriptor;
  void operator()(DnnActivationDescriptor descriptor) const;
};
struct DnnFilterDescriptorDeleter {
  using pointer = DnnFilterDescriptor;
  void operator()(DnnFilterDescriptor descriptor) const;
};
struct DnnDropoutDescriptorDeleter {
  using pointer = DnnDropoutDescriptor;
  void operator()(DnnDropoutDescriptor descriptor) const;
};
struct DnnRnnDescriptorDeleter {
  using pointer = DnnRnnDescriptor;
  void operator()(DnnRnnDescriptor descriptor) const;
};
}  // namespace internal

// RAII wrappers for resources. Instances own the underlying resource.
//
// They are implemented as std::unique_ptrs with custom deleters.
//
// Use get() and release() to access the non-owning handle, please use with
// appropriate care.
using OwningDnnHandle = internal::OwningResource<internal::DnnHandleDeleter>;
using OwningDnnTensorDescriptor =
    internal::OwningResource<internal::DnnTensorDescriptorDeleter>;
using OwningDnnConvolutionDescriptor =
    internal::OwningResource<internal::DnnConvolutionDescriptorDeleter>;
using OwningDnnPoolingDescriptor =
    internal::OwningResource<internal::DnnPoolingDescriptorDeleter>;
using OwningDnnActivationDescriptor =
    internal::OwningResource<internal::DnnActivationDescriptorDeleter>;
using OwningDnnFilterDescriptor =
    internal::OwningResource<internal::DnnFilterDescriptorDeleter>;
using OwningDnnDropoutDescriptor =
    internal::OwningResource<internal::DnnDropoutDescriptorDeleter>;
using OwningDnnRnnDescriptor =
    internal::OwningResource<internal::DnnRnnDescriptorDeleter>;

struct DnnConvolutionDescriptorData {
  llvm::SmallVector<int, kDnnDimMax()> paddings;
  llvm::SmallVector<int, kDnnDimMax()> filter_strides;
  llvm::SmallVector<int, kDnnDimMax()> dilations;
  DnnConvolutionMode mode;
  DnnDataType math_type;  // Unspecified for MIOpen.
};

struct DnnPoolingDescriptorData {
  llvm::SmallVector<int, kDnnDimMax()> window_dimensions;
  llvm::SmallVector<int, kDnnDimMax()> paddings;
  llvm::SmallVector<int, kDnnDimMax()> strides;
  DnnPoolingMode mode;
  DnnNanPropagation nan_propagation;  // Unspecified for MIOpen.
};

struct DnnActivationDescriptorData {
  DnnActivationMode mode;
  DnnNanPropagation nan_propagation;  // Unspecified for MIOpen.
  double coefficient;
};

struct DnnTensorDescriptorData {
  DnnDataType data_type;
  llvm::SmallVector<int, kDnnDimMax()> dimensions;
  llvm::SmallVector<int, kDnnDimMax()> strides;
};

struct DnnRnnDescriptorData {
  int hidden_size;
  int num_layers;
  DnnDropoutDescriptor dropout_desc;
  DnnRnnInputMode input_mode;
  DnnDirectionMode direction;
  DnnRnnMode mode;
  int algorithm;
  DnnDataType math_type;  // Unspecified for MIOpen.
};

mlir::TypeID GetDnnDataTypeId(DnnDataType data_type);
// Returns vector size of 1 and dimension of -1 if data_type is not vectorized.
std::pair<int, int> GetTensorVectorizedSizeAndDim(DnnDataType data_type);
DnnDataType GetUnvectorizedDnnDataType(DnnDataType data_type);
DnnDataType GetConvAccumulatorType(DnnDataType data_type,
                                   bool fp32_computation_for_fp16);
DnnDataType GetConvActivationType(DnnDataType data_type,
                                  bool fp32_computation_for_fp16);

llvm::Expected<LibraryVersion> DnnGetVersion(Platform platform);

llvm::Expected<OwningDnnHandle> DnnCreate(CurrentContext current);
llvm::Error DnnDestroy(DnnHandle handle);
llvm::Error DnnSetStream(DnnHandle handle, Stream stream);
llvm::Expected<Stream> DnnGetStream(DnnHandle handle);

llvm::Expected<OwningDnnTensorDescriptor> DnnCreateTensorDescriptor(
    Platform platform);
llvm::Error DnnDestroyTensorDescriptor(DnnTensorDescriptor descriptor);

llvm::Error DnnSetTensorDescriptor(DnnTensorDescriptor descriptor,
                                   DnnDataType data_type,
                                   llvm::ArrayRef<int> dimensions,
                                   llvm::ArrayRef<int> strides);
llvm::Expected<DnnTensorDescriptorData> DnnGetTensorDescriptor(
    DnnTensorDescriptor descriptor);

llvm::Expected<OwningDnnConvolutionDescriptor> DnnCreateConvolutionDescriptor(
    Platform platform);
llvm::Error DnnDestroyConvolutionDescriptor(
    DnnConvolutionDescriptor descriptor);

llvm::Error DnnSetConvolutionDescriptor(DnnConvolutionDescriptor descriptor,
                                        llvm::ArrayRef<int> pad,
                                        llvm::ArrayRef<int> filter_stride,
                                        llvm::ArrayRef<int> dilation,
                                        DnnConvolutionMode mode,
                                        DnnDataType compute_type);

llvm::Expected<OwningDnnPoolingDescriptor> DnnCreatePoolingDescriptor(
    Platform platform);
llvm::Error DnnDestroyPoolingDescriptor(DnnPoolingDescriptor descriptor);

llvm::Expected<OwningDnnActivationDescriptor> DnnCreateActivationDescriptor(
    Platform platform);
llvm::Error DnnDestroyActivationDescriptor(DnnActivationDescriptor descriptor);

llvm::Error DnnSetActivationDescriptor(DnnActivationDescriptor descriptor,
                                       DnnActivationMode mode,
                                       bool nan_progapation,
                                       double coefficient);

llvm::Expected<OwningDnnFilterDescriptor> DnnCreateFilterDescriptor(
    Platform platform);
llvm::Error DnnDestroyFilterDescriptor(DnnFilterDescriptor descriptor);

llvm::Error DnnSetFilterDescriptor(DnnFilterDescriptor descriptor,
                                   DnnDataType data_type, int32_t format,
                                   llvm::ArrayRef<int> dimensions);

llvm::Expected<OwningDnnDropoutDescriptor> DnnCreateDropoutDescriptor(
    Platform platform);
llvm::Error DnnDestroyDropoutDescriptor(DnnDropoutDescriptor descriptor);

llvm::Expected<OwningDnnRnnDescriptor> DnnCreateRnnDescriptor(
    Platform platform);
llvm::Error DnnDestroyRnnDescriptor(DnnRnnDescriptor descriptor);

llvm::Error DnnSetTensor(CurrentContext current, DnnHandle handle,
                         DnnTensorDescriptor y_desc, Pointer<void> y,
                         Pointer<const void> value_ptr);
llvm::Error DnnScaleTensor(CurrentContext current, DnnHandle handle,
                           DnnTensorDescriptor y_desc, Pointer<void> y,
                           Pointer<const void> alpha);

llvm::Error DnnSetPoolingDescriptor(DnnPoolingDescriptor descriptor,
                                    DnnPoolingMode mode,
                                    DnnNanPropagation nan_propagation,
                                    llvm::ArrayRef<int> window_dimensions,
                                    llvm::ArrayRef<int> paddings,
                                    llvm::ArrayRef<int> strides);

llvm::Error DnnSetConvolutionGroupCount(DnnConvolutionDescriptor descriptor,
                                        int group_count);
llvm::Expected<int> DnnGetConvolutionGroupCount(
    DnnConvolutionDescriptor descriptor);
llvm::Expected<llvm::SmallVector<int, kDnnDimMax()>>
DnnGetConvolutionForwardOutputDim(DnnConvolutionDescriptor conv_desc,
                                  DnnTensorDescriptor input_tensor_desc,
                                  DnnFilterDescriptor filter_desc);
llvm::Error DnnConvolutionForward(
    CurrentContext current, DnnHandle handle, DnnDataType scale_type,
    DnnTensorDescriptor x_desc, Pointer<const void> x,
    DnnFilterDescriptor w_desc, Pointer<const void> w,
    DnnConvolutionDescriptor conv_desc, DnnConvFwdAlgo algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes,
    DnnTensorDescriptor y_desc, Pointer<void> y);
llvm::Error DnnConvolutionBackwardData(
    CurrentContext current, DnnHandle handle, DnnDataType scale_type,
    DnnFilterDescriptor w_desc, Pointer<const void> w,
    DnnTensorDescriptor dy_desc, Pointer<const void> dy,
    DnnConvolutionDescriptor conv_desc, DnnConvBwdDataAlgo algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes,
    DnnTensorDescriptor dx_desc, Pointer<void> dx);
llvm::Error DnnConvolutionBackwardFilter(
    CurrentContext current, DnnHandle handle, DnnDataType scale_type,
    DnnTensorDescriptor x_desc, Pointer<const void> x,
    DnnTensorDescriptor dy_desc, Pointer<const void> dy,
    DnnConvolutionDescriptor conv_desc, DnnConvBwdFilterAlgo algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes,
    DnnFilterDescriptor dw_desc, Pointer<void> dw);
llvm::Error DnnConvolutionBackwardBias(CurrentContext current, DnnHandle handle,
                                       Pointer<const void> alpha,
                                       DnnTensorDescriptor dy_desc,
                                       Pointer<const void> dy,
                                       DnnTensorDescriptor db_desc,
                                       Pointer<void> db);
llvm::Expected<llvm::SmallVector<int, kDnnDimMax()>>
DnnGetPoolingForwardOutputDim(const DnnPoolingDescriptor pooling_desc,
                              const DnnTensorDescriptor input_tensor_desc);
llvm::Error DnnPoolingForward(CurrentContext current, DnnHandle handle,
                              const DnnPoolingDescriptor pooling_desc,
                              Pointer<const void> alpha,
                              const DnnTensorDescriptor x_desc,
                              Pointer<const void> x, Pointer<const void> beta,
                              const DnnTensorDescriptor y_desc,
                              Pointer<void> y);

llvm::Error DnnActivationForward(CurrentContext current, DnnHandle handle,
                                 DnnActivationDescriptor activation_desc,
                                 Pointer<const void> alpha,
                                 const DnnTensorDescriptor x_desc,
                                 Pointer<const void> x,
                                 Pointer<const void> beta,
                                 const DnnTensorDescriptor y_desc,
                                 Pointer<void> y);
llvm::Error DnnActivationBackward(
    CurrentContext current, DnnHandle handle,
    DnnActivationDescriptor activation_desc, Pointer<const void> alpha,
    const DnnTensorDescriptor y_desc, Pointer<const void> y,
    const DnnTensorDescriptor dy_desc, Pointer<const void> dy,
    const DnnTensorDescriptor x_desc, Pointer<const void> x,
    Pointer<const void> beta, const DnnTensorDescriptor dx_desc,
    Pointer<void> dx);
llvm::Error DnnBatchNormalizationForwardInference(
    CurrentContext current, DnnHandle handle, DnnBatchNormMode mode,
    Pointer<const void> alpha, Pointer<const void> beta,
    DnnTensorDescriptor x_desc, Pointer<const void> x,
    DnnTensorDescriptor y_desc, Pointer<void> y,
    DnnTensorDescriptor bn_scale_bias_mean_var_desc,
    Pointer<const void> bn_scale, Pointer<const void> bn_bias,
    Pointer<const void> estimated_mean, Pointer<const void> estimated_variance,
    double epsilon);
llvm::Expected<size_t> DnnDropoutGetStatesSize(DnnHandle handle);
llvm::Expected<size_t> DnnDropoutGetReserveSpaceSize(CurrentContext current,
                                                     DnnTensorDescriptor xdesc);
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
    Pointer<void> workspace, size_t workspace_size_bytes);

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
    Pointer<void> reserve_space, size_t reserve_space_size_in_bytes);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_DNN_WRAPPER_H_
