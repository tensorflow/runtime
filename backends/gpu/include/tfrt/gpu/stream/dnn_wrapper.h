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

//===- dnn_wrapper.h --------------------------------------------*- C++ -*-===//
//
// Thin abstraction layer for cuDNN and MIOpen.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_GPU_STREAM_DNN_WRAPPER_H_
#define TFRT_GPU_STREAM_DNN_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "tfrt/gpu/stream/stream_wrapper.h"

namespace tfrt {
namespace gpu {
namespace stream {

constexpr int kDnnDimMax() { return 8; }

enum class DnnErrQueryMode {
  kRawcode,
  kNonblocking,
  kBlocking,
};

enum class DnnDataType {
  kFloat,
  kDouble,
  kHalf,
  kInt8,
  kInt32,
  kInt8x4,
  kUint8,
  kUint8x4,
  kInt8x32,
};

enum class DnnTensorFormat {
  kNchw,
  kNhwc,
  kNchwVectC,
};

enum class DnnRnnInputMode {
  kLinear,
  kSkip,
};

enum class DnnDirectionMode {
  kUnidirectional,
  kBidirectional,
};

enum class DnnRnnMode {
  kRnnRelu,
  kRnnTanh,
  kLstm,
  kGru,
};

enum class DnnOpTensorOp {
  kOpTensorAdd,
  kOpTensorMul,
  kOpTensorMin,
  kOpTensorMax,
  kOpTensorSqrt,
  kOpTensorNot,
};

enum class DnnReduceTensorOp {
  kReduceTensorAdd,
  kReduceTensorMul,
  kReduceTensorMin,
  kReduceTensorMax,
  kReduceTensorAmax,
  kReduceTensorAvg,
  kReduceTensorNorm1,
  kReduceTensorNorm2,
  kReduceTensorMulNoZeros,
};

enum class DnnReduceTensorIndices {
  kReduceTensorNoIndices,
  kReduceTensorFlattenedIndicesw,
};

enum class DnnIndicesType {
  kDnn32BitIndices,
  kDnn64BitIndices,
  kDnn16BitIndices,
  kDnn8BitIndices,
};

enum class DnnNanPropagation {
  kNotPropagateNan,
  kPropagateNan,
};

enum class DnnReorderType {
  kDefaultReorder,
  kNoReorder,
};

enum class DnnMathType {
  kDefaultMath,
  kTensorOpMath,
  kTensorOpMathAllowConversion,
};

enum class DnnConvolutionMode { kConvolution, kCrossCorrelation };

enum class DnnDeterminism {
  kNonDeterministic,
  kDeterministic,
};

enum class DnnSoftmaxAlgorithm {
  kSoftmaxFast,
  kSoftmaxAccurate,
  kSoftmaxLog,
};

enum class DnnSoftmaxMode {
  kSoftmaxModeInstance,
  kSoftmaxModeChannel,
};

enum class DnnPoolingMode {
  kPoolingMax,
  kPoolingAverageCountIncludePadding,
  kPoolingAverageCountExcludePadding,
  kPoolingMaxDeterministic,
};

enum class DnnActivationMode {
  kActivationSigmoid,
  kActivationRelu,
  kActivationTanh,
  kActivationClippedRelu,
  kActivationElu,
  kActivationIdentity,
};

enum class DnnBatchNormMode {
  // bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice,
  // normalized over N slice)
  kBatchnormPerActivation,
  // bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized
  // over Nx1xHxW subtensors)
  kBatchnormSpatial,
  // bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized
  // over Nx1xHxW subtensors). May be faster than BATCHNORM_SPATIAL but
  // imposes some limits on the range of values
  kBatchnormSpatialPersistent,
};

enum class DnnBatchNormOps {
  kBatchnormOpsBn,               // do batch normalization only
  kBatchnormOpsBnActivation,     // do batchNorm, then activation
  kBatchnormOpsBnAddActivation,  // do batchNorm, then elemWiseAdd, then
                                 // activation
};

enum class DnnRNNInputMode { kLinearInput, kSkipInput };

enum class DnnRNNMode {
  kRnnRelu,  // Stock RNN with ReLu activation
  kRnnTanh,  // Stock RNN with tanh activation
  kLstm,     // LSTM with no peephole connections
  kGru,      // Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z *
             // h(t-1);
};

enum class DnnRNNAlgo {
  kRnnAlgoStandard,
  kRnnAlgoPersistStatic,
  kRnnAlgoPersistDynamic,
  kRnnAlgoCount,
};

enum class DnnRNNBiasMode {
  kRnnNoBias,         // rnn cell formulas do not use biases
  kRnnSingleInpBias,  // rnn cell formulas use one input bias in input GEMM
  kRnnDoubleBias,     // default, rnn cell formulas use two bias vectors
  kRnnSingleRecBias,  // rnn cell formulas use one recurrent bias in recurrent
                      // GEMM
};

enum class DnnRNNClipMode {
  kRnnClipNone,    // disables LSTM cell clipping
  kRnnClipMinmax,  // enables LSTM cell clipping
};

struct DnnOpTensorDescriptorData {
  DnnOpTensorOp op;
  DnnDataType math_type;
  DnnNanPropagation nan_propagation;
};

struct DnnConvolutionDescriptorData {
  llvm::SmallVector<int, kDnnDimMax()> paddings;
  llvm::SmallVector<int, kDnnDimMax()> filter_strides;
  llvm::SmallVector<int, kDnnDimMax()> dilations;
  DnnConvolutionMode mode;
  DnnDataType math_type;
};

struct DnnPoolingDescriptorData {
  llvm::SmallVector<int, kDnnDimMax()> window_dimensions;
  llvm::SmallVector<int, kDnnDimMax()> paddings;
  llvm::SmallVector<int, kDnnDimMax()> strides;
  DnnPoolingMode mode;
  DnnNanPropagation nan_propagation;
};

struct DnnLibraryVersion {
  int major;
  int minor;
  int patch;
};

struct DnnActivationDescriptorData {
  DnnActivationMode mode;
  DnnNanPropagation nan_propagation;
  double coefficient;
};

struct DnnRnnClipData {
  DnnRNNClipMode mode;
  DnnNanPropagation nan_propagation;
  double left_clip;
  double right_clip;
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

// Return types for functions returning multiple values.
struct DnnTensorDescriptorData {
  DnnDataType data_type;
  llvm::SmallVector<int, kDnnDimMax()> dimensions;
  llvm::SmallVector<int, kDnnDimMax()> strides;
};
struct DnnFilterDescriptorData {
  DnnDataType data_type;
  DnnTensorFormat format;
  llvm::SmallVector<int, kDnnDimMax()> dimensions;
};
struct DnnDropoutDescriptorData {
  float dropout;
  Pointer<void> states;
  size_t state_size;
  uint64_t seed;
};
struct DnnRnnDescriptorData {
  int hidden_size;
  int num_layers;
  DnnDropoutDescriptor dropout_desc;
  DnnRnnInputMode input_mode;
  DnnDirectionMode direction;
  DnnRnnMode mode;
  int algorithm;
  DnnDataType math_precision;
};

llvm::Expected<DnnLibraryVersion> DnnGetVersion(Platform platform);

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

llvm::Expected<OwningDnnPoolingDescriptor> DnnCreatePoolingDescriptor(
    Platform platform);
llvm::Error DnnDestroyPoolingDescriptor(DnnPoolingDescriptor descriptor);

llvm::Expected<OwningDnnActivationDescriptor> DnnCreateActivationDescriptor(
    Platform platform);
llvm::Error DnnDestroyActivationDescriptor(DnnActivationDescriptor descriptor);

llvm::Error DnnSetTensor(CurrentContext current, DnnHandle handle,
                         DnnTensorDescriptor y_desc, Pointer<void> y,
                         Pointer<const void> value_ptr);
llvm::Error DnnScaleTensor(CurrentContext current, DnnHandle handle,
                           DnnTensorDescriptor y_desc, Pointer<void> y,
                           Pointer<const void> alpha);
llvm::Expected<OwningDnnFilterDescriptor> DnnCreateFilterDescriptor(
    Platform platform);
llvm::Error DnnDestroyFilterDescriptor(DnnFilterDescriptor descriptor);

llvm::Expected<OwningDnnDropoutDescriptor> DnnCreateDropoutDescriptor(
    Platform platform);
llvm::Error DnnDestroyDropoutDescriptor(DnnDropoutDescriptor descriptor);

llvm::Expected<OwningDnnRnnDescriptor> DnnCreateRnnDescriptor(
    Platform platform);
llvm::Error DnnDestroyRnnDescriptor(DnnRnnDescriptor descriptor);

llvm::Error DnnSetPoolingDescriptor(CurrentContext current,
                                    DnnPoolingDescriptor descriptor,
                                    DnnPoolingMode mode,
                                    DnnNanPropagation nan_propagation,
                                    llvm::ArrayRef<int> window_dimensions,
                                    llvm::ArrayRef<int> paddings,
                                    llvm::ArrayRef<int> strides);

llvm::Expected<DnnMathType> DnnGetConvolutionMathType(
    DnnConvolutionDescriptor descriptor);
llvm::Error DnnSetConvolutionGroupCount(DnnConvolutionDescriptor descriptor,
                                        int group_count);
llvm::Expected<int> DnnGetConvolutionGroupCount(
    DnnConvolutionDescriptor descriptor);
llvm::Expected<llvm::SmallVector<int, kDnnDimMax()>>
DnnGetConvolutionForwardOutputDim(DnnConvolutionDescriptor conv_desc,
                                  DnnTensorDescriptor input_tensor_desc,
                                  DnnFilterDescriptor filter_desc);
llvm::Error DnnConvolutionBackwardBias(
    CurrentContext current, DnnHandle handle, Pointer<const void> alpha,
    DnnTensorDescriptor dy_desc, Pointer<const void> dy,
    Pointer<const void> beta, DnnTensorDescriptor db_desc, Pointer<void> db);
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

llvm::Error DnnPoolingBackward(
    CurrentContext current, DnnHandle handle,
    const DnnPoolingDescriptor pooling_desc, Pointer<const void> alpha,
    const DnnTensorDescriptor y_desc, Pointer<const void> y,
    const DnnTensorDescriptor dy_desc, Pointer<const void> dy,
    const DnnTensorDescriptor x_desc, Pointer<const void> x,
    Pointer<const void> beta, const DnnTensorDescriptor dx_desc,
    Pointer<void> dx);

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

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_STREAM_DNN_WRAPPER_H_
