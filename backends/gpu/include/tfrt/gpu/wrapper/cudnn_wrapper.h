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

// Thin wrapper around the cuDNN API adding llvm::Error.
#ifndef TFRT_GPU_WRAPPER_CUDNN_WRAPPER_H_
#define TFRT_GPU_WRAPPER_CUDNN_WRAPPER_H_

#include "cudnn.h"           // from @cudnn_headers
#include "cudnn_frontend.h"  // from @cudnn_frontend
#include "tfrt/gpu/wrapper/dnn_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

namespace internal {
// Specialize ErrorData to include cuDNN API log.
template <>
struct ErrorData<cudnnStatus_t> {
  cudnnStatus_t result;
  const char* expr;
  StackTrace stack_trace;
  std::string log;
};
llvm::raw_ostream& operator<<(llvm::raw_ostream&,
                              const ErrorData<cudnnStatus_t>&);
}  // namespace internal

Error MakeError(cudnnStatus_t result, const char* expr);

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cudnnStatus_t status);

template <>
Expected<cudnnDataType_t> Parse<cudnnDataType_t>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cudnnDataType_t value);
template <>
Expected<cudnnConvolutionMode_t> Parse<cudnnConvolutionMode_t>(
    llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              cudnnConvolutionMode_t value);
template <>
Expected<cudnnActivationMode_t> Parse<cudnnActivationMode_t>(
    llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              cudnnActivationMode_t value);
template <>
Expected<cudnnMathType_t> Parse<cudnnMathType_t>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cudnnMathType_t value);
template <>
Expected<cudnnConvolutionFwdAlgo_t> Parse<cudnnConvolutionFwdAlgo_t>(
    llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              cudnnConvolutionFwdAlgo_t value);
template <>
Expected<cudnnConvolutionBwdDataAlgo_t> Parse<cudnnConvolutionBwdDataAlgo_t>(
    llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              cudnnConvolutionBwdDataAlgo_t value);
template <>
Expected<cudnnConvolutionBwdFilterAlgo_t>
Parse<cudnnConvolutionBwdFilterAlgo_t>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              cudnnConvolutionBwdFilterAlgo_t value);

template <>
struct PlatformTypeTraits<DnnDataTypeTag, cudnnDataType_t>
    : public CudaPlatformType {};
template <>
struct PlatformTypeTraits<DnnConvolutionModeTag, cudnnConvolutionMode_t>
    : public CudaPlatformType {};
template <>
struct PlatformTypeTraits<DnnActivationModeTag, cudnnActivationMode_t>
    : public CudaPlatformType {};
template <>
struct PlatformTypeTraits<DnnMathTypeTag, cudnnMathType_t>
    : public CudaPlatformType {};
template <>
struct PlatformTypeTraits<DnnConvFwdAlgoTag, cudnnConvolutionFwdAlgo_t>
    : public CudaPlatformType {};
template <>
struct PlatformTypeTraits<DnnConvBwdDataAlgoTag, cudnnConvolutionBwdDataAlgo_t>
    : public CudaPlatformType {};
template <>
struct PlatformTypeTraits<DnnConvBwdFilterAlgoTag,
                          cudnnConvolutionBwdFilterAlgo_t>
    : public CudaPlatformType {};
template <>
struct PlatformTypeTraits<DnnNanPropagationTag, cudnnNanPropagation_t>
    : public CudaPlatformType {};

namespace internal {
struct CudnnPersistentRnnPlanDeleter {
  using pointer = cudnnPersistentRNNPlan_t;
  void operator()(cudnnPersistentRNNPlan_t plan) const;
};
}  // namespace internal

using OwningCudnnPersistentRnnPlan =
    internal::OwningResource<internal::CudnnPersistentRnnPlanDeleter>;

// Return types for functions returning multiple values.
struct CudnnFilterDescriptorData {
  cudnnDataType_t data_type;
  cudnnTensorFormat_t format;
  llvm::SmallVector<int, kDnnDimMax()> dimensions;
};
struct CudnnTransformDescriptorData {
  cudnnTensorFormat_t destination_format;
  llvm::SmallVector<int, kDnnDimMax()> paddings_before;
  llvm::SmallVector<int, kDnnDimMax()> paddings_after;
  llvm::SmallVector<unsigned, kDnnDimMax() - 2> fold;
  cudnnFoldingDirection_t direction;
};
struct CudnnOpTensorDescriptorData {
  cudnnOpTensorOp_t op;
  cudnnDataType_t math_type;
  cudnnNanPropagation_t nan_propagation;
};
struct CudnnReduceTensorDescriptorData {
  cudnnReduceTensorOp_t op;
  cudnnDataType_t math_type;
  cudnnNanPropagation_t nan_propagation;
  bool compute_indices;
  cudnnIndicesType_t index_type;
};
struct CudnnDropoutDescriptorData {
  float dropout;
  Pointer<void> states;
  size_t state_size;
  uint64_t seed;
};
struct CudnnRnnClipData {
  cudnnRNNClipMode_t mode;
  cudnnNanPropagation_t nan_propagation;
  double left_clip;
  double right_clip;
};

mlir::TypeID GetCudnnDataTypeId(cudnnDataType_t data_type);
std::pair<int, int> GetCudnnVectorizedSizeAndDim(cudnnDataType_t data_type);
cudnnDataType_t GetUnvectorizedCudnnDataType(cudnnDataType_t data_type);
cudnnDataType_t GetCudnnConvAccumulatorType(cudnnDataType_t data_type,
                                            bool fp32_computation_for_fp16);
cudnnDataType_t GetCudnnConvActivationType(cudnnDataType_t data_type,
                                           bool fp32_computation_for_fp16);

llvm::Expected<cudnnStatus_t> CudnnQueryRuntimeError(cudnnHandle_t handle,
                                                     cudnnErrQueryMode_t mode,
                                                     cudnnRuntimeTag_t* tag);
llvm::Expected<LibraryVersion> CudnnGetVersion();
llvm::Expected<OwningDnnHandle> CudnnCreate(CurrentContext current);
llvm::Error CudnnDestroy(cudnnHandle_t handle);
llvm::Error CudnnSetStream(cudnnHandle_t handle, cudaStream_t stream);
llvm::Expected<cudaStream_t> CudnnGetStream(cudnnHandle_t handle);

llvm::Expected<OwningDnnTensorDescriptor> CudnnCreateTensorDescriptor();
llvm::Error CudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t descriptor);
llvm::Error CudnnSetTensorDescriptor(cudnnTensorDescriptor_t descriptor,
                                     cudnnDataType_t data_type,
                                     llvm::ArrayRef<int> dimensions,
                                     llvm::ArrayRef<int> strides);
llvm::Expected<DnnTensorDescriptorData> CudnnGetTensorDescriptor(
    cudnnTensorDescriptor_t descriptor);
llvm::Expected<size_t> CudnnGetTensorSizeInBytes(
    cudnnTensorDescriptor_t descriptor);

llvm::Expected<cudnnTensorTransformDescriptor_t>
CudnnCreateTensorTransformDescriptor();
llvm::Error CudnnDestroyTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t descriptor);
llvm::Error CudnnSetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t descriptor,
    cudnnTensorFormat_t dest_format, llvm::ArrayRef<int> pad_before,
    llvm::ArrayRef<int> pad_after, llvm::ArrayRef<unsigned> fold,
    cudnnFoldingDirection_t direction);
llvm::Expected<CudnnTransformDescriptorData> CudnnGetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t descriptor, uint32_t rank);
llvm::Error CudnnTransformTensor(
    CurrentContext current, cudnnHandle_t handle, Pointer<const void> alpha,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, cudnnTensorDescriptor_t y_desc, Pointer<void> y);
llvm::Error CudnnTransformTensor(CurrentContext current, cudnnHandle_t handle,
                                 cudnnTensorTransformDescriptor_t trans_desc,
                                 Pointer<const void> alpha,
                                 cudnnTensorDescriptor_t src_desc,
                                 Pointer<const void> src_data,
                                 Pointer<const void> beta,
                                 cudnnTensorDescriptor_t dest_desc,
                                 Pointer<void> dest_data);

llvm::Error CudnnGetFoldedConvBackwardDataDescriptors(
    cudnnHandle_t handle, cudnnFilterDescriptor_t filter_desc,
    cudnnTensorDescriptor_t diff_desc, cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t grad_desc, cudnnTensorFormat_t transform_format,
    cudnnFilterDescriptor_t folded_filter_desc,
    cudnnTensorDescriptor_t padded_diff_desc,
    cudnnConvolutionDescriptor_t folded_conv_desc,
    cudnnTensorDescriptor_t folded_grad_desc,
    cudnnTensorTransformDescriptor_t filter_fold_trans_desc,
    cudnnTensorTransformDescriptor_t diff_pad_trans_desc,
    cudnnTensorTransformDescriptor_t grad_fold_trans_desc,
    cudnnTensorTransformDescriptor_t grad_unfold_trans_desc);

llvm::Error CudnnAddTensor(CurrentContext current, cudnnHandle_t handle,
                           Pointer<const void> alpha,
                           cudnnTensorDescriptor_t a_desc,
                           Pointer<const void> a, Pointer<const void> beta,
                           cudnnTensorDescriptor_t c_desc, Pointer<void> c);

llvm::Expected<cudnnOpTensorDescriptor_t> CudnnCreateOpTensorDescriptor();
llvm::Error CudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t op_tensor_desc,
                                       cudnnOpTensorOp_t op_tensor_op,
                                       cudnnDataType_t op_tensor_comp_type,
                                       cudnnNanPropagation_t op_tensor_nan_opt);
llvm::Expected<CudnnOpTensorDescriptorData> CudnnGetOpTensorDescriptor(
    cudnnOpTensorDescriptor_t descriptor);
llvm::Error CudnnDestroyOpTensorDescriptor(
    cudnnOpTensorDescriptor_t descriptor);
llvm::Error CudnnOpTensor(CurrentContext current, cudnnHandle_t handle,
                          cudnnOpTensorDescriptor_t op_tensor_desc,
                          Pointer<const void> alpha1,
                          cudnnTensorDescriptor_t a_desc, Pointer<const void> a,
                          Pointer<const void> alpha2,
                          cudnnTensorDescriptor_t b_desc, Pointer<const void> b,
                          Pointer<const void> beta,
                          cudnnTensorDescriptor_t c_desc, Pointer<void> c);

llvm::Expected<cudnnReduceTensorDescriptor_t>
CudnnCreateReduceTensorDescriptor();
llvm::Error CudnnSetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t descriptor,
    cudnnReduceTensorOp_t reduce_tensor_op,
    cudnnDataType_t reduce_tensor_comp_type,
    cudnnNanPropagation_t reduce_tensor_nan_opt,
    cudnnReduceTensorIndices_t reduce_tensor_indices,
    cudnnIndicesType_t reduce_tensor_indices_type);
llvm::Expected<CudnnReduceTensorDescriptorData> CudnnGetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t descriptor);
llvm::Error CudnnDestroyReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t descriptor);
llvm::Expected<size_t> CudnnGetReductionIndicesSize(
    cudnnHandle_t handle, cudnnReduceTensorDescriptor_t reduce_tensor_desc,
    cudnnTensorDescriptor_t a_desc, cudnnTensorDescriptor_t c_desc);
llvm::Expected<size_t> CudnnGetReductionWorkspaceSize(
    cudnnHandle_t handle, cudnnReduceTensorDescriptor_t reduce_tensor_desc,
    cudnnTensorDescriptor_t a_desc, cudnnTensorDescriptor_t c_desc);
llvm::Error CudnnReduceTensor(
    CurrentContext current, cudnnHandle_t handle,
    cudnnReduceTensorDescriptor_t reduce_tensor_desc, Pointer<void> indices,
    size_t indices_size_in_bytes, Pointer<void> workspace,
    size_t workspace_size_in_bytes, Pointer<const void> alpha,
    cudnnTensorDescriptor_t a_desc, Pointer<const void> a,
    Pointer<const void> beta, cudnnTensorDescriptor_t c_desc, Pointer<void> c);

llvm::Error CudnnSetTensor(CurrentContext current, cudnnHandle_t handle,
                           cudnnTensorDescriptor_t y_desc, Pointer<void> y,
                           Pointer<const void> value_ptr);
llvm::Error CudnnScaleTensor(CurrentContext current, cudnnHandle_t handle,
                             cudnnTensorDescriptor_t y_desc, Pointer<void> y,
                             Pointer<const void> alpha);

llvm::Expected<OwningDnnFilterDescriptor> CudnnCreateFilterDescriptor();
llvm::Error CudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t descriptor);
llvm::Error CudnnSetFilterDescriptor(cudnnFilterDescriptor_t descriptor,
                                     cudnnDataType_t data_type,
                                     cudnnTensorFormat_t format,
                                     llvm::ArrayRef<int> dimensions);
llvm::Expected<CudnnFilterDescriptorData> CudnnGetFilterDescriptor(
    cudnnFilterDescriptor_t descriptor);
llvm::Expected<size_t> CudnnGetFilterSizeInBytes(
    cudnnFilterDescriptor_t descriptor);
llvm::Error CudnnTransformFilter(CurrentContext current, cudnnHandle_t handle,
                                 cudnnTensorTransformDescriptor_t trans_desc,
                                 Pointer<const void> alpha,
                                 cudnnFilterDescriptor_t src_desc,
                                 Pointer<const void> src_data,
                                 Pointer<const void> beta,
                                 cudnnFilterDescriptor_t dest_desc,
                                 Pointer<void> dest_data);
llvm::Error CudnnReorderFilterAndBias(
    CurrentContext current, cudnnHandle_t handle,
    cudnnFilterDescriptor_t descriptor, cudnnReorderType_t reorder_type,
    Pointer<const void> filter_data, Pointer<void> reordered_filter_data,
    int reorder_bias, Pointer<const void> bias_data,
    Pointer<void> reordered_bias_data);

llvm::Expected<OwningDnnConvolutionDescriptor>
CudnnCreateConvolutionDescriptor();
llvm::Error CudnnDestroyConvolutionDescriptor(
    cudnnConvolutionDescriptor_t descriptor);
llvm::Error CudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t descriptor,
                                        cudnnMathType_t math_type);
llvm::Expected<cudnnMathType_t> CudnnGetConvolutionMathType(
    cudnnConvolutionDescriptor_t descriptor);
llvm::Error CudnnSetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t descriptor, int group_count);
llvm::Expected<int> CudnnGetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t descriptor);
llvm::Error CudnnSetConvolutionReorderType(
    cudnnConvolutionDescriptor_t descriptor, cudnnReorderType_t reorder_type);
llvm::Expected<cudnnReorderType_t> CudnnGetConvolutionReorderType(
    cudnnConvolutionDescriptor_t descriptor);
llvm::Error CudnnSetConvolutionDescriptor(
    cudnnConvolutionDescriptor_t descriptor, llvm::ArrayRef<int> pad,
    llvm::ArrayRef<int> filter_stride, llvm::ArrayRef<int> dilation,
    cudnnConvolutionMode_t mode, cudnnDataType_t compute_type);
llvm::Expected<DnnConvolutionDescriptorData> CudnnGetConvolutionDescriptor(
    cudnnConvolutionDescriptor_t descriptor);
llvm::Expected<llvm::SmallVector<int, kDnnDimMax()>>
CudnnGetConvolutionForwardOutputDim(cudnnConvolutionDescriptor_t conv_desc,
                                    cudnnTensorDescriptor_t input_tensor_desc,
                                    cudnnFilterDescriptor_t filter_desc);
llvm::Expected<int> CudnnGetConvolutionForwardAlgorithmMaxCount(
    cudnnHandle_t handle);
llvm::Expected<llvm::SmallVector<cudnnConvolutionFwdAlgoPerf_t, 1>>
CudnnFindConvolutionForwardAlgorithm(
    CurrentContext current, cudnnHandle_t handle,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnFilterDescriptor_t w_desc, Pointer<const void> w,
    cudnnConvolutionDescriptor_t conv_desc, cudnnTensorDescriptor_t y_desc,
    Pointer<void> y, int algo_count, Pointer<void> work_space,
    size_t work_space_size_in_bytes);
llvm::Expected<llvm::SmallVector<cudnnConvolutionFwdAlgoPerf_t, 1>>
CudnnGetConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                    cudnnTensorDescriptor_t src_desc,
                                    cudnnFilterDescriptor_t filter_desc,
                                    cudnnConvolutionDescriptor_t conv_desc,
                                    cudnnTensorDescriptor_t dest_desc,
                                    int algo_count);
llvm::Expected<size_t> CudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, cudnnTensorDescriptor_t x_desc,
    cudnnFilterDescriptor_t w_desc, cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t y_desc, cudnnConvolutionFwdAlgo_t algo);
llvm::Error CudnnConvolutionForward(
    CurrentContext current, cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnFilterDescriptor_t w_desc, Pointer<const void> w,
    cudnnConvolutionDescriptor_t conv_desc, cudnnConvolutionFwdAlgo_t algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes, const void* beta,
    cudnnTensorDescriptor_t y_desc, Pointer<void> y);
llvm::Error CudnnConvolutionBiasActivationForward(
    CurrentContext current, cudnnHandle_t handle, const void* alpha1,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnFilterDescriptor_t w_desc, Pointer<const void> w,
    cudnnConvolutionDescriptor_t conv_desc, cudnnConvolutionFwdAlgo_t algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes,
    const void* alpha2, cudnnTensorDescriptor_t z_desc, Pointer<const void> z,
    cudnnTensorDescriptor_t bias_desc, Pointer<const void> bias,
    cudnnActivationDescriptor_t activation_desc, cudnnTensorDescriptor_t y_desc,
    Pointer<void> y);
llvm::Error CudnnConvolutionBackwardBias(
    CurrentContext current, cudnnHandle_t handle, Pointer<const void> alpha,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    Pointer<const void> beta, cudnnTensorDescriptor_t db_desc,
    Pointer<void> db);
llvm::Expected<int> CudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
    cudnnHandle_t handle);
llvm::Expected<llvm::SmallVector<cudnnConvolutionBwdFilterAlgoPerf_t, 1>>
CudnnFindConvolutionBackwardFilterAlgorithm(
    CurrentContext current, cudnnHandle_t handle,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> y,
    cudnnConvolutionDescriptor_t conv_desc, cudnnFilterDescriptor_t dw_desc,
    Pointer<void> dw, int algo_count, Pointer<void> work_space,
    size_t work_space_size_in_bytes);
llvm::Expected<llvm::SmallVector<cudnnConvolutionBwdFilterAlgoPerf_t, 1>>
CudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, cudnnTensorDescriptor_t src_desc,
    cudnnTensorDescriptor_t diff_desc, cudnnConvolutionDescriptor_t conv_desc,
    cudnnFilterDescriptor_t grad_desc, int algo_count);
llvm::Expected<size_t> CudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, cudnnTensorDescriptor_t x_desc,
    cudnnTensorDescriptor_t dy_desc, cudnnConvolutionDescriptor_t conv_desc,
    cudnnFilterDescriptor_t grad_desc, cudnnConvolutionBwdFilterAlgo_t algo);
llvm::Error CudnnConvolutionBackwardFilter(
    CurrentContext current, cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnConvolutionBwdFilterAlgo_t algo, Pointer<void> work_space,
    size_t work_space_size_in_bytes, const void* beta,
    cudnnFilterDescriptor_t dw_desc, Pointer<void> dw);
llvm::Expected<int> CudnnGetConvolutionBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle);
llvm::Expected<llvm::SmallVector<cudnnConvolutionBwdDataAlgoPerf_t, 1>>
CudnnFindConvolutionBackwardDataAlgorithm(
    CurrentContext current, cudnnHandle_t handle,
    cudnnFilterDescriptor_t w_desc, Pointer<const void> w,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    cudnnConvolutionDescriptor_t conv_desc, cudnnTensorDescriptor_t dx_desc,
    Pointer<void> dx, int algo_count, Pointer<void> work_space,
    size_t work_space_size_in_bytes);
llvm::Expected<llvm::SmallVector<cudnnConvolutionBwdDataAlgoPerf_t, 1>>
CudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                         cudnnFilterDescriptor_t filter_desc,
                                         cudnnTensorDescriptor_t diff_desc,
                                         cudnnConvolutionDescriptor_t conv_desc,
                                         cudnnTensorDescriptor_t grad_desc,
                                         int algo_count);
llvm::Expected<size_t> CudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, cudnnFilterDescriptor_t w_desc,
    cudnnTensorDescriptor_t dy_desc, cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t dx_desc, cudnnConvolutionBwdDataAlgo_t algo);
llvm::Error CudnnConvolutionBackwardData(
    CurrentContext current, cudnnHandle_t handle, const void* alpha,
    cudnnFilterDescriptor_t w_desc, Pointer<const void> w,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    cudnnConvolutionDescriptor_t conv_desc, cudnnConvolutionBwdDataAlgo_t algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes, const void* beta,
    cudnnTensorDescriptor_t dx_desc, Pointer<void> dx);

llvm::Error CudnnIm2Col(CurrentContext current, cudnnHandle_t handle,
                        cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
                        cudnnFilterDescriptor_t w_desc,
                        cudnnConvolutionDescriptor_t conv_desc,
                        Pointer<void> col_buffer);
llvm::Error CudnnSoftmaxForward(
    CurrentContext current, cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, Pointer<const void> alpha,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, cudnnTensorDescriptor_t y_desc, Pointer<void> y);
llvm::Error CudnnSoftmaxBackward(
    CurrentContext current, cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, Pointer<const void> alpha,
    cudnnTensorDescriptor_t y_desc, Pointer<const void> y,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    Pointer<const void> beta, cudnnTensorDescriptor_t dx_desc,
    Pointer<void> dx);

llvm::Expected<OwningDnnPoolingDescriptor> CudnnCreatePoolingDescriptor();
llvm::Error CudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t descriptor);
llvm::Error CudnnSetPoolingDescriptor(cudnnPoolingDescriptor_t descriptor,
                                      cudnnPoolingMode_t mode,
                                      cudnnNanPropagation_t nan_propagation,
                                      llvm::ArrayRef<int> window_dimensions,
                                      llvm::ArrayRef<int> paddings,
                                      llvm::ArrayRef<int> strides);
llvm::Expected<DnnPoolingDescriptorData> CudnnGetPoolingDescriptor(
    const cudnnPoolingDescriptor_t descriptor);
llvm::Expected<llvm::SmallVector<int, kDnnDimMax()>>
CudnnGetPoolingForwardOutputDim(
    const cudnnPoolingDescriptor_t pooling_desc,
    const cudnnTensorDescriptor_t input_tensor_desc);
llvm::Error CudnnPoolingForward(CurrentContext current, cudnnHandle_t handle,
                                const cudnnPoolingDescriptor_t pooling_desc,
                                Pointer<const void> alpha,
                                const cudnnTensorDescriptor_t x_desc,
                                Pointer<const void> x, Pointer<const void> beta,
                                const cudnnTensorDescriptor_t y_desc,
                                Pointer<void> y);
llvm::Error CudnnPoolingBackward(
    CurrentContext current, cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t pooling_desc, Pointer<const void> alpha,
    const cudnnTensorDescriptor_t y_desc, Pointer<const void> y,
    const cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    const cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, const cudnnTensorDescriptor_t dx_desc,
    Pointer<void> dx);

llvm::Expected<OwningDnnActivationDescriptor> CudnnCreateActivationDescriptor();
llvm::Error CudnnDestroyActivationDescriptor(
    cudnnActivationDescriptor_t descriptor);
llvm::Error CudnnSetActivationDescriptor(cudnnActivationDescriptor_t descriptor,
                                         cudnnActivationMode_t mode,
                                         cudnnNanPropagation_t nan_propagation,
                                         double coefficient);
llvm::Expected<DnnActivationDescriptorData> CudnnGetActivationDescriptor(
    const cudnnActivationDescriptor_t activation_desc);
llvm::Error CudnnActivationForward(CurrentContext current, cudnnHandle_t handle,
                                   cudnnActivationDescriptor_t activation_desc,
                                   Pointer<const void> alpha,
                                   const cudnnTensorDescriptor_t x_desc,
                                   Pointer<const void> x,
                                   Pointer<const void> beta,
                                   const cudnnTensorDescriptor_t y_desc,
                                   Pointer<void> y);
llvm::Error CudnnActivationBackward(
    CurrentContext current, cudnnHandle_t handle,
    cudnnActivationDescriptor_t activation_desc, Pointer<const void> alpha,
    const cudnnTensorDescriptor_t y_desc, Pointer<const void> y,
    const cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    const cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, const cudnnTensorDescriptor_t dx_desc,
    Pointer<void> dx);

llvm::Expected<size_t> CudnnGetBatchNormalizationForwardTrainingWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bn_ops,
    cudnnTensorDescriptor_t x_desc, cudnnTensorDescriptor_t z_desc,
    cudnnTensorDescriptor_t y_desc,
    cudnnTensorDescriptor_t bn_scale_bias_mean_var_desc,
    cudnnActivationDescriptor_t activation_desc);
llvm::Expected<size_t> CudnnGetBatchNormalizationBackwardWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bn_ops,
    cudnnTensorDescriptor_t x_desc, cudnnTensorDescriptor_t y_desc,
    cudnnTensorDescriptor_t dy_desc, cudnnTensorDescriptor_t dz_desc,
    cudnnTensorDescriptor_t dx_desc,
    cudnnTensorDescriptor_t d_bn_scale_bias_desc,
    cudnnActivationDescriptor_t activation_desc);
llvm::Expected<size_t> CudnnGetBatchNormalizationTrainingReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bn_ops,
    cudnnActivationDescriptor_t activation_desc,
    cudnnTensorDescriptor_t x_desc);
llvm::Error CudnnBatchNormalizationForwardTrainingEx(
    CurrentContext current, cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bn_ops, Pointer<const void> alpha,
    Pointer<const void> beta, cudnnTensorDescriptor_t x_desc,
    Pointer<const void> x_data, cudnnTensorDescriptor_t z_desc,
    Pointer<const void> z_data, cudnnTensorDescriptor_t y_desc,
    Pointer<void> y_data, cudnnTensorDescriptor_t bn_scale_bias_mean_var_desc,
    Pointer<const void> bn_scale, Pointer<const void> bn_bias,
    double exponential_average_factor, Pointer<void> result_running_mean,
    Pointer<void> result_running_variance, double epsilon,
    Pointer<void> result_save_mean, Pointer<void> result_save_inv_variance,
    cudnnActivationDescriptor_t activation_desc, Pointer<void> workspace,
    size_t work_space_size_in_bytes, Pointer<void> reserve_space,
    size_t reserve_space_size_in_bytes);
llvm::Error CudnnBatchNormalizationForwardInference(
    CurrentContext current, cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    Pointer<const void> alpha, Pointer<const void> beta,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnTensorDescriptor_t y_desc, Pointer<void> y,
    cudnnTensorDescriptor_t bn_scale_bias_mean_var_desc,
    Pointer<const void> bn_scale, Pointer<const void> bn_bias,
    Pointer<const void> estimated_mean, Pointer<const void> estimated_variance,
    double epsilon);
llvm::Error CudnnBatchNormalizationBackwardEx(
    CurrentContext current, cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bn_ops, Pointer<const void> alpha_data_diff,
    Pointer<const void> beta_data_diff, Pointer<const void> alpha_param_diff,
    Pointer<const void> beta_param_diff, cudnnTensorDescriptor_t x_desc,
    Pointer<const void> x_data, cudnnTensorDescriptor_t y_desc,
    Pointer<const void> y_data, cudnnTensorDescriptor_t dy_desc,
    Pointer<const void> dy_data, cudnnTensorDescriptor_t dz_desc,
    Pointer<void> dz_data, cudnnTensorDescriptor_t dx_desc,
    Pointer<void> dx_data, cudnnTensorDescriptor_t d_bn_scale_bias_desc,
    Pointer<const void> bn_scale_data, Pointer<const void> bn_bias_data,
    Pointer<void> d_bn_scale_data, Pointer<void> d_bn_bias_data, double epsilon,
    Pointer<const void> saved_mean, Pointer<const void> saved_inv_variance,
    cudnnActivationDescriptor_t activation_desc, Pointer<void> work_space,
    size_t work_space_size_in_bytes, Pointer<void> reserve_space,
    size_t reserve_space_size_in_bytes);

llvm::Expected<OwningDnnDropoutDescriptor> CudnnCreateDropoutDescriptor();
llvm::Error CudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t descriptor);
llvm::Expected<size_t> CudnnDropoutGetStatesSize(cudnnHandle_t handle);
llvm::Expected<size_t> CudnnDropoutGetReserveSpaceSize(
    cudnnTensorDescriptor_t xdesc);
llvm::Error CudnnSetDropoutDescriptor(CurrentContext current,
                                      cudnnHandle_t handle,
                                      cudnnDropoutDescriptor_t descriptor,
                                      float dropout, Pointer<void> states,
                                      size_t states_size_bytes, uint64_t seed);
llvm::Expected<CudnnDropoutDescriptorData> CudnnGetDropoutDescriptor(
    cudnnHandle_t handle, cudnnDropoutDescriptor_t descriptor);
llvm::Error CudnnRestoreDropoutDescriptor(CurrentContext current,
                                          cudnnHandle_t handle,
                                          cudnnDropoutDescriptor_t descriptor,
                                          float dropout, Pointer<void> states,
                                          size_t state_size_in_bytes,
                                          uint64_t seed);
llvm::Error CudnnDropoutForward(CurrentContext current, cudnnHandle_t handle,
                                cudnnDropoutDescriptor_t descriptor,
                                cudnnTensorDescriptor_t xdesc,
                                Pointer<const void> x,
                                cudnnTensorDescriptor_t ydesc, Pointer<void> y,
                                Pointer<void> reserve_space,
                                size_t reserve_space_size_in_bytes);
llvm::Error CudnnDropoutBackward(CurrentContext current, cudnnHandle_t handle,
                                 cudnnDropoutDescriptor_t descriptor,
                                 cudnnTensorDescriptor_t dydesc,
                                 Pointer<const void> dy,
                                 cudnnTensorDescriptor_t dxdesc,
                                 Pointer<void> dx, Pointer<void> reserve_space,
                                 size_t reserve_space_size_in_bytes);

llvm::Expected<OwningDnnRnnDescriptor> CudnnCreateRnnDescriptor();
llvm::Error CudnnDestroyRnnDescriptor(cudnnRNNDescriptor_t descriptor);
llvm::Error CudnnSetRnnDescriptor(cudnnHandle_t handle,
                                  cudnnRNNDescriptor_t descriptor,
                                  int hidden_size, int num_layers,
                                  cudnnDropoutDescriptor_t dropout,
                                  cudnnRNNInputMode_t input_mode,
                                  cudnnDirectionMode_t direction,
                                  cudnnRNNMode_t mode, cudnnRNNAlgo_t algorithm,
                                  cudnnDataType_t math_precision);
llvm::Expected<DnnRnnDescriptorData> CudnnGetRnnDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor);
llvm::Error CudnnSetRnnMatrixMathType(cudnnRNNDescriptor_t descriptor,
                                      cudnnMathType_t m_type);
llvm::Expected<cudnnMathType_t> CudnnGetRnnMatrixMathType(
    cudnnRNNDescriptor_t descriptor);
llvm::Error CudnnSetRnnBiasMode(cudnnRNNDescriptor_t descriptor,
                                cudnnRNNBiasMode_t bias_mode);
llvm::Expected<cudnnRNNBiasMode_t> CudnnGetRnnBiasMode(
    cudnnRNNDescriptor_t descriptor);
llvm::Error CudnnRnnSetClip(cudnnHandle_t handle,
                            cudnnRNNDescriptor_t descriptor,
                            cudnnRNNClipMode_t clip_mode,
                            cudnnNanPropagation_t clip_nan_opt, double lclip,
                            double rclip);
llvm::Expected<CudnnRnnClipData> CudnnRnnGetClip(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor);

llvm::Expected<size_t> CudnnGetRnnParamsSize(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn_descriptor,
    cudnnTensorDescriptor_t tensor_descriptor, cudnnDataType_t data_type);
llvm::Expected<size_t> CudnnGetRnnWorkspaceSize(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn_descriptor,
    llvm::ArrayRef<cudnnTensorDescriptor_t> tensor_descriptors);
llvm::Expected<size_t> CudnnGetRnnTrainingReserveSize(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn_descriptor,
    llvm::ArrayRef<cudnnTensorDescriptor_t> tensor_descriptors);

llvm::Expected<Pointer<void>> CudnnGetRnnLinLayerMatrixParams(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn_descriptor, int pseudo_layer,
    cudnnTensorDescriptor_t tensor_descriptor,
    const cudnnFilterDescriptor_t filter_descriptor,
    Pointer<const void> weights, int layer_index,
    cudnnFilterDescriptor_t matrix_descriptor);
llvm::Expected<Pointer<void>> CudnnGetRnnLinLayerBiasParams(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn_descriptor, int pseudo_layer,
    cudnnTensorDescriptor_t tensor_descriptor,
    const cudnnFilterDescriptor_t filter_descriptor,
    Pointer<const void> weights, int layer_index,
    cudnnFilterDescriptor_t bias_descriptor);

llvm::Expected<OwningCudnnPersistentRnnPlan> CudnnCreatePersistentRnnPlan(
    cudnnRNNDescriptor_t descriptor, int batch_size, cudnnDataType_t data_type);
llvm::Error CudnnDestroyPersistentRnnPlan(cudnnPersistentRNNPlan_t plan);
llvm::Error CudnnSetPersistentRnnPlan(cudnnRNNDescriptor_t descriptor,
                                      cudnnPersistentRNNPlan_t plan);

llvm::Error CudnnRnnForwardInference(
    CurrentContext current, cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnn_descriptor,
    llvm::ArrayRef<cudnnTensorDescriptor_t> input_descriptors,
    Pointer<const void> input_data,
    cudnnTensorDescriptor_t hidden_input_descriptor,
    Pointer<const void> hidden_input_data,
    cudnnTensorDescriptor_t cell_input_descriptor,
    Pointer<const void> cell_input_data,
    cudnnFilterDescriptor_t filter_descriptor, Pointer<const void> filter_data,
    llvm::ArrayRef<cudnnTensorDescriptor_t> output_descriptors,
    Pointer<void> output_data, cudnnTensorDescriptor_t hidden_output_descriptor,
    Pointer<void> hidden_output_data,
    cudnnTensorDescriptor_t cell_output_descriptor,
    Pointer<void> cell_output_data, Pointer<void> workspace,
    size_t workspace_size_bytes);
llvm::Error CudnnRnnForwardTraining(
    CurrentContext current, cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnn_descriptor,
    llvm::ArrayRef<cudnnTensorDescriptor_t> input_descriptors,
    Pointer<const void> input_data,
    cudnnTensorDescriptor_t hidden_input_descriptor,
    Pointer<const void> hidden_input_data,
    cudnnTensorDescriptor_t cell_input_descriptor,
    Pointer<const void> cell_input_data,
    cudnnFilterDescriptor_t filter_descriptor, Pointer<const void> filter_data,
    llvm::ArrayRef<cudnnTensorDescriptor_t> output_descriptors,
    Pointer<void> output_data, cudnnTensorDescriptor_t hidden_output_descriptor,
    Pointer<void> hidden_output_data,
    cudnnTensorDescriptor_t cell_output_descriptor,
    Pointer<void> cell_output_data, Pointer<void> workspace,
    size_t workspace_size_bytes, Pointer<void> reserve_space,
    size_t reserve_space_size_in_bytes);
llvm::Error CudnnBackendExecute(CurrentContext current, cudnnHandle_t handle,
                                cudnnBackendDescriptor_t execution_plan,
                                cudnnBackendDescriptor_t variant_pack);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CUDNN_WRAPPER_H_
