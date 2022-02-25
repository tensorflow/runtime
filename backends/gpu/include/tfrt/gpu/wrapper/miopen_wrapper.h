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

// Thin wrapper around the MIOpen API adding llvm::Error.
#ifndef TFRT_GPU_WRAPPER_MIOPEN_WRAPPER_H_
#define TFRT_GPU_WRAPPER_MIOPEN_WRAPPER_H_

#include "miopen_stub.h"
#include "tfrt/gpu/wrapper/dnn_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

// Placeholder value for math type attributes, which is only supported by cuDNN.
extern const DnnMathType kRocmDefaultMath;

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, miopenStatus_t status);

template <>
Expected<miopenDataType_t> Parse<miopenDataType_t>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, miopenDataType_t value);
template <>
Expected<miopenConvolutionMode_t> Parse<miopenConvolutionMode_t>(
    llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              miopenConvolutionMode_t value);
template <>
Expected<miopenActivationMode_t> Parse<miopenActivationMode_t>(
    llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              miopenActivationMode_t value);

template <>
struct PlatformTypeTraits<DnnDataTypeTag, miopenDataType_t>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<DnnConvolutionModeTag, miopenConvolutionMode_t>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<DnnActivationModeTag, miopenActivationMode_t>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<DnnConvFwdAlgoTag, uint64_t>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<DnnConvBwdDataAlgoTag, uint64_t>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<DnnConvBwdFilterAlgoTag, uint64_t>
    : public RocmPlatformType {};

mlir::TypeID GetMiopenDataTypeId(miopenDataType_t data_type);
std::pair<int, int> GetMiopenVectorizedSizeAndDim(miopenDataType_t data_type);
miopenDataType_t GetUnvectorizedMiopenDataType(miopenDataType_t data_type);
miopenDataType_t GetMiopenConvAccumulatorType(miopenDataType_t data_type,
                                              bool fp32_computation_for_fp16);
miopenDataType_t GetMiopenConvActivationType(miopenDataType_t data_type,
                                             bool fp32_computation_for_fp16);

llvm::Expected<LibraryVersion> MiopenGetVersion();
llvm::Expected<OwningDnnHandle> MiopenCreate(CurrentContext current);
llvm::Error MiopenDestroy(miopenHandle_t handle);
llvm::Error MiopenSetStream(miopenHandle_t handle, hipStream_t stream);
llvm::Expected<hipStream_t> MiopenGetStream(miopenHandle_t handle);

// Tensor descriptor.
llvm::Expected<OwningDnnTensorDescriptor> MiopenCreateTensorDescriptor();
llvm::Error MiopenDestroyTensorDescriptor(miopenTensorDescriptor_t descriptor);
llvm::Error MiopenSetTensorDescriptor(miopenTensorDescriptor_t descriptor,
                                      miopenDataType_t data_type,
                                      llvm::ArrayRef<int> dimensions,
                                      llvm::ArrayRef<int> strides);
llvm::Expected<DnnTensorDescriptorData> MiopenGetTensorDescriptor(
    miopenTensorDescriptor_t descriptor);
llvm::Expected<size_t> MiopenGetTensorNumBytes(
    miopenTensorDescriptor_t descriptor);

// Consolution descriptor.
llvm::Expected<OwningDnnConvolutionDescriptor>
MiopenCreateConvolutionDescriptor();
llvm::Error MiopenDestroyConvolutionDescriptor(
    miopenConvolutionDescriptor_t descriptor);
llvm::Error MiopenSetConvolutionGroupCount(
    miopenConvolutionDescriptor_t descriptor, int group_count);
llvm::Error MiopenInitConvolutionDescriptor(
    miopenConvolutionDescriptor_t descriptor, llvm::ArrayRef<int> pad,
    llvm::ArrayRef<int> filter_stride, llvm::ArrayRef<int> dilation,
    miopenConvolutionMode_t mode);
llvm::Expected<DnnConvolutionDescriptorData> MiopenGetConvolutionDescriptor(
    miopenConvolutionDescriptor_t descriptor);

// Convolution forward.
llvm::Expected<size_t> MiopenConvolutionForwardGetSolutionCount(
    miopenHandle_t handle, miopenTensorDescriptor_t w_desc,
    miopenTensorDescriptor_t x_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t y_desc);
llvm::Expected<llvm::SmallVector<miopenConvSolution_t, 1>>
MiopenConvolutionForwardGetSolution(miopenHandle_t handle,
                                    miopenTensorDescriptor_t w_desc,
                                    miopenTensorDescriptor_t x_desc,
                                    miopenConvolutionDescriptor_t conv_desc,
                                    miopenTensorDescriptor_t y_desc,
                                    size_t solution_count);
llvm::Expected<size_t> MiopenConvolutionForwardGetSolutionWorkspaceSize(
    miopenHandle_t handle, miopenTensorDescriptor_t w_desc,
    miopenTensorDescriptor_t x_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t y_desc, uint64_t solution);
llvm::Error MiopenConvolutionForwardImmediate(
    CurrentContext current, miopenHandle_t handle,
    miopenTensorDescriptor_t w_desc, Pointer<const void> w,
    miopenTensorDescriptor_t x_desc, Pointer<const void> x,
    miopenConvolutionDescriptor_t conv_desc, miopenTensorDescriptor_t y_desc,
    Pointer<void> y, Pointer<void> work_space, size_t work_space_size_in_bytes,
    uint64_t solution);

// Convolution backward data.
llvm::Expected<size_t> MiopenConvolutionBackwardDataGetSolutionCount(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t w_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dx_desc);
llvm::Expected<llvm::SmallVector<miopenConvSolution_t, 1>>
MiopenConvolutionBackwardDataGetSolution(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t w_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dx_desc, size_t solution_count);
llvm::Expected<size_t> MiopenConvolutionBackwardDataGetSolutionWorkspaceSize(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t w_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dx_desc, uint64_t solution);
llvm::Error MiopenConvolutionBackwardDataImmediate(
    CurrentContext current, miopenHandle_t handle,
    miopenTensorDescriptor_t dy_desc, Pointer<const void> dy,
    miopenTensorDescriptor_t w_desc, Pointer<const void> w,
    miopenConvolutionDescriptor_t conv_desc, miopenTensorDescriptor_t dx_desc,
    Pointer<void> dx, Pointer<void> work_space, size_t work_space_size_in_bytes,
    uint64_t solution);

// Convolution backward weights.
llvm::Expected<size_t> MiopenConvolutionBackwardWeightsGetSolutionCount(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t x_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dw_desc);
llvm::Expected<llvm::SmallVector<miopenConvSolution_t, 1>>
MiopenConvolutionBackwardWeightsGetSolution(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t x_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dw_desc, size_t solution_count);
llvm::Expected<size_t> MiopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t x_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dw_desc, uint64_t solution);
llvm::Error MiopenConvolutionBackwardWeightsImmediate(
    CurrentContext current, miopenHandle_t handle,
    miopenTensorDescriptor_t dy_desc, Pointer<const void> dy,
    miopenTensorDescriptor_t x_desc, Pointer<const void> x,
    miopenConvolutionDescriptor_t conv_desc, miopenTensorDescriptor_t dw_desc,
    Pointer<void> dw, Pointer<void> work_space, size_t work_space_size_in_bytes,
    uint64_t solution);

llvm::Expected<OwningDnnPoolingDescriptor> MiopenCreatePoolingDescriptor();
llvm::Error MiopenDestroyPoolingDescriptor(
    miopenPoolingDescriptor_t descriptor);
llvm::Error MiopenSetPoolingDescriptor(miopenPoolingDescriptor_t descriptor,
                                       miopenPoolingMode_t mode,
                                       llvm::ArrayRef<int> window_dimensions,
                                       llvm::ArrayRef<int> paddings,
                                       llvm::ArrayRef<int> strides);
llvm::Expected<DnnPoolingDescriptorData> MiopenGetPoolingDescriptor(
    const miopenPoolingDescriptor_t descriptor);
llvm::Expected<llvm::SmallVector<int, kDnnDimMax()>>
MiopenGetPoolingForwardOutputDim(
    const miopenPoolingDescriptor_t pooling_desc,
    const miopenTensorDescriptor_t input_tensor_desc);
llvm::Error MiopenPoolingForward(
    CurrentContext current, miopenHandle_t handle,
    const miopenPoolingDescriptor_t pooling_desc, Pointer<const void> alpha,
    const miopenTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, const miopenTensorDescriptor_t y_desc,
    Pointer<void> y, bool do_backward, Pointer<void> workspace,
    size_t workspace_size_bytes);
llvm::Expected<size_t> MiopenPoolingGetWorkSpaceSize(
    const miopenPoolingDescriptor_t pooling_desc,
    const miopenTensorDescriptor_t y_desc);
llvm::Error MiopenPoolingBackward(
    CurrentContext current, miopenHandle_t handle,
    const miopenPoolingDescriptor_t pooling_desc, Pointer<const void> alpha,
    const miopenTensorDescriptor_t y_desc, Pointer<const void> y,
    const miopenTensorDescriptor_t dy_desc, Pointer<const void> dy,
    const miopenTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, const miopenTensorDescriptor_t dx_desc,
    Pointer<void> dx, Pointer<void> workspace);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_MIOPEN_WRAPPER_H_
