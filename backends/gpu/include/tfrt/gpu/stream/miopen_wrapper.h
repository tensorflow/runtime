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
#ifndef TFRT_GPU_STREAM_MIOPEN_WRAPPER_H_
#define TFRT_GPU_STREAM_MIOPEN_WRAPPER_H_

#include "miopen_stub.h"
#include "tfrt/gpu/stream/dnn_wrapper.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace gpu {
namespace stream {

struct MiopenErrorData {
  miopenStatus_t result;
  const char* expr;
  StackTrace stack_trace;
};
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const MiopenErrorData& data);
// Wraps a miopenStatus_t into an llvm::ErrorInfo.
using MiopenErrorInfo = TupleErrorInfo<MiopenErrorData>;
miopenStatus_t GetResult(const MiopenErrorInfo& info);

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, miopenDataType_t dtype);

template <>
struct PlatformTypeTraits<miopenDataType_t, DnnDataTypeTag>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<uint64_t, DnnConvFwdAlgoTag>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<uint64_t, DnnConvBwdDataAlgoTag>
    : public RocmPlatformType {};
template <>
struct PlatformTypeTraits<uint64_t, DnnConvBwdWeightsAlgoTag>
    : public RocmPlatformType {};

// Return types for functions returning multiple values.
struct MiopenTensorDescriptorData {
  miopenDataType_t data_type;
  llvm::SmallVector<int, kDnnDimMax()> dimensions;
  llvm::SmallVector<int, kDnnDimMax()> strides;
};
struct MiopenConvolutionDescriptorData {
  llvm::SmallVector<int, kDnnDimMax()> paddings;
  llvm::SmallVector<int, kDnnDimMax()> filter_strides;
  llvm::SmallVector<int, kDnnDimMax()> dilations;
  miopenConvolutionMode_t mode;
};

llvm::Expected<DnnLibraryVersion> MiopenGetVersion();
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
llvm::Expected<MiopenTensorDescriptorData> MiopenGetTensorDescriptor(
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
llvm::Expected<MiopenConvolutionDescriptorData> MiopenGetConvolutionDescriptor(
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

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_STREAM_MIOPEN_WRAPPER_H_
