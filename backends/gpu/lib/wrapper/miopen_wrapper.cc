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

// Thin wrapper around the MIOpen API adding llvm::Error.
#include "tfrt/gpu/wrapper/miopen_wrapper.h"

#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

Expected<miopenDataType_t> ParseMiopenDataType(llvm::StringRef name) {
  if (name == "miopenHalf") return miopenHalf;
  if (name == "miopenFloat") return miopenFloat;
  if (name == "miopenInt32") return miopenInt32;
  if (name == "miopenInt8") return miopenInt8;
  if (name == "miopenInt8x4") return miopenInt8x4;
  if (name == "miopenBFloat16") return miopenBFloat16;
  return MakeStringError("Unknown miopenDataType_t: ", name);
}

llvm::Expected<LibraryVersion> MiopenGetVersion() {
  size_t major, minor, patch;
  RETURN_IF_ERROR(miopenGetVersion(&major, &minor, &patch));
  return LibraryVersion{static_cast<int>(major), static_cast<int>(minor),
                        static_cast<int>(patch)};
}

llvm::Expected<OwningDnnHandle> MiopenCreate(CurrentContext current) {
  CheckHipContext(current);
  miopenHandle_t handle = nullptr;
  RETURN_IF_ERROR(miopenCreate(&handle));
  return OwningDnnHandle(handle);
}

llvm::Error MiopenDestroy(miopenHandle_t handle) {
  return TO_ERROR(miopenDestroy(handle));
}

llvm::Error MiopenSetStream(miopenHandle_t handle, hipStream_t stream) {
  return TO_ERROR(miopenSetStream(handle, stream));
}

llvm::Expected<hipStream_t> MiopenGetStream(miopenHandle_t handle) {
  hipStream_t stream = nullptr;
  RETURN_IF_ERROR(miopenGetStream(handle, &stream));
  return stream;
}

llvm::Expected<OwningDnnTensorDescriptor> MiopenCreateTensorDescriptor() {
  miopenTensorDescriptor_t descriptor = nullptr;
  RETURN_IF_ERROR(miopenCreateTensorDescriptor(&descriptor));
  return OwningDnnTensorDescriptor(descriptor);
}

llvm::Error MiopenDestroyTensorDescriptor(miopenTensorDescriptor_t descriptor) {
  return TO_ERROR(miopenDestroyTensorDescriptor(descriptor));
}

llvm::Error MiopenSetTensorDescriptor(miopenTensorDescriptor_t descriptor,
                                      miopenDataType_t data_type,
                                      llvm::ArrayRef<int> dimensions,
                                      llvm::ArrayRef<int> strides) {
  if (dimensions.size() != strides.size()) {
    return MakeStringError("Expected dimensions and strides to be equal size");
  }
  return TO_ERROR(miopenSetTensorDescriptor(
      descriptor, data_type, dimensions.size(),
      const_cast<int*>(dimensions.data()), const_cast<int*>(strides.data())));
}

llvm::Expected<DnnTensorDescriptorData> MiopenGetTensorDescriptor(
    miopenTensorDescriptor_t descriptor) {
  int rank = 0;
  RETURN_IF_ERROR(miopenGetTensorDescriptorSize(descriptor, &rank));
  miopenDataType_t data_type;
  DnnTensorDescriptorData data;
  data.dimensions.resize(rank);
  data.strides.resize(rank);
  RETURN_IF_ERROR(miopenGetTensorDescriptor(
      descriptor, &data_type, data.dimensions.data(), data.strides.data()));
  data.data_type = data_type;
  return data;
}

llvm::Expected<size_t> MiopenGetTensorNumBytes(
    miopenTensorDescriptor_t descriptor) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(miopenGetTensorNumBytes(descriptor, &size_in_bytes));
  return size_in_bytes;
}

llvm::Expected<OwningDnnConvolutionDescriptor>
MiopenCreateConvolutionDescriptor() {
  miopenConvolutionDescriptor_t conv_desc;
  RETURN_IF_ERROR(miopenCreateConvolutionDescriptor(&conv_desc));
  return OwningDnnConvolutionDescriptor(conv_desc);
}

llvm::Error MiopenDestroyConvolutionDescriptor(
    miopenConvolutionDescriptor_t descriptor) {
  return TO_ERROR(miopenDestroyConvolutionDescriptor(descriptor));
}

llvm::Error MiopenSetConvolutionGroupCount(
    miopenConvolutionDescriptor_t descriptor, int group_count) {
  return TO_ERROR(miopenSetConvolutionGroupCount(descriptor, group_count));
}

llvm::Error MiopenInitConvolutionDescriptor(
    miopenConvolutionDescriptor_t descriptor, llvm::ArrayRef<int> pad,
    llvm::ArrayRef<int> filter_stride, llvm::ArrayRef<int> dilation,
    miopenConvolutionMode_t mode) {
  if (pad.size() != filter_stride.size() || pad.size() != dilation.size()) {
    return MakeStringError(
        "Expected paddings, filter_strides and dilations arrays of equal size");
  }
  return TO_ERROR(miopenInitConvolutionNdDescriptor(
      descriptor, pad.size(), const_cast<int*>(pad.data()),
      const_cast<int*>(filter_stride.data()), const_cast<int*>(dilation.data()),
      mode));
}

llvm::Expected<DnnConvolutionDescriptorData> MiopenGetConvolutionDescriptor(
    miopenConvolutionDescriptor_t descriptor) {
  int rank = 0;
  miopenConvolutionMode_t mode;
  DnnConvolutionDescriptorData data;
  data.paddings.resize(kDnnDimMax());
  data.filter_strides.resize((kDnnDimMax()));
  data.dilations.resize(kDnnDimMax());
  RETURN_IF_ERROR(miopenGetConvolutionNdDescriptor(
      descriptor, kDnnDimMax(), &rank, data.paddings.data(),
      data.filter_strides.data(), data.dilations.data(), &mode));
  data.mode = static_cast<DnnConvolutionMode>(mode);
  data.paddings.resize(rank);
  data.filter_strides.resize(rank);
  data.dilations.resize(rank);
  return data;
}

llvm::Expected<size_t> MiopenConvolutionForwardGetSolutionCount(
    miopenHandle_t handle, miopenTensorDescriptor_t w_desc,
    miopenTensorDescriptor_t x_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t y_desc) {
  size_t count;
  RETURN_IF_ERROR(miopenConvolutionForwardGetSolutionCount(
      handle, w_desc, x_desc, conv_desc, y_desc, &count));
  return count;
}

llvm::Expected<llvm::SmallVector<miopenConvSolution_t, 1>>
MiopenConvolutionForwardGetSolution(miopenHandle_t handle,
                                    miopenTensorDescriptor_t w_desc,
                                    miopenTensorDescriptor_t x_desc,
                                    miopenConvolutionDescriptor_t conv_desc,
                                    miopenTensorDescriptor_t y_desc,
                                    size_t solution_count) {
  llvm::SmallVector<miopenConvSolution_t, 1> solutions(solution_count);
  RETURN_IF_ERROR(miopenConvolutionForwardGetSolution(
      handle, w_desc, x_desc, conv_desc, y_desc, solution_count,
      &solution_count, solutions.data()));
  solutions.resize(solution_count);
  return solutions;
}

llvm::Expected<size_t> MiopenConvolutionForwardGetSolutionWorkspaceSize(
    miopenHandle_t handle, miopenTensorDescriptor_t w_desc,
    miopenTensorDescriptor_t x_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t y_desc, uint64_t solution) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(miopenConvolutionForwardGetSolutionWorkspaceSize(
      handle, w_desc, x_desc, conv_desc, y_desc, solution, &size_in_bytes));
  return size_in_bytes;
}

llvm::Error MiopenConvolutionForwardImmediate(
    CurrentContext current, miopenHandle_t handle,
    miopenTensorDescriptor_t w_desc, Pointer<const void> w,
    miopenTensorDescriptor_t x_desc, Pointer<const void> x,
    miopenConvolutionDescriptor_t conv_desc, miopenTensorDescriptor_t y_desc,
    Pointer<void> y, Pointer<void> work_space, size_t work_space_size_in_bytes,
    uint64_t solution) {
  CheckHipContext(current);
  return TO_ERROR(miopenConvolutionForwardImmediate(
      handle, w_desc, ToRocm(w), x_desc, ToRocm(x), conv_desc, y_desc,
      ToRocm(y), ToRocm(work_space), work_space_size_in_bytes, solution));
}

llvm::Expected<size_t> MiopenConvolutionBackwardDataGetSolutionCount(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t w_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dx_desc) {
  size_t count;
  RETURN_IF_ERROR(miopenConvolutionBackwardDataGetSolutionCount(
      handle, dy_desc, w_desc, conv_desc, dx_desc, &count));
  return count;
}

llvm::Expected<llvm::SmallVector<miopenConvSolution_t, 1>>
MiopenConvolutionBackwardDataGetSolution(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t w_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dx_desc, size_t solution_count) {
  llvm::SmallVector<miopenConvSolution_t, 1> solutions(solution_count);
  RETURN_IF_ERROR(miopenConvolutionBackwardDataGetSolution(
      handle, dy_desc, w_desc, conv_desc, dx_desc, solution_count,
      &solution_count, solutions.data()));
  solutions.resize(solution_count);
  return solutions;
}

llvm::Expected<size_t> MiopenConvolutionBackwardDataGetSolutionWorkspaceSize(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t w_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dx_desc, uint64_t solution) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(miopenConvolutionBackwardDataGetSolutionWorkspaceSize(
      handle, dy_desc, w_desc, conv_desc, dx_desc, solution, &size_in_bytes));
  return size_in_bytes;
}

llvm::Error MiopenConvolutionBackwardDataImmediate(
    CurrentContext current, miopenHandle_t handle,
    miopenTensorDescriptor_t dy_desc, Pointer<const void> dy,
    miopenTensorDescriptor_t w_desc, Pointer<const void> w,
    miopenConvolutionDescriptor_t conv_desc, miopenTensorDescriptor_t dx_desc,
    Pointer<void> dx, Pointer<void> work_space, size_t work_space_size_in_bytes,
    uint64_t solution) {
  CheckHipContext(current);
  return TO_ERROR(miopenConvolutionBackwardDataImmediate(
      handle, dy_desc, ToRocm(dy), w_desc, ToRocm(w), conv_desc, dx_desc,
      ToRocm(dx), ToRocm(work_space), work_space_size_in_bytes, solution));
}

llvm::Expected<size_t> MiopenConvolutionBackwardWeightsGetSolutionCount(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t x_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dw_desc) {
  size_t count;
  RETURN_IF_ERROR(miopenConvolutionBackwardWeightsGetSolutionCount(
      handle, dy_desc, x_desc, conv_desc, dw_desc, &count));
  return count;
}

llvm::Expected<llvm::SmallVector<miopenConvSolution_t, 1>>
MiopenConvolutionBackwardWeightsGetSolution(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t x_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dw_desc, size_t solution_count) {
  llvm::SmallVector<miopenConvSolution_t, 1> solutions(solution_count);
  RETURN_IF_ERROR(miopenConvolutionBackwardWeightsGetSolution(
      handle, dy_desc, x_desc, conv_desc, dw_desc, solution_count,
      &solution_count, solutions.data()));
  solutions.resize(solution_count);
  return solutions;
}

llvm::Expected<size_t> MiopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
    miopenHandle_t handle, miopenTensorDescriptor_t dy_desc,
    miopenTensorDescriptor_t x_desc, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t dw_desc, uint64_t solution) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
      handle, dy_desc, x_desc, conv_desc, dw_desc, solution, &size_in_bytes));
  return size_in_bytes;
}

llvm::Error MiopenConvolutionBackwardWeightsImmediate(
    CurrentContext current, miopenHandle_t handle,
    miopenTensorDescriptor_t dy_desc, Pointer<const void> dy,
    miopenTensorDescriptor_t x_desc, Pointer<const void> x,
    miopenConvolutionDescriptor_t conv_desc, miopenTensorDescriptor_t dw_desc,
    Pointer<void> dw, Pointer<void> work_space, size_t work_space_size_in_bytes,
    uint64_t solution) {
  CheckHipContext(current);
  return TO_ERROR(miopenConvolutionBackwardWeightsImmediate(
      handle, dy_desc, ToRocm(dy), x_desc, ToRocm(x), conv_desc, dw_desc,
      ToRocm(dw), ToRocm(work_space), work_space_size_in_bytes, solution));
}

llvm::Expected<OwningDnnPoolingDescriptor> MiopenCreatePoolingDescriptor() {
  miopenPoolingDescriptor_t pooling_desc;
  RETURN_IF_ERROR(miopenCreatePoolingDescriptor(&pooling_desc));
  return OwningDnnPoolingDescriptor(pooling_desc);
}

llvm::Error MiopenDestroyPoolingDescriptor(
    miopenPoolingDescriptor_t descriptor) {
  return TO_ERROR(miopenDestroyPoolingDescriptor(descriptor));
}

llvm::Error MiopenSetPoolingDescriptor(miopenPoolingDescriptor_t descriptor,
                                       miopenPoolingMode_t mode,
                                       llvm::ArrayRef<int> window_dimensions,
                                       llvm::ArrayRef<int> paddings,
                                       llvm::ArrayRef<int> strides) {
  if (window_dimensions.size() != paddings.size() ||
      paddings.size() != strides.size()) {
    return MakeStringError(
        "Expected window dimension, padding, and stride arrays of equal size");
  }
  return TO_ERROR(miopenSetNdPoolingDescriptor(
      descriptor, mode, window_dimensions.size(),
      const_cast<int*>(window_dimensions.data()),
      const_cast<int*>(paddings.data()), const_cast<int*>(strides.data())));
}

llvm::Expected<DnnPoolingDescriptorData> MiopenGetPoolingDescriptor(
    const miopenPoolingDescriptor_t descriptor) {
  miopenPoolingMode_t mode;
  DnnPoolingDescriptorData data;
  int rank = 0;
  data.window_dimensions.resize(kDnnDimMax());
  data.paddings.resize(kDnnDimMax());
  data.strides.resize(kDnnDimMax());
  RETURN_IF_ERROR(miopenGetNdPoolingDescriptor(
      descriptor, kDnnDimMax(), &mode, &rank, data.window_dimensions.data(),
      data.paddings.data(), data.strides.data()));
  data.mode = static_cast<DnnPoolingMode>(mode);
  data.window_dimensions.resize(rank);
  data.paddings.resize(rank);
  data.strides.resize(rank);
  return data;
}

llvm::Expected<llvm::SmallVector<int, kDnnDimMax()>>
MiopenGetPoolingForwardOutputDim(
    const miopenPoolingDescriptor_t pooling_desc,
    const miopenTensorDescriptor_t input_tensor_desc) {
  llvm::SmallVector<int, kDnnDimMax()> output_dim(kDnnDimMax());
  RETURN_IF_ERROR(miopenGetPoolingNdForwardOutputDim(
      pooling_desc, input_tensor_desc, kDnnDimMax(), output_dim.data()));
  return output_dim;
}

llvm::Error MiopenPoolingForward(
    CurrentContext current, miopenHandle_t handle,
    const miopenPoolingDescriptor_t pooling_desc, Pointer<const void> alpha,
    const miopenTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, const miopenTensorDescriptor_t y_desc,
    Pointer<void> y, bool do_backward, Pointer<void> workspace,
    size_t workspace_size_bytes) {
  CheckHipContext(current);
  return TO_ERROR(miopenPoolingForward(
      handle, pooling_desc, ToRocm(alpha), x_desc, ToRocm(x), ToRocm(beta),
      y_desc, ToRocm(y), do_backward, ToRocm(workspace), workspace_size_bytes));
}

llvm::Expected<size_t> MiopenPoolingGetWorkSpaceSize(
    const miopenPoolingDescriptor_t pooling_desc,
    const miopenTensorDescriptor_t y_desc) {
  size_t workspace_bytes;
  RETURN_IF_ERROR(
      miopenPoolingGetWorkSpaceSizeV2(pooling_desc, y_desc, &workspace_bytes));
  return workspace_bytes;
}

llvm::Error MiopenPoolingBackward(
    CurrentContext current, miopenHandle_t handle,
    const miopenPoolingDescriptor_t pooling_desc, Pointer<const void> alpha,
    const miopenTensorDescriptor_t y_desc, Pointer<const void> y,
    const miopenTensorDescriptor_t dy_desc, Pointer<const void> dy,
    const miopenTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, const miopenTensorDescriptor_t dx_desc,
    Pointer<void> dx, Pointer<void> workspace) {
  CheckHipContext(current);
  return TO_ERROR(miopenPoolingBackward(
      handle, pooling_desc, ToRocm(alpha), y_desc, ToRocm(y), dy_desc,
      ToRocm(dy), x_desc, ToRocm(x), ToRocm(beta), dx_desc, ToRocm(dx),
      ToRocm(workspace)));
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
