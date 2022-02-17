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

// Thin wrapper around the cuDNN API adding llvm::Error.
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"

#include <cstdlib>
#include <cstring>

#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/support/logging.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

// Returns reference to API log of the last cuDNN API call.
static std::string& CudnnLogTop() {
  thread_local auto string = new std::string;
  return *string;
}

llvm::raw_ostream& internal::operator<<(llvm::raw_ostream& os,
                                        const ErrorData<cudnnStatus_t>& data) {
  operator<< /**/<cudnnStatus_t>(os, data);
  if (data.log.empty()) return os;
  if (!data.stack_trace) os << '\n';
  return os << data.log;
}

Error MakeError(cudnnStatus_t result, const char* expr) {
  return internal::MakeError(result, expr, std::move(CudnnLogTop()));
}

// cuDNN API logging callback.
static void CudnnCallback(cudnnSeverity_t /*severity*/, void* /*user_data*/,
                          const cudnnDebug_t* /*dbg*/, const char* msg) {
  CudnnLogTop().clear();
  llvm::raw_string_ostream os(CudnnLogTop());
  // Each line in msg is '\0' terminated, the message ends with '\0\0'.
  while (*msg) {
    os << '\n' << msg;
    msg += std::strlen(msg) + 1;
  }
}

void internal::CudnnPersistentRnnPlanDeleter::operator()(
    cudnnPersistentRNNPlan_t plan) const {
  LogIfError(CudnnDestroyPersistentRnnPlan(plan));
}

llvm::Expected<cudnnStatus_t> CudnnQueryRuntimeError(cudnnHandle_t handle,
                                                     cudnnErrQueryMode_t mode,
                                                     cudnnRuntimeTag_t* tag) {
  cudnnStatus_t rstatus;
  RETURN_IF_ERROR(cudnnQueryRuntimeError(handle, &rstatus, mode, tag));
  return rstatus;
}

llvm::Expected<LibraryVersion> CudnnGetVersion() {
  LibraryVersion version;
  RETURN_IF_ERROR(
      cudnnGetProperty(libraryPropertyType::MAJOR_VERSION, &version.major));
  RETURN_IF_ERROR(
      cudnnGetProperty(libraryPropertyType::MINOR_VERSION, &version.minor));
  RETURN_IF_ERROR(
      cudnnGetProperty(libraryPropertyType::PATCH_LEVEL, &version.patch));
  return version;
}

llvm::Expected<OwningDnnHandle> CudnnCreate(CurrentContext current) {
  CheckCudaContext(current);
  auto set_callback_result = [] {
    auto env_contains = [](const char* key, const char* value) {
      const char* env = std::getenv(key);
      return env && !std::strcmp(env, value);
    };

    // Do not register a callback unless 'CUDNN_LOGDEST_DBG=tfrt' to avoid the
    // performance penalty pre cuDNN 8.1 which ignored CUDNN_LOGDEST_DBG.
    if (!env_contains("CUDNN_LOGDEST_DBG", "tfrt")) return CUDNN_STATUS_SUCCESS;

    // Warn when 'CUDNN_LOGINFO_DBG=1' because the user likely does not want to
    // to write the log to the 'tfrt' file, which is the bevior starting with
    // cuDNN 8.2.1.
    if (env_contains("CUDNN_LOGINFO_DBG", "1")) {
      TFRT_LOG(WARNING)
          << "CUDNN_LOGDEST_DBG=tfrt should not be combined with "
             "CUDNN_LOGINFO_DBG=1, cuDNN logs will be written to 'tfrt' file.";
    }

    // Note: For the callback to get triggered, CUDNN_LOGDEST_DBG must be set
    // and CUDNN_LOGINFO_DBG must not be 1.
    return cudnnSetCallback(/*mask=*/~0, nullptr, CudnnCallback);
  }();
  cudnnHandle_t handle = nullptr;
  RETURN_IF_ERROR(cudnnCreate(&handle));
  RETURN_IF_ERROR(set_callback_result);
  return OwningDnnHandle(handle);
}

llvm::Error CudnnDestroy(cudnnHandle_t handle) {
  return TO_ERROR(cudnnDestroy(handle));
}

llvm::Error CudnnSetStream(cudnnHandle_t handle, cudaStream_t stream) {
  return TO_ERROR(cudnnSetStream(handle, stream));
}

llvm::Expected<cudaStream_t> CudnnGetStream(cudnnHandle_t handle) {
  cudaStream_t stream = nullptr;
  RETURN_IF_ERROR(cudnnGetStream(handle, &stream));
  return stream;
}

llvm::Expected<OwningDnnTensorDescriptor> CudnnCreateTensorDescriptor() {
  cudnnTensorDescriptor_t descriptor = nullptr;
  RETURN_IF_ERROR(cudnnCreateTensorDescriptor(&descriptor));
  return OwningDnnTensorDescriptor(descriptor);
}

llvm::Error CudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t descriptor) {
  return TO_ERROR(cudnnDestroyTensorDescriptor(descriptor));
}

// Returns the dimensions of a CUDNN_TENSOR_NCHW_VECT_C tensor if the arguments
// describe such a tensor, llvm::None otherwise.
llvm::Optional<llvm::SmallVector<int, 4>> GetNchwVectDimensions(
    cudnnDataType_t data_type, llvm::ArrayRef<int> dimensions,
    llvm::ArrayRef<int> strides) {
  switch (data_type) {
    case CUDNN_DATA_INT8x4:
    case CUDNN_DATA_UINT8x4:
    case CUDNN_DATA_INT8x32:
      break;
    default:
      return llvm::None;
  }
  // Test whether format is NCHW (i.e. dense row-major).
  int stride = 1;
  assert(dimensions.size() == strides.size());
  for (int i = dimensions.size() - 1; i >= 0; --i) {
    if (strides[i] != stride) return llvm::None;
    stride *= dimensions[i];
  }
  auto result = llvm::to_vector<4>(dimensions);
  if (result.size() < 2) return llvm::None;
  // Multiply channel count by vector width.
  result[1] *= data_type == CUDNN_DATA_INT8x32 ? 32 : 4;
  return result;
}

llvm::Error CudnnSetTensorDescriptor(cudnnTensorDescriptor_t descriptor,
                                     cudnnDataType_t data_type,
                                     llvm::ArrayRef<int> dimensions,
                                     llvm::ArrayRef<int> strides) {
  if (dimensions.size() != strides.size())
    return MakeStringError("Expected dimensions and strides to be equal size");

  if (auto nhwc_dims = GetNchwVectDimensions(data_type, dimensions, strides)) {
    // cuDNN (v8.2) reports unsupported (IMPLICIT_PRECOMP_GEMM) convolutions for
    // vectorized tensor descriptors initialized from the non-Ex API below.
    // Detect CUDNN_TENSOR_NCHW_VECT_C tensors and use the Ex API instead.
    return TO_ERROR(cudnnSetTensorNdDescriptorEx(
        descriptor, CUDNN_TENSOR_NCHW_VECT_C, data_type, nhwc_dims->size(),
        nhwc_dims->data()));
  }

  return TO_ERROR(
      cudnnSetTensorNdDescriptor(descriptor, data_type, dimensions.size(),
                                 dimensions.data(), strides.data()));
}

llvm::Expected<DnnTensorDescriptorData> CudnnGetTensorDescriptor(
    cudnnTensorDescriptor_t descriptor) {
  cudnnDataType_t data_type;
  DnnTensorDescriptorData data;
  int rank = 0;
  data.dimensions.resize(kDnnDimMax());
  data.strides.resize(kDnnDimMax());
  RETURN_IF_ERROR(
      cudnnGetTensorNdDescriptor(descriptor, kDnnDimMax(), &data_type, &rank,
                                 data.dimensions.data(), data.strides.data()));
  data.data_type = data_type;
  data.dimensions.resize(rank);
  data.strides.resize(rank);
  return data;
}

llvm::Expected<size_t> CudnnGetTensorSizeInBytes(
    cudnnTensorDescriptor_t descriptor) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(cudnnGetTensorSizeInBytes(descriptor, &size_in_bytes));
  return size_in_bytes;
}

llvm::Expected<cudnnTensorTransformDescriptor_t>
CudnnCreateTensorTransformDescriptor() {
  cudnnTensorTransformDescriptor_t transform_desc;
  RETURN_IF_ERROR(cudnnCreateTensorTransformDescriptor(&transform_desc));
  return transform_desc;
}

llvm::Error CudnnDestroyTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t descriptor) {
  return TO_ERROR(cudnnDestroyTensorTransformDescriptor(descriptor));
}

llvm::Error CudnnSetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t descriptor,
    cudnnTensorFormat_t dest_format, llvm::ArrayRef<int> pad_before,
    llvm::ArrayRef<int> pad_after, llvm::ArrayRef<unsigned> fold,
    cudnnFoldingDirection_t direction) {
  if (pad_before.size() != pad_after.size()) {
    return MakeStringError(
        "Expected before and after padding arrays of equal size");
  }
  if (!fold.empty() && fold.size() + 2 != pad_before.size()) {
    return MakeStringError(
        "Expected fold array to be empty or 2 "
        "elements shorter than padding arrays");
  }
  return TO_ERROR(cudnnSetTensorTransformDescriptor(
      descriptor, pad_before.size(), dest_format, pad_before.data(),
      pad_after.data(), fold.empty() ? nullptr : fold.data(), direction));
}

llvm::Expected<CudnnTransformDescriptorData> CudnnGetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t descriptor, uint32_t rank) {
  CudnnTransformDescriptorData data;
  data.paddings_after.resize(rank);
  data.paddings_before.resize(rank);
  data.fold.resize(rank - 2);
  RETURN_IF_ERROR(cudnnGetTensorTransformDescriptor(
      descriptor, rank, &data.destination_format, data.paddings_before.data(),
      data.paddings_after.data(), data.fold.data(), &data.direction));
  return data;
}

llvm::Error CudnnTransformTensor(
    CurrentContext current, cudnnHandle_t handle, Pointer<const void> alpha,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, cudnnTensorDescriptor_t y_desc, Pointer<void> y) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnTransformTensor(handle, ToCuda(alpha), x_desc, ToCuda(x),
                                       ToCuda(beta), y_desc, ToCuda(y)));
}

llvm::Error CudnnTransformTensor(CurrentContext current, cudnnHandle_t handle,
                                 cudnnTensorTransformDescriptor_t trans_desc,
                                 Pointer<const void> alpha,
                                 cudnnTensorDescriptor_t src_desc,
                                 Pointer<const void> src_data,
                                 Pointer<const void> beta,
                                 cudnnTensorDescriptor_t dest_desc,
                                 Pointer<void> dest_data) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnTransformTensorEx(
      handle, trans_desc, ToCuda(alpha), src_desc, ToCuda(src_data),
      ToCuda(beta), dest_desc, ToCuda(dest_data)));
}

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
    cudnnTensorTransformDescriptor_t grad_unfold_trans_desc) {
  return TO_ERROR(cudnnGetFoldedConvBackwardDataDescriptors(
      handle, filter_desc, diff_desc, conv_desc, grad_desc, transform_format,
      folded_filter_desc, padded_diff_desc, folded_conv_desc, folded_grad_desc,
      filter_fold_trans_desc, diff_pad_trans_desc, grad_fold_trans_desc,
      grad_unfold_trans_desc));
}

llvm::Error CudnnAddTensor(CurrentContext current, cudnnHandle_t handle,
                           Pointer<const void> alpha,
                           cudnnTensorDescriptor_t a_desc,
                           Pointer<const void> a, Pointer<const void> beta,
                           cudnnTensorDescriptor_t c_desc, Pointer<void> c) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnAddTensor(handle, ToCuda(alpha), a_desc, ToCuda(a),
                                 ToCuda(beta), c_desc, ToCuda(c)));
}

llvm::Expected<cudnnOpTensorDescriptor_t> CudnnCreateOpTensorDescriptor() {
  cudnnOpTensorDescriptor_t op_tensor_desc;
  RETURN_IF_ERROR(cudnnCreateOpTensorDescriptor(&op_tensor_desc));
  return op_tensor_desc;
}

llvm::Error CudnnSetOpTensorDescriptor(
    cudnnOpTensorDescriptor_t op_tensor_desc, cudnnOpTensorOp_t op_tensor_op,
    cudnnDataType_t op_tensor_comp_type,
    cudnnNanPropagation_t op_tensor_nan_opt) {
  return TO_ERROR(cudnnSetOpTensorDescriptor(
      op_tensor_desc, op_tensor_op, op_tensor_comp_type, op_tensor_nan_opt));
}

llvm::Expected<CudnnOpTensorDescriptorData> CudnnGetOpTensorDescriptor(
    cudnnOpTensorDescriptor_t descriptor) {
  CudnnOpTensorDescriptorData data;
  RETURN_IF_ERROR(cudnnGetOpTensorDescriptor(
      descriptor, &data.op, &data.math_type, &data.nan_propagation));
  return data;
}

llvm::Error CudnnDestroyOpTensorDescriptor(
    cudnnOpTensorDescriptor_t descriptor) {
  return TO_ERROR(cudnnDestroyOpTensorDescriptor(descriptor));
}

llvm::Error CudnnOpTensor(CurrentContext current, cudnnHandle_t handle,
                          cudnnOpTensorDescriptor_t op_tensor_desc,
                          Pointer<const void> alpha1,
                          cudnnTensorDescriptor_t a_desc, Pointer<const void> a,
                          Pointer<const void> alpha2,
                          cudnnTensorDescriptor_t b_desc, Pointer<const void> b,
                          Pointer<const void> beta,
                          cudnnTensorDescriptor_t c_desc, Pointer<void> c) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnOpTensor(handle, op_tensor_desc, ToCuda(alpha1), a_desc,
                                ToCuda(a), ToCuda(alpha2), b_desc, ToCuda(b),
                                ToCuda(beta), c_desc, ToCuda(c)));
}

llvm::Expected<cudnnReduceTensorDescriptor_t>
CudnnCreateReduceTensorDescriptor() {
  cudnnReduceTensorDescriptor_t reduce_tensor_desc;
  RETURN_IF_ERROR(cudnnCreateReduceTensorDescriptor(&reduce_tensor_desc));
  return reduce_tensor_desc;
}

llvm::Error CudnnSetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t descriptor,
    cudnnReduceTensorOp_t reduce_tensor_op,
    cudnnDataType_t reduce_tensor_comp_type,
    cudnnNanPropagation_t reduce_tensor_nan_opt,
    cudnnReduceTensorIndices_t reduce_tensor_indices,
    cudnnIndicesType_t reduce_tensor_indices_type) {
  return TO_ERROR(cudnnSetReduceTensorDescriptor(
      descriptor, reduce_tensor_op, reduce_tensor_comp_type,
      reduce_tensor_nan_opt, reduce_tensor_indices,
      reduce_tensor_indices_type));
}

llvm::Expected<CudnnReduceTensorDescriptorData> CudnnGetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t descriptor) {
  CudnnReduceTensorDescriptorData data;
  cudnnReduceTensorIndices_t compute_indices;
  RETURN_IF_ERROR(cudnnGetReduceTensorDescriptor(
      descriptor, &data.op, &data.math_type, &data.nan_propagation,
      &compute_indices, &data.index_type));
  data.compute_indices =
      CUDNN_REDUCE_TENSOR_FLATTENED_INDICES == compute_indices;
  return data;
}

llvm::Error CudnnDestroyReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t descriptor) {
  return TO_ERROR(cudnnDestroyReduceTensorDescriptor(descriptor));
}

llvm::Expected<size_t> CudnnGetReductionIndicesSize(
    cudnnHandle_t handle, cudnnReduceTensorDescriptor_t reduce_tensor_desc,
    cudnnTensorDescriptor_t a_desc, cudnnTensorDescriptor_t c_desc) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(cudnnGetReductionIndicesSize(handle, reduce_tensor_desc,
                                               a_desc, c_desc, &size_in_bytes));
  return size_in_bytes;
}

llvm::Expected<size_t> CudnnGetReductionWorkspaceSize(
    cudnnHandle_t handle, cudnnReduceTensorDescriptor_t reduce_tensor_desc,
    cudnnTensorDescriptor_t a_desc, cudnnTensorDescriptor_t c_desc) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(
      handle, reduce_tensor_desc, a_desc, c_desc, &size_in_bytes));
  return size_in_bytes;
}

llvm::Error CudnnReduceTensor(
    CurrentContext current, cudnnHandle_t handle,
    cudnnReduceTensorDescriptor_t reduce_tensor_desc, Pointer<void> indices,
    size_t indices_size_in_bytes, Pointer<void> workspace,
    size_t workspace_size_in_bytes, Pointer<const void> alpha,
    cudnnTensorDescriptor_t a_desc, Pointer<const void> a,
    Pointer<const void> beta, cudnnTensorDescriptor_t c_desc, Pointer<void> c) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnReduceTensor(
      handle, reduce_tensor_desc, ToCuda(indices), indices_size_in_bytes,
      ToCuda(workspace), workspace_size_in_bytes, ToCuda(alpha), a_desc,
      ToCuda(a), ToCuda(beta), c_desc, ToCuda(c)));
}

llvm::Error CudnnSetTensor(CurrentContext current, cudnnHandle_t handle,
                           cudnnTensorDescriptor_t y_desc, Pointer<void> y,
                           Pointer<const void> value_ptr) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnSetTensor(handle, y_desc, ToCuda(y), ToCuda(value_ptr)));
}

llvm::Error CudnnScaleTensor(CurrentContext current, cudnnHandle_t handle,
                             cudnnTensorDescriptor_t y_desc, Pointer<void> y,
                             Pointer<const void> alpha) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnScaleTensor(handle, y_desc, ToCuda(y), ToCuda(alpha)));
}

llvm::Expected<OwningDnnFilterDescriptor> CudnnCreateFilterDescriptor() {
  cudnnFilterDescriptor_t descriptor = nullptr;
  RETURN_IF_ERROR(cudnnCreateFilterDescriptor(&descriptor));
  return OwningDnnFilterDescriptor(descriptor);
}

llvm::Error CudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t descriptor) {
  return TO_ERROR(cudnnDestroyFilterDescriptor(descriptor));
}

llvm::Error CudnnSetFilterDescriptor(cudnnFilterDescriptor_t descriptor,
                                     cudnnDataType_t data_type,
                                     cudnnTensorFormat_t format,
                                     llvm::ArrayRef<int> dimensions) {
  return TO_ERROR(cudnnSetFilterNdDescriptor(
      descriptor, data_type, format, dimensions.size(), dimensions.data()));
}

llvm::Expected<CudnnFilterDescriptorData> CudnnGetFilterDescriptor(
    cudnnFilterDescriptor_t descriptor) {
  CudnnFilterDescriptorData data;
  int rank = 0;
  data.dimensions.resize(kDnnDimMax());
  RETURN_IF_ERROR(cudnnGetFilterNdDescriptor(descriptor, kDnnDimMax(),
                                             &data.data_type, &data.format,
                                             &rank, data.dimensions.data()));
  data.dimensions.resize(rank);
  return data;
}

llvm::Expected<size_t> CudnnGetFilterSizeInBytes(
    cudnnFilterDescriptor_t descriptor) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(cudnnGetFilterSizeInBytes(descriptor, &size_in_bytes));
  return size_in_bytes;
}

llvm::Error CudnnTransformFilter(CurrentContext current, cudnnHandle_t handle,
                                 cudnnTensorTransformDescriptor_t trans_desc,
                                 Pointer<const void> alpha,
                                 cudnnFilterDescriptor_t src_desc,
                                 Pointer<const void> src_data,
                                 Pointer<const void> beta,
                                 cudnnFilterDescriptor_t dest_desc,
                                 Pointer<void> dest_data) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnTransformFilter(handle, trans_desc, ToCuda(alpha),
                                       src_desc, ToCuda(src_data), ToCuda(beta),
                                       dest_desc, ToCuda(dest_data)));
}

llvm::Error CudnnReorderFilterAndBias(
    CurrentContext current, cudnnHandle_t handle,
    cudnnFilterDescriptor_t descriptor, cudnnReorderType_t reorder_type,
    Pointer<const void> filter_data, Pointer<void> reordered_filter_data,
    int reorder_bias, Pointer<const void> bias_data,
    Pointer<void> reordered_bias_data) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnReorderFilterAndBias(
      handle, descriptor, reorder_type, ToCuda(filter_data),
      ToCuda(reordered_filter_data), reorder_bias, ToCuda(bias_data),
      ToCuda(reordered_bias_data)));
}

llvm::Expected<OwningDnnConvolutionDescriptor>
CudnnCreateConvolutionDescriptor() {
  cudnnConvolutionDescriptor_t conv_desc;
  RETURN_IF_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
  return OwningDnnConvolutionDescriptor(conv_desc);
}

llvm::Error CudnnDestroyConvolutionDescriptor(
    cudnnConvolutionDescriptor_t descriptor) {
  return TO_ERROR(cudnnDestroyConvolutionDescriptor(descriptor));
}

llvm::Error CudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t descriptor,
                                        cudnnMathType_t math_type) {
  return TO_ERROR(cudnnSetConvolutionMathType(descriptor, math_type));
}

llvm::Expected<cudnnMathType_t> CudnnGetConvolutionMathType(
    cudnnConvolutionDescriptor_t descriptor) {
  cudnnMathType_t math_type;
  RETURN_IF_ERROR(cudnnGetConvolutionMathType(descriptor, &math_type));
  return math_type;
}

llvm::Error CudnnSetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t descriptor, int group_count) {
  return TO_ERROR(cudnnSetConvolutionGroupCount(descriptor, group_count));
}

llvm::Expected<int> CudnnGetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t descriptor) {
  int group_count;
  RETURN_IF_ERROR(cudnnGetConvolutionGroupCount(descriptor, &group_count));
  return group_count;
}

llvm::Error CudnnSetConvolutionReorderType(
    cudnnConvolutionDescriptor_t descriptor, cudnnReorderType_t reorder_type) {
  return TO_ERROR(cudnnSetConvolutionReorderType(descriptor, reorder_type));
}

llvm::Expected<cudnnReorderType_t> CudnnGetConvolutionReorderType(
    cudnnConvolutionDescriptor_t descriptor) {
  cudnnReorderType_t reorder_type;
  RETURN_IF_ERROR(cudnnGetConvolutionReorderType(descriptor, &reorder_type));
  return reorder_type;
}

llvm::Error CudnnSetConvolutionDescriptor(
    cudnnConvolutionDescriptor_t descriptor, llvm::ArrayRef<int> pad,
    llvm::ArrayRef<int> filter_stride, llvm::ArrayRef<int> dilation,
    cudnnConvolutionMode_t mode, cudnnDataType_t compute_type) {
  if (pad.size() != filter_stride.size() || pad.size() != dilation.size()) {
    return MakeStringError(
        "Expected paddings, filter_strides and dilations arrays of equal size");
  }
  return TO_ERROR(cudnnSetConvolutionNdDescriptor(
      descriptor, pad.size(), pad.data(), filter_stride.data(), dilation.data(),
      mode, compute_type));
}

llvm::Expected<DnnConvolutionDescriptorData> CudnnGetConvolutionDescriptor(
    cudnnConvolutionDescriptor_t descriptor) {
  int rank = 0;
  cudnnConvolutionMode_t mode;
  cudnnDataType_t math_type;
  DnnConvolutionDescriptorData data;
  data.paddings.resize(kDnnDimMax());
  data.filter_strides.resize((kDnnDimMax()));
  data.dilations.resize(kDnnDimMax());
  RETURN_IF_ERROR(cudnnGetConvolutionNdDescriptor(
      descriptor, kDnnDimMax(), &rank, data.paddings.data(),
      data.filter_strides.data(), data.dilations.data(), &mode, &math_type));
  data.paddings.resize(rank);
  data.filter_strides.resize(rank);
  data.dilations.resize(rank);
  data.mode = static_cast<DnnConvolutionMode>(mode);
  data.math_type = math_type;
  return data;
}

llvm::Expected<llvm::SmallVector<int, kDnnDimMax()>>
CudnnGetConvolutionForwardOutputDim(cudnnConvolutionDescriptor_t conv_desc,
                                    cudnnTensorDescriptor_t input_tensor_desc,
                                    cudnnFilterDescriptor_t filter_desc) {
  llvm::SmallVector<int, kDnnDimMax()> output_dim(kDnnDimMax());
  RETURN_IF_ERROR(cudnnGetConvolutionNdForwardOutputDim(
      conv_desc, input_tensor_desc, filter_desc, kDnnDimMax(),
      output_dim.data()));
  return output_dim;
}

llvm::Expected<int> CudnnGetConvolutionForwardAlgorithmMaxCount(
    cudnnHandle_t handle) {
  int count;
  RETURN_IF_ERROR(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &count));
  return count;
}

llvm::Expected<llvm::SmallVector<cudnnConvolutionFwdAlgoPerf_t, 1>>
CudnnFindConvolutionForwardAlgorithm(
    CurrentContext current, cudnnHandle_t handle,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnFilterDescriptor_t w_desc, Pointer<const void> w,
    cudnnConvolutionDescriptor_t conv_desc, cudnnTensorDescriptor_t y_desc,
    Pointer<void> y, int algo_count, Pointer<void> work_space,
    size_t work_space_size_in_bytes) {
  CheckCudaContext(current);
  llvm::SmallVector<cudnnConvolutionFwdAlgoPerf_t, 1> perf_results(algo_count);
  RETURN_IF_ERROR(cudnnFindConvolutionForwardAlgorithmEx(
      handle, x_desc, ToCuda(x), w_desc, ToCuda(w), conv_desc, y_desc,
      ToCuda(y), algo_count, &algo_count, perf_results.data(),
      ToCuda(work_space), work_space_size_in_bytes));
  perf_results.resize(algo_count);
  return perf_results;
}

llvm::Expected<llvm::SmallVector<cudnnConvolutionFwdAlgoPerf_t, 1>>
CudnnGetConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                    cudnnTensorDescriptor_t src_desc,
                                    cudnnFilterDescriptor_t filter_desc,
                                    cudnnConvolutionDescriptor_t conv_desc,
                                    cudnnTensorDescriptor_t dest_desc,
                                    int algo_count) {
  llvm::SmallVector<cudnnConvolutionFwdAlgoPerf_t, 1> perf_results(algo_count);
  RETURN_IF_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(
      handle, src_desc, filter_desc, conv_desc, dest_desc, algo_count,
      &algo_count, perf_results.data()));
  perf_results.resize(algo_count);
  return perf_results;
}

llvm::Expected<size_t> CudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, cudnnTensorDescriptor_t x_desc,
    cudnnFilterDescriptor_t w_desc, cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t y_desc, cudnnConvolutionFwdAlgo_t algo) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
      handle, x_desc, w_desc, conv_desc, y_desc, algo, &size_in_bytes));
  return size_in_bytes;
}

llvm::Error CudnnConvolutionForward(
    CurrentContext current, cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnFilterDescriptor_t w_desc, Pointer<const void> w,
    cudnnConvolutionDescriptor_t conv_desc, cudnnConvolutionFwdAlgo_t algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes, const void* beta,
    cudnnTensorDescriptor_t y_desc, Pointer<void> y) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnConvolutionForward(
      handle, alpha, x_desc, ToCuda(x), w_desc, ToCuda(w), conv_desc, algo,
      ToCuda(work_space), work_space_size_in_bytes, beta, y_desc, ToCuda(y)));
}

llvm::Error CudnnConvolutionBiasActivationForward(
    CurrentContext current, cudnnHandle_t handle, const void* alpha1,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnFilterDescriptor_t w_desc, Pointer<const void> w,
    cudnnConvolutionDescriptor_t conv_desc, cudnnConvolutionFwdAlgo_t algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes,
    const void* alpha2, cudnnTensorDescriptor_t z_desc, Pointer<const void> z,
    cudnnTensorDescriptor_t bias_desc, Pointer<const void> bias,
    cudnnActivationDescriptor_t activation_desc, cudnnTensorDescriptor_t y_desc,
    Pointer<void> y) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnConvolutionBiasActivationForward(
      handle, alpha1, x_desc, ToCuda(x), w_desc, ToCuda(w), conv_desc, algo,
      ToCuda(work_space), work_space_size_in_bytes, alpha2, z_desc, ToCuda(z),
      bias_desc, ToCuda(bias), activation_desc, y_desc, ToCuda(y)));
}

llvm::Error CudnnConvolutionBackwardBias(
    CurrentContext current, cudnnHandle_t handle, Pointer<const void> alpha,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    Pointer<const void> beta, cudnnTensorDescriptor_t db_desc,
    Pointer<void> db) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnConvolutionBackwardBias(handle, ToCuda(alpha), dy_desc,
                                               ToCuda(dy), ToCuda(beta),
                                               db_desc, ToCuda(db)));
}

llvm::Expected<int> CudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
    cudnnHandle_t handle) {
  int count;
  RETURN_IF_ERROR(
      cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, &count));
  return count;
}

llvm::Expected<llvm::SmallVector<cudnnConvolutionBwdFilterAlgoPerf_t, 1>>
CudnnFindConvolutionBackwardFilterAlgorithm(
    CurrentContext current, cudnnHandle_t handle,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> y,
    cudnnConvolutionDescriptor_t conv_desc, cudnnFilterDescriptor_t dw_desc,
    Pointer<void> dw, int algo_count, Pointer<void> work_space,
    size_t work_space_size_in_bytes) {
  CheckCudaContext(current);
  llvm::SmallVector<cudnnConvolutionBwdFilterAlgoPerf_t, 1> perf_results;
  RETURN_IF_ERROR(cudnnFindConvolutionBackwardFilterAlgorithmEx(
      handle, x_desc, ToCuda(x), dy_desc, ToCuda(y), conv_desc, dw_desc,
      ToCuda(dw), algo_count, &algo_count, perf_results.data(),
      ToCuda(work_space), work_space_size_in_bytes));
  perf_results.resize(algo_count);
  return perf_results;
}

llvm::Expected<llvm::SmallVector<cudnnConvolutionBwdFilterAlgoPerf_t, 1>>
CudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, cudnnTensorDescriptor_t src_desc,
    cudnnTensorDescriptor_t diff_desc, cudnnConvolutionDescriptor_t conv_desc,
    cudnnFilterDescriptor_t grad_desc, int algo_count) {
  llvm::SmallVector<cudnnConvolutionBwdFilterAlgoPerf_t, 1> perf_results;
  RETURN_IF_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
      handle, src_desc, diff_desc, conv_desc, grad_desc, algo_count,
      &algo_count, perf_results.data()));
  perf_results.resize(algo_count);
  return perf_results;
}

llvm::Expected<size_t> CudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, cudnnTensorDescriptor_t x_desc,
    cudnnTensorDescriptor_t dy_desc, cudnnConvolutionDescriptor_t conv_desc,
    cudnnFilterDescriptor_t grad_desc, cudnnConvolutionBwdFilterAlgo_t algo) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle, x_desc, dy_desc, conv_desc, grad_desc, algo, &size_in_bytes));
  return size_in_bytes;
}

llvm::Error CudnnConvolutionBackwardFilter(
    CurrentContext current, cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnConvolutionBwdFilterAlgo_t algo, Pointer<void> work_space,
    size_t work_space_size_in_bytes, const void* beta,
    cudnnFilterDescriptor_t dw_desc, Pointer<void> dw) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnConvolutionBackwardFilter(
      handle, alpha, x_desc, ToCuda(x), dy_desc, ToCuda(dy), conv_desc, algo,
      ToCuda(work_space), work_space_size_in_bytes, beta, dw_desc, ToCuda(dw)));
}

llvm::Expected<int> CudnnGetConvolutionBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle) {
  int count;
  RETURN_IF_ERROR(
      cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, &count));
  return count;
}

llvm::Expected<llvm::SmallVector<cudnnConvolutionBwdDataAlgoPerf_t, 1>>
CudnnFindConvolutionBackwardDataAlgorithm(
    CurrentContext current, cudnnHandle_t handle,
    cudnnFilterDescriptor_t w_desc, Pointer<const void> w,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    cudnnConvolutionDescriptor_t conv_desc, cudnnTensorDescriptor_t dx_desc,
    Pointer<void> dx, int algo_count, Pointer<void> work_space,
    size_t work_space_size_in_bytes) {
  CheckCudaContext(current);
  llvm::SmallVector<cudnnConvolutionBwdDataAlgoPerf_t, 1> perf_results;
  RETURN_IF_ERROR(cudnnFindConvolutionBackwardDataAlgorithmEx(
      handle, w_desc, ToCuda(w), dy_desc, ToCuda(dy), conv_desc, dx_desc,
      ToCuda(dx), algo_count, &algo_count, perf_results.data(),
      ToCuda(work_space), work_space_size_in_bytes));
  perf_results.resize(algo_count);
  return perf_results;
}

llvm::Expected<llvm::SmallVector<cudnnConvolutionBwdDataAlgoPerf_t, 1>>
CudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                         cudnnFilterDescriptor_t filter_desc,
                                         cudnnTensorDescriptor_t diff_desc,
                                         cudnnConvolutionDescriptor_t conv_desc,
                                         cudnnTensorDescriptor_t grad_desc,
                                         int algo_count) {
  llvm::SmallVector<cudnnConvolutionBwdDataAlgoPerf_t, 1> perf_results;
  RETURN_IF_ERROR(cudnnGetConvolutionBackwardDataAlgorithm_v7(
      handle, filter_desc, diff_desc, conv_desc, grad_desc, algo_count,
      &algo_count, perf_results.data()));
  perf_results.resize(algo_count);
  return perf_results;
}

llvm::Expected<size_t> CudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, cudnnFilterDescriptor_t w_desc,
    cudnnTensorDescriptor_t dy_desc, cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t dx_desc, cudnnConvolutionBwdDataAlgo_t algo) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle, w_desc, dy_desc, conv_desc, dx_desc, algo, &size_in_bytes));
  return size_in_bytes;
}

llvm::Error CudnnConvolutionBackwardData(
    CurrentContext current, cudnnHandle_t handle, const void* alpha,
    cudnnFilterDescriptor_t w_desc, Pointer<const void> w,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    cudnnConvolutionDescriptor_t conv_desc, cudnnConvolutionBwdDataAlgo_t algo,
    Pointer<void> work_space, size_t work_space_size_in_bytes, const void* beta,
    cudnnTensorDescriptor_t dx_desc, Pointer<void> dx) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnConvolutionBackwardData(
      handle, alpha, w_desc, ToCuda(w), dy_desc, ToCuda(dy), conv_desc, algo,
      ToCuda(work_space), work_space_size_in_bytes, beta, dx_desc, ToCuda(dx)));
}

llvm::Error CudnnIm2Col(CurrentContext current, cudnnHandle_t handle,
                        cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
                        cudnnFilterDescriptor_t w_desc,
                        cudnnConvolutionDescriptor_t conv_desc,
                        Pointer<void> col_buffer) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnIm2Col(handle, x_desc, ToCuda(x), w_desc, conv_desc,
                              ToCuda(col_buffer)));
}

llvm::Error CudnnSoftmaxForward(
    CurrentContext current, cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, Pointer<const void> alpha,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, cudnnTensorDescriptor_t y_desc, Pointer<void> y) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnSoftmaxForward(handle, algo, mode, ToCuda(alpha), x_desc,
                                      ToCuda(x), ToCuda(beta), y_desc,
                                      ToCuda(y)));
}

llvm::Error CudnnSoftmaxBackward(
    CurrentContext current, cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, Pointer<const void> alpha,
    cudnnTensorDescriptor_t y_desc, Pointer<const void> y,
    cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    Pointer<const void> beta, cudnnTensorDescriptor_t dx_desc,
    Pointer<void> dx) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnSoftmaxBackward(handle, algo, mode, ToCuda(alpha),
                                       y_desc, ToCuda(y), dy_desc, ToCuda(dy),
                                       ToCuda(beta), dx_desc, ToCuda(dx)));
}

llvm::Expected<OwningDnnPoolingDescriptor> CudnnCreatePoolingDescriptor() {
  cudnnPoolingDescriptor_t pooling_desc;
  RETURN_IF_ERROR(cudnnCreatePoolingDescriptor(&pooling_desc));
  return OwningDnnPoolingDescriptor(pooling_desc);
}

llvm::Error CudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t descriptor) {
  return TO_ERROR(cudnnDestroyPoolingDescriptor(descriptor));
}

llvm::Error CudnnSetPoolingDescriptor(cudnnPoolingDescriptor_t descriptor,
                                      cudnnPoolingMode_t mode,
                                      cudnnNanPropagation_t nan_propagation,
                                      llvm::ArrayRef<int> window_dimensions,
                                      llvm::ArrayRef<int> paddings,
                                      llvm::ArrayRef<int> strides) {
  if (window_dimensions.size() != paddings.size() ||
      paddings.size() != strides.size()) {
    return MakeStringError(
        "Expected window dimension, padding, and stride arrays of equal size");
  }
  return TO_ERROR(cudnnSetPoolingNdDescriptor(
      descriptor, mode, nan_propagation, window_dimensions.size(),
      window_dimensions.data(), paddings.data(), strides.data()));
}

llvm::Expected<DnnPoolingDescriptorData> CudnnGetPoolingDescriptor(
    const cudnnPoolingDescriptor_t descriptor) {
  cudnnPoolingMode_t mode;
  cudnnNanPropagation_t nan_propagation;
  DnnPoolingDescriptorData data;
  int rank = 0;
  data.window_dimensions.resize(kDnnDimMax());
  data.paddings.resize(kDnnDimMax());
  data.strides.resize(kDnnDimMax());
  RETURN_IF_ERROR(cudnnGetPoolingNdDescriptor(
      descriptor, kDnnDimMax(), &mode, &nan_propagation, &rank,
      data.window_dimensions.data(), data.paddings.data(),
      data.strides.data()));
  data.mode = static_cast<DnnPoolingMode>(mode);
  data.nan_propagation = nan_propagation;
  data.window_dimensions.resize(rank);
  data.paddings.resize(rank);
  data.strides.resize(rank);
  return data;
}

llvm::Expected<llvm::SmallVector<int, kDnnDimMax()>>
CudnnGetPoolingForwardOutputDim(
    const cudnnPoolingDescriptor_t pooling_desc,
    const cudnnTensorDescriptor_t input_tensor_desc) {
  llvm::SmallVector<int, kDnnDimMax()> output_dim(kDnnDimMax());
  RETURN_IF_ERROR(cudnnGetPoolingNdForwardOutputDim(
      pooling_desc, input_tensor_desc, kDnnDimMax(), output_dim.data()));
  return output_dim;
}

llvm::Error CudnnPoolingForward(CurrentContext current, cudnnHandle_t handle,
                                const cudnnPoolingDescriptor_t pooling_desc,
                                Pointer<const void> alpha,
                                const cudnnTensorDescriptor_t x_desc,
                                Pointer<const void> x, Pointer<const void> beta,
                                const cudnnTensorDescriptor_t y_desc,
                                Pointer<void> y) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnPoolingForward(handle, pooling_desc, ToCuda(alpha),
                                      x_desc, ToCuda(x), ToCuda(beta), y_desc,
                                      ToCuda(y)));
}

llvm::Error CudnnPoolingBackward(
    CurrentContext current, cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t pooling_desc, Pointer<const void> alpha,
    const cudnnTensorDescriptor_t y_desc, Pointer<const void> y,
    const cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    const cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, const cudnnTensorDescriptor_t dx_desc,
    Pointer<void> dx) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnPoolingBackward(
      handle, pooling_desc, ToCuda(alpha), y_desc, ToCuda(y), dy_desc,
      ToCuda(dy), x_desc, ToCuda(x), ToCuda(beta), dx_desc, ToCuda(dx)));
}

llvm::Expected<OwningDnnActivationDescriptor>
CudnnCreateActivationDescriptor() {
  cudnnActivationDescriptor_t activation_desc;
  RETURN_IF_ERROR(cudnnCreateActivationDescriptor(&activation_desc));
  return OwningDnnActivationDescriptor(activation_desc);
}

llvm::Error CudnnDestroyActivationDescriptor(
    cudnnActivationDescriptor_t descriptor) {
  return TO_ERROR(cudnnDestroyActivationDescriptor(descriptor));
}

llvm::Error CudnnSetActivationDescriptor(cudnnActivationDescriptor_t descriptor,
                                         cudnnActivationMode_t mode,
                                         cudnnNanPropagation_t nan_propagation,
                                         double coefficient) {
  return TO_ERROR(cudnnSetActivationDescriptor(descriptor, mode,
                                               nan_propagation, coefficient));
}

llvm::Expected<DnnActivationDescriptorData> CudnnGetActivationDescriptor(
    const cudnnActivationDescriptor_t activation_desc) {
  cudnnActivationMode_t mode;
  cudnnNanPropagation_t nan_propagation;
  DnnActivationDescriptorData data;
  RETURN_IF_ERROR(cudnnGetActivationDescriptor(
      activation_desc, &mode, &nan_propagation, &data.coefficient));
  data.mode = static_cast<DnnActivationMode>(mode);
  data.nan_propagation = nan_propagation;
  return data;
}

llvm::Error CudnnActivationForward(CurrentContext current, cudnnHandle_t handle,
                                   cudnnActivationDescriptor_t activation_desc,
                                   Pointer<const void> alpha,
                                   const cudnnTensorDescriptor_t x_desc,
                                   Pointer<const void> x,
                                   Pointer<const void> beta,
                                   const cudnnTensorDescriptor_t y_desc,
                                   Pointer<void> y) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnActivationForward(handle, activation_desc, ToCuda(alpha),
                                         x_desc, ToCuda(x), ToCuda(beta),
                                         y_desc, ToCuda(y)));
}

llvm::Error CudnnActivationBackward(
    CurrentContext current, cudnnHandle_t handle,
    cudnnActivationDescriptor_t activation_desc, Pointer<const void> alpha,
    const cudnnTensorDescriptor_t y_desc, Pointer<const void> y,
    const cudnnTensorDescriptor_t dy_desc, Pointer<const void> dy,
    const cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    Pointer<const void> beta, const cudnnTensorDescriptor_t dx_desc,
    Pointer<void> dx) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnActivationBackward(
      handle, activation_desc, ToCuda(alpha), y_desc, ToCuda(y), dy_desc,
      ToCuda(dy), x_desc, ToCuda(x), ToCuda(beta), dx_desc, ToCuda(dx)));
}

llvm::Expected<size_t> CudnnGetBatchNormalizationForwardTrainingWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bn_ops,
    cudnnTensorDescriptor_t x_desc, cudnnTensorDescriptor_t z_desc,
    cudnnTensorDescriptor_t y_desc,
    cudnnTensorDescriptor_t bn_scale_bias_mean_var_desc,
    cudnnActivationDescriptor_t activation_desc) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      handle, mode, bn_ops, x_desc, z_desc, y_desc, bn_scale_bias_mean_var_desc,
      activation_desc, &size_in_bytes));
  return size_in_bytes;
}

llvm::Expected<size_t> CudnnGetBatchNormalizationBackwardWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bn_ops,
    cudnnTensorDescriptor_t x_desc, cudnnTensorDescriptor_t y_desc,
    cudnnTensorDescriptor_t dy_desc, cudnnTensorDescriptor_t dz_desc,
    cudnnTensorDescriptor_t dx_desc,
    cudnnTensorDescriptor_t d_bn_scale_bias_desc,
    cudnnActivationDescriptor_t activation_desc) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
      handle, mode, bn_ops, x_desc, y_desc, dy_desc, dz_desc, dx_desc,
      d_bn_scale_bias_desc, activation_desc, &size_in_bytes));
  return size_in_bytes;
}

llvm::Expected<size_t> CudnnGetBatchNormalizationTrainingReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bn_ops,
    cudnnActivationDescriptor_t activation_desc,
    cudnnTensorDescriptor_t x_desc) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      handle, mode, bn_ops, activation_desc, x_desc, &size_in_bytes));
  return size_in_bytes;
}

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
    size_t reserve_space_size_in_bytes) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnBatchNormalizationForwardTrainingEx(
      handle, mode, bn_ops, ToCuda(alpha), ToCuda(beta), x_desc, ToCuda(x_data),
      z_desc, ToCuda(z_data), y_desc, ToCuda(y_data),
      bn_scale_bias_mean_var_desc, ToCuda(bn_scale), ToCuda(bn_bias),
      exponential_average_factor, ToCuda(result_running_mean),
      ToCuda(result_running_variance), epsilon, ToCuda(result_save_mean),
      ToCuda(result_save_inv_variance), activation_desc, ToCuda(workspace),
      work_space_size_in_bytes, ToCuda(reserve_space),
      reserve_space_size_in_bytes));
}

llvm::Error CudnnBatchNormalizationForwardInference(
    CurrentContext current, cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    Pointer<const void> alpha, Pointer<const void> beta,
    cudnnTensorDescriptor_t x_desc, Pointer<const void> x,
    cudnnTensorDescriptor_t y_desc, Pointer<void> y,
    cudnnTensorDescriptor_t bn_scale_bias_mean_var_desc,
    Pointer<const void> bn_scale, Pointer<const void> bn_bias,
    Pointer<const void> estimated_mean, Pointer<const void> estimated_variance,
    double epsilon) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnBatchNormalizationForwardInference(
      handle, mode, ToCuda(alpha), ToCuda(beta), x_desc, ToCuda(x), y_desc,
      ToCuda(y), bn_scale_bias_mean_var_desc, ToCuda(bn_scale), ToCuda(bn_bias),
      ToCuda(estimated_mean), ToCuda(estimated_variance), epsilon));
}

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
    size_t reserve_space_size_in_bytes) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnBatchNormalizationBackwardEx(
      handle, mode, bn_ops, ToCuda(alpha_data_diff), ToCuda(beta_data_diff),
      ToCuda(alpha_param_diff), ToCuda(beta_param_diff), x_desc, ToCuda(x_data),
      y_desc, ToCuda(y_data), dy_desc, ToCuda(dy_data), dz_desc,
      ToCuda(dz_data), dx_desc, ToCuda(dx_data), d_bn_scale_bias_desc,
      ToCuda(bn_scale_data), ToCuda(bn_bias_data), ToCuda(d_bn_scale_data),
      ToCuda(d_bn_bias_data), epsilon, ToCuda(saved_mean),
      ToCuda(saved_inv_variance), activation_desc, ToCuda(work_space),
      work_space_size_in_bytes, ToCuda(reserve_space),
      reserve_space_size_in_bytes));
}

llvm::Expected<OwningDnnDropoutDescriptor> CudnnCreateDropoutDescriptor() {
  cudnnDropoutDescriptor_t descriptor = nullptr;
  RETURN_IF_ERROR(cudnnCreateDropoutDescriptor(&descriptor));
  return OwningDnnDropoutDescriptor(descriptor);
}

llvm::Error CudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t descriptor) {
  return TO_ERROR(cudnnDestroyDropoutDescriptor(descriptor));
}

llvm::Expected<size_t> CudnnDropoutGetStatesSize(cudnnHandle_t handle) {
  size_t size_bytes = 0;
  RETURN_IF_ERROR(cudnnDropoutGetStatesSize(handle, &size_bytes));
  return size_bytes;
}

llvm::Expected<size_t> CudnnDropoutGetReserveSpaceSize(
    cudnnTensorDescriptor_t xdesc) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(cudnnDropoutGetReserveSpaceSize(xdesc, &size_in_bytes));
  return size_in_bytes;
}

llvm::Error CudnnSetDropoutDescriptor(CurrentContext current,
                                      cudnnHandle_t handle,
                                      cudnnDropoutDescriptor_t descriptor,
                                      float dropout, Pointer<void> states,
                                      size_t states_size_bytes, uint64_t seed) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnSetDropoutDescriptor(descriptor, handle, dropout,
                                            states.raw(Platform::CUDA),
                                            states_size_bytes, seed));
}

llvm::Error CudnnRestoreDropoutDescriptor(CurrentContext current,
                                          cudnnHandle_t handle,
                                          cudnnDropoutDescriptor_t descriptor,
                                          float dropout, Pointer<void> states,
                                          size_t state_size_in_bytes,
                                          uint64_t seed) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnRestoreDropoutDescriptor(
      descriptor, handle, dropout, ToCuda(states), state_size_in_bytes, seed));
}

llvm::Expected<CudnnDropoutDescriptorData> CudnnGetDropoutDescriptor(
    cudnnHandle_t handle, cudnnDropoutDescriptor_t descriptor) {
  CudnnDropoutDescriptorData data;
  void* states;
  unsigned long long seed;  // NOLINT(google-runtime-int)
  RETURN_IF_ERROR(cudnnGetDropoutDescriptor(descriptor, handle, &data.dropout,
                                            &states, &seed));
  data.seed = seed;
  data.states = Pointer<void>(states, Platform::CUDA);
  return data;
}

llvm::Error CudnnDropoutForward(CurrentContext current, cudnnHandle_t handle,
                                cudnnDropoutDescriptor_t descriptor,
                                cudnnTensorDescriptor_t xdesc,
                                Pointer<const void> x,
                                cudnnTensorDescriptor_t ydesc, Pointer<void> y,
                                Pointer<void> reserve_space,
                                size_t reserve_space_size_in_bytes) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnDropoutForward(handle, descriptor, xdesc, ToCuda(x),
                                      ydesc, ToCuda(y), ToCuda(reserve_space),
                                      reserve_space_size_in_bytes));
}

llvm::Error CudnnDropoutBackward(CurrentContext current, cudnnHandle_t handle,
                                 cudnnDropoutDescriptor_t descriptor,
                                 cudnnTensorDescriptor_t dydesc,
                                 Pointer<const void> dy,
                                 cudnnTensorDescriptor_t dxdesc,
                                 Pointer<void> dx, Pointer<void> reserve_space,
                                 size_t reserve_space_size_in_bytes) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnDropoutBackward(
      handle, descriptor, dydesc, ToCuda(dy), dxdesc, ToCuda(dx),
      ToCuda(reserve_space), reserve_space_size_in_bytes));
}

llvm::Expected<OwningDnnRnnDescriptor> CudnnCreateRnnDescriptor() {
  cudnnRNNDescriptor_t descriptor = nullptr;
  RETURN_IF_ERROR(cudnnCreateRNNDescriptor(&descriptor));
  return OwningDnnRnnDescriptor(descriptor);
}

llvm::Error CudnnDestroyRnnDescriptor(cudnnRNNDescriptor_t descriptor) {
  return TO_ERROR(cudnnDestroyRNNDescriptor(descriptor));
}

llvm::Error CudnnSetRnnDescriptor(cudnnHandle_t handle,
                                  cudnnRNNDescriptor_t descriptor,
                                  int hidden_size, int num_layers,
                                  cudnnDropoutDescriptor_t dropout,
                                  cudnnRNNInputMode_t input_mode,
                                  cudnnDirectionMode_t direction,
                                  cudnnRNNMode_t mode, cudnnRNNAlgo_t algorithm,
                                  cudnnDataType_t math_precision) {
  return TO_ERROR(cudnnSetRNNDescriptor_v6(
      handle, descriptor, hidden_size, num_layers, dropout, input_mode,
      direction, mode, algorithm, math_precision));
}

llvm::Expected<DnnRnnDescriptorData> CudnnGetRnnDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor) {
  cudnnDropoutDescriptor_t dropout_descriptor;
  cudnnRNNInputMode_t input_mode;
  cudnnDirectionMode_t direction;
  cudnnRNNMode_t mode;
  cudnnRNNAlgo_t algorithm;
  cudnnDataType_t math_type;
  DnnRnnDescriptorData data;
#if CUDNN_VERSION < 8000
  const auto cudnnGetRNNDescriptor_v6 = cudnnGetRNNDescriptor;
#endif
  RETURN_IF_ERROR(cudnnGetRNNDescriptor_v6(
      handle, descriptor, &data.hidden_size, &data.num_layers,
      &dropout_descriptor, &input_mode, &direction, &mode, &algorithm,
      &math_type));
  data.dropout_desc = dropout_descriptor;
  data.input_mode = static_cast<DnnRnnInputMode>(input_mode);
  data.direction = static_cast<DnnDirectionMode>(direction);
  data.mode = static_cast<DnnRnnMode>(mode);
  data.algorithm = algorithm;
  data.math_type = math_type;
  return data;
}

llvm::Error CudnnSetRnnMatrixMathType(cudnnRNNDescriptor_t descriptor,
                                      cudnnMathType_t m_type) {
  return TO_ERROR(cudnnSetRNNMatrixMathType(descriptor, m_type));
}

llvm::Expected<cudnnMathType_t> CudnnGetRnnMatrixMathType(
    cudnnRNNDescriptor_t descriptor) {
  cudnnMathType_t m_type;
  RETURN_IF_ERROR(cudnnGetRNNMatrixMathType(descriptor, &m_type));
  return m_type;
}

llvm::Error CudnnSetRnnBiasMode(cudnnRNNDescriptor_t descriptor,
                                cudnnRNNBiasMode_t bias_mode) {
  return TO_ERROR(cudnnSetRNNBiasMode(descriptor, bias_mode));
}

llvm::Expected<cudnnRNNBiasMode_t> CudnnGetRnnBiasMode(
    cudnnRNNDescriptor_t descriptor) {
  cudnnRNNBiasMode_t bias_mode;
  RETURN_IF_ERROR(cudnnGetRNNBiasMode(descriptor, &bias_mode));
  return bias_mode;
}

llvm::Error CudnnRnnSetClip(cudnnHandle_t handle,
                            cudnnRNNDescriptor_t descriptor,
                            cudnnRNNClipMode_t clip_mode,
                            cudnnNanPropagation_t clip_nan_opt, double lclip,
                            double rclip) {
  return TO_ERROR(cudnnRNNSetClip(handle, descriptor, clip_mode, clip_nan_opt,
                                  lclip, rclip));
}

llvm::Expected<CudnnRnnClipData> CudnnRnnGetClip(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor) {
  CudnnRnnClipData data;
  RETURN_IF_ERROR(cudnnRNNGetClip(handle, descriptor, &data.mode,
                                  &data.nan_propagation, &data.left_clip,
                                  &data.right_clip));
  return data;
}

llvm::Expected<size_t> CudnnGetRnnParamsSize(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn_descriptor,
    cudnnTensorDescriptor_t tensor_descriptor, cudnnDataType_t data_type) {
  size_t size_bytes = 0;
  RETURN_IF_ERROR(cudnnGetRNNParamsSize(
      handle, rnn_descriptor, tensor_descriptor, &size_bytes, data_type));
  return size_bytes;
}

llvm::Expected<size_t> CudnnGetRnnWorkspaceSize(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn_descriptor,
    llvm::ArrayRef<cudnnTensorDescriptor_t> tensor_descriptors) {
  size_t size_bytes = 0;
  RETURN_IF_ERROR(cudnnGetRNNWorkspaceSize(
      handle, rnn_descriptor, tensor_descriptors.size(),
      tensor_descriptors.data(), &size_bytes));
  return size_bytes;
}

llvm::Expected<size_t> CudnnGetRnnTrainingReserveSize(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn_descriptor,
    llvm::ArrayRef<cudnnTensorDescriptor_t> tensor_descriptors) {
  size_t size_bytes = 0;
  RETURN_IF_ERROR(cudnnGetRNNTrainingReserveSize(
      handle, rnn_descriptor, tensor_descriptors.size(),
      tensor_descriptors.data(), &size_bytes));
  return size_bytes;
}

llvm::Expected<Pointer<void>> CudnnGetRnnLinLayerMatrixParams(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn_descriptor, int pseudo_layer,
    cudnnTensorDescriptor_t tensor_descriptor,
    cudnnFilterDescriptor_t filter_descriptor, Pointer<const void> weights,
    int layer_index, cudnnFilterDescriptor_t matrix_descriptor) {
  void* matrix_ptr = nullptr;
  RETURN_IF_ERROR(cudnnGetRNNLinLayerMatrixParams(
      handle, rnn_descriptor, pseudo_layer, tensor_descriptor,
      filter_descriptor, ToCuda(weights), layer_index, matrix_descriptor,
      &matrix_ptr));
  return Pointer<void>(matrix_ptr, Platform::CUDA);
}

llvm::Expected<Pointer<void>> CudnnGetRnnLinLayerBiasParams(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn_descriptor, int pseudo_layer,
    cudnnTensorDescriptor_t tensor_descriptor,
    cudnnFilterDescriptor_t filter_descriptor, Pointer<const void> weights,
    int layer_index, cudnnFilterDescriptor_t bias_descriptor) {
  void* bias_ptr = nullptr;
  RETURN_IF_ERROR(cudnnGetRNNLinLayerBiasParams(
      handle, rnn_descriptor, pseudo_layer, tensor_descriptor,
      filter_descriptor, ToCuda(weights), layer_index, bias_descriptor,
      &bias_ptr));
  return Pointer<void>(bias_ptr, Platform::CUDA);
}

llvm::Expected<OwningCudnnPersistentRnnPlan> CudnnCreatePersistentRnnPlan(
    cudnnRNNDescriptor_t descriptor, int batch_size,
    cudnnDataType_t data_type) {
  cudnnPersistentRNNPlan_t plan = nullptr;
  RETURN_IF_ERROR(
      cudnnCreatePersistentRNNPlan(descriptor, batch_size, data_type, &plan));
  return OwningCudnnPersistentRnnPlan(plan);
}

llvm::Error CudnnDestroyPersistentRnnPlan(cudnnPersistentRNNPlan_t plan) {
  return TO_ERROR(cudnnDestroyPersistentRNNPlan(plan));
}

llvm::Error CudnnSetPersistentRnnPlan(cudnnRNNDescriptor_t descriptor,
                                      cudnnPersistentRNNPlan_t plan) {
  return TO_ERROR(cudnnSetPersistentRNNPlan(descriptor, plan));
}

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
    size_t workspace_size_bytes) {
  CheckCudaContext(current);
  if (input_descriptors.size() != output_descriptors.size()) {
    return MakeStringError(
        "Expected input and output descriptor arrays of equal size");
  }
  return TO_ERROR(cudnnRNNForwardInference(
      handle, rnn_descriptor, input_descriptors.size(),
      input_descriptors.data(), ToCuda(input_data), hidden_input_descriptor,
      ToCuda(hidden_input_data), cell_input_descriptor, ToCuda(cell_input_data),
      filter_descriptor, ToCuda(filter_data), output_descriptors.data(),
      ToCuda(output_data), hidden_input_descriptor, ToCuda(hidden_output_data),
      cell_output_descriptor, ToCuda(cell_output_data), ToCuda(workspace),
      workspace_size_bytes));
}

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
    size_t reserve_space_size_in_bytes) {
  CheckCudaContext(current);
  if (input_descriptors.size() != output_descriptors.size()) {
    return MakeStringError(
        "Expected input and output descriptor arrays of equal size");
  }
  return TO_ERROR(cudnnRNNForwardTraining(
      handle, rnn_descriptor, input_descriptors.size(),
      input_descriptors.data(), ToCuda(input_data), hidden_input_descriptor,
      ToCuda(hidden_input_data), cell_input_descriptor, ToCuda(cell_input_data),
      filter_descriptor, ToCuda(filter_data), output_descriptors.data(),
      ToCuda(output_data), hidden_input_descriptor, ToCuda(hidden_output_data),
      cell_output_descriptor, ToCuda(cell_output_data), ToCuda(workspace),
      workspace_size_bytes, ToCuda(reserve_space),
      reserve_space_size_in_bytes));
}

llvm::Error CudnnBackendExecute(CurrentContext current, cudnnHandle_t handle,
                                cudnnBackendDescriptor_t execution_plan,
                                cudnnBackendDescriptor_t variant_pack) {
  CheckCudaContext(current);
  return TO_ERROR(cudnnBackendExecute(handle, execution_plan, variant_pack));
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
