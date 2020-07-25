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
#include <memory>

#include "tfrt/gpu/stream/stream_wrapper.h"

namespace tfrt {
namespace gpu {
namespace stream {

enum class DnnDataType {
  FLOAT = 0,
  DOUBLE = 1,
  HALF = 2,
  INT8 = 3,
  INT32 = 4,
  INT8x4 = 5,
  UINT8 = 6,
  UINT8x4 = 7,
  INT8x32 = 8,
};

enum class DnnTensorFormat {
  NCHW = 0,
  NHWC = 1,
  NCHW_VECT_C = 2,
};

enum class DnnRnnInputMode {
  LINEAR = 0,
  SKIP = 1,
};

enum class DnnDirectionMode {
  UNIDIRECTIONAL = 0,
  BIDIRECTIONAL = 1,
};

enum class DnnRnnMode {
  RNN_RELU = 0,
  RNN_TANH = 1,
  LSTM = 2,
  GRU = 3,
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
    Resource<cudnnFilterDescriptor_t, miopenFilterDescriptor_t>;
using DnnDropoutDescriptor =
    Resource<cudnnDropoutDescriptor_t, miopenDropoutDescriptor_t>;
using DnnRnnDescriptor = Resource<cudnnRNNDescriptor_t, miopenRNNDescriptor_t>;
using DnnPersistentRnnPlan =
    Resource<cudnnPersistentRNNPlan_t, miopenPersistentRNNPlan_t>;

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
struct DnnPersistentRnnPlanDeleter {
  using pointer = DnnPersistentRnnPlan;
  void operator()(DnnPersistentRnnPlan plan) const;
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
using OwningPersistentRnnPlan =
    internal::OwningResource<internal::DnnPersistentRnnPlanDeleter>;

static constexpr int kDnnDimMax = 8;

// Return types for functions returning multiple values.
struct DnnTensorDescriptorData {
  DnnDataType data_type;
  llvm::SmallVector<int, kDnnDimMax> dimensions;
  llvm::SmallVector<int, kDnnDimMax> strides;
};
struct DnnFilterDescriptorData {
  DnnDataType data_type;
  DnnTensorFormat format;
  llvm::SmallVector<int, kDnnDimMax> dimensions;
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

llvm::Expected<OwningDnnHandle> DnnCreate(CurrentContext current);
llvm::Error DnnDestroy(DnnHandle handle);
llvm::Error DnnSetStream(DnnHandle handle, Stream stream);
llvm::Expected<Stream> DnnGetStream(DnnHandle handle);

llvm::Expected<OwningDnnTensorDescriptor> DnnCreateTensorDescriptor(
    Platform platform);
llvm::Error DnnDestroyTensorDescriptor(DnnTensorDescriptor descriptor);

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

llvm::Expected<OwningDnnFilterDescriptor> DnnCreateFilterDescriptor(
    Platform platform);
llvm::Error DnnDestroyFilterDescriptor(DnnFilterDescriptor descriptor);

llvm::Expected<OwningDnnDropoutDescriptor> DnnCreateDropoutDescriptor(
    Platform platform);
llvm::Error DnnDestroyDropoutDescriptor(DnnDropoutDescriptor descriptor);

llvm::Expected<OwningDnnRnnDescriptor> DnnCreateRnnDescriptor(
    Platform platform);
llvm::Error DnnDestroyRnnDescriptor(DnnRnnDescriptor descriptor);

llvm::Expected<OwningPersistentRnnPlan> DnnCreatePersistentRnnPlan(
    DnnRnnDescriptor descriptor, int batch_size, DnnDataType data_type);
llvm::Error DnnDestroyPersistentRnnPlan(DnnPersistentRnnPlan plan);

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_STREAM_DNN_WRAPPER_H_
