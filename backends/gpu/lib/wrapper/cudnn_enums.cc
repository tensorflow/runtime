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

// cuDNN enum parsers and printers.
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"
#include "tfrt/support/fp16.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return os << "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return os << "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return os << "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return os << "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return os << "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return os << "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return os << "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return os << "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return os << "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return os << "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return os << "CUDNN_STATUS_LICENSE_ERROR";
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
      return os << "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
      return os << "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
      return os << "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
    default:
      return os << llvm::formatv("cudnnStatus_t({0})",
                                 static_cast<int>(status));
  }
}

template <>
Expected<cudnnDataType_t> Parse<cudnnDataType_t>(llvm::StringRef name) {
  if (name == "CUDNN_DATA_FLOAT") return CUDNN_DATA_FLOAT;
  if (name == "CUDNN_DATA_DOUBLE") return CUDNN_DATA_DOUBLE;
  if (name == "CUDNN_DATA_HALF") return CUDNN_DATA_HALF;
  if (name == "CUDNN_DATA_INT8") return CUDNN_DATA_INT8;
  if (name == "CUDNN_DATA_INT32") return CUDNN_DATA_INT32;
  if (name == "CUDNN_DATA_INT8x4") return CUDNN_DATA_INT8x4;
  if (name == "CUDNN_DATA_UINT8") return CUDNN_DATA_UINT8;
  if (name == "CUDNN_DATA_UINT8x4") return CUDNN_DATA_UINT8x4;
  if (name == "CUDNN_DATA_INT8x32") return CUDNN_DATA_INT8x32;
  return MakeStringError("Unknown cudnnDataType_t: ", name);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, cudnnDataType_t value) {
  switch (value) {
    case CUDNN_DATA_FLOAT:
      return os << "CUDNN_DATA_FLOAT";
    case CUDNN_DATA_DOUBLE:
      return os << "CUDNN_DATA_DOUBLE";
    case CUDNN_DATA_HALF:
      return os << "CUDNN_DATA_HALF";
    case CUDNN_DATA_INT8:
      return os << "CUDNN_DATA_INT8";
    case CUDNN_DATA_UINT8:
      return os << "CUDNN_DATA_UINT8";
    case CUDNN_DATA_INT32:
      return os << "CUDNN_DATA_INT32";
    case CUDNN_DATA_INT8x4:
      return os << "CUDNN_DATA_INT8x4";
    case CUDNN_DATA_INT8x32:
      return os << "CUDNN_DATA_INT8x32";
    case CUDNN_DATA_UINT8x4:
      return os << "CUDNN_DATA_UINT8x4";
    default:
      return os << llvm::formatv("cudnnDataType_t({0})",
                                 static_cast<int>(value));
  }
}

template <>
Expected<cudnnConvolutionMode_t> Parse<cudnnConvolutionMode_t>(
    llvm::StringRef name) {
  if (name == "CUDNN_CONVOLUTION") return CUDNN_CONVOLUTION;
  if (name == "CUDNN_CROSS_CORRELATION") return CUDNN_CROSS_CORRELATION;
  return MakeStringError("Unknown cudnnConvolutionMode_t: ", name);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              cudnnConvolutionMode_t value) {
  switch (value) {
    case CUDNN_CONVOLUTION:
      return os << "CUDNN_CONVOLUTION";
    case CUDNN_CROSS_CORRELATION:
      return os << "CUDNN_CROSS_CORRELATION";
    default:
      return os << llvm::formatv("cudnnConvolutionMode_t({0})",
                                 static_cast<int>(value));
  }
}

mlir::TypeID GetCudnnDataTypeId(cudnnDataType_t data_type) {
  switch (data_type) {
    case CUDNN_DATA_FLOAT:
      return mlir::TypeID::get<float>();
    case CUDNN_DATA_DOUBLE:
      return mlir::TypeID::get<double>();
    case CUDNN_DATA_HALF:
      return mlir::TypeID::get<fp16>();
    case CUDNN_DATA_INT8:
      return mlir::TypeID::get<int8_t>();
    case CUDNN_DATA_INT32:
      return mlir::TypeID::get<int32_t>();
    case CUDNN_DATA_UINT8:
      return mlir::TypeID::get<uint8_t>();
    case CUDNN_DATA_INT64:
      return mlir::TypeID::get<int64_t>();
    default:
      return {};
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
