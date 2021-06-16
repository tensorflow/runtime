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

// This file contains functions related to BTF (Binary Tensor Format)

#include "tfrt/tensor/btf.h"

#include "llvm/Support/raw_ostream.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace btf {

DType::Kind ToDTypeKind(TensorDType type) {
  switch (type) {
    case TensorDType::kInt8:
      return DType::I8;
    case TensorDType::kInt16:
      return DType::I16;
    case TensorDType::kInt32:
      return DType::I32;
    case TensorDType::kInt64:
      return DType::I64;
    case TensorDType::kFloat32:
      return DType::F32;
    case TensorDType::kFloat64:
      return DType::F64;
    case TensorDType::kUInt8:
      return DType::UI8;
    case TensorDType::kUInt16:
      return DType::UI16;
    case TensorDType::kUInt32:
      return DType::UI32;
    case TensorDType::kUInt64:
      return DType::UI64;
  }
}

Expected<TensorDType> ToTensorDType(DType::Kind type) {
  switch (type) {
    case DType::I8:
      return TensorDType::kInt8;
    case DType::I16:
      return TensorDType::kInt16;
    case DType::I32:
      return TensorDType::kInt32;
    case DType::I64:
      return TensorDType::kInt64;
    case DType::F32:
      return TensorDType::kFloat32;
    case DType::F64:
      return TensorDType::kFloat64;
    case DType::UI8:
      return TensorDType::kUInt8;
    case DType::UI16:
      return TensorDType::kUInt16;
    case DType::UI32:
      return TensorDType::kUInt32;
    case DType::UI64:
      return TensorDType::kUInt64;
    default:
      return MakeStringError("failed to cast DType to TensorDType");
  }
}

raw_ostream& operator<<(raw_ostream& os, const TensorDType& dtype) {
  switch (dtype) {
    case TensorDType::kInt8:
      return os << "i8";
    case TensorDType::kInt16:
      return os << "i16";
    case TensorDType::kInt32:
      return os << "i32";
    case TensorDType::kInt64:
      return os << "i64";
    case TensorDType::kFloat32:
      return os << "f32";
    case TensorDType::kFloat64:
      return os << "f64";
    case TensorDType::kUInt8:
      return os << "ui8";
    case TensorDType::kUInt16:
      return os << "ui16";
    case TensorDType::kUInt32:
      return os << "ui32";
    case TensorDType::kUInt64:
      return os << "ui64";
  }
  return os << "BadDtype(" << static_cast<uint64_t>(dtype) << ')';
}

raw_ostream& operator<<(raw_ostream& os, const TensorLayout& layout) {
  switch (layout) {
    case TensorLayout::kRMD:
      return os << "Row-Major Dense tensor";
    case TensorLayout::kCOO_EXPERIMENTAL:
      return os << "COOrdinate list sparse tensor";
  }
  return os << "Unknown";
}

}  // namespace btf
}  // namespace tfrt
