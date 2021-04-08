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

namespace tfrt {
namespace btf {

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
