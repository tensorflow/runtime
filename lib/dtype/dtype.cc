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

//===- dtype.cc -----------------------------------------------------------===//
//
// This file implements DType class.
//
//===----------------------------------------------------------------------===//

#include "tfrt/dtype/dtype.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace tfrt {

const char *DType::GetName() const {
  switch (kind_) {
    case Invalid:
      return "invalid";
    case Unsupported:
      return "unsupported";
    case UI8:
      return "u8";
    case UI16:
      return "u16";
    case UI32:
      return "u32";
    case UI64:
      return "u64";
    case I1:
      return "u1";
    case I8:
      return "i8";
    case I16:
      return "i16";
    case I32:
      return "i32";
    case I64:
      return "i64";
    case F32:
      return "f32";
    case F64:
      return "f64";
    case BOOL:
      return "bool";
    case COMPLEX64:
      return "complex64";
    case COMPLEX128:
      return "complex128";
    case F16:
      return "f16";
    case BF16:
      return "bf16";
    case String:
      return "str";
  }
}

// Return the size of one value of this dtype when represented on the host.
size_t DType::GetHostSize() const {
  switch (kind_) {
    case Invalid:
      assert(0 && "invalid dtype has no size");
      return ~size_t(0);
    case Unsupported:
      assert(0 && "unsupported dtype has no size");
      return ~size_t(0);
    case String:
      assert(0 &&
             "GetHostSize() is not expected to be called with string type");
      return ~size_t(0);
    case BF16:
      return sizeof(TypeForDTypeKind<DType::BF16>);
    case I1:
      return sizeof(TypeForDTypeKind<DType::I1>);
    case BOOL:
      return sizeof(TypeForDTypeKind<DType::BOOL>);
    case COMPLEX64:
      return sizeof(TypeForDTypeKind<DType::COMPLEX64>);
    case COMPLEX128:
      return sizeof(TypeForDTypeKind<DType::COMPLEX128>);
#define DTYPE_NUMERIC(ENUM) \
  case ENUM:                \
    return sizeof(TypeForDTypeKind<DType::ENUM>);
#include "tfrt/dtype/dtype.def"
  }
}

// Return the alignment of this dtype when represented on the host.
size_t DType::GetHostAlignment() const {
  switch (kind_) {
    case Invalid:
      assert(0 && "invalid dtype has no alignment");
      return ~size_t(0);
    case Unsupported:
      assert(0 && "unspupported dtype has no alignment");
      return ~size_t(0);
    case String:
      assert(
          0 &&
          "GetHostAlignment() is not expected to be called with string type");
      return ~size_t(0);
    case BF16:
      return alignof(TypeForDTypeKind<DType::BF16>);
    case I1:
      return alignof(TypeForDTypeKind<DType::I1>);
    case BOOL:
      return alignof(TypeForDTypeKind<DType::BOOL>);
    case COMPLEX64:
      return alignof(TypeForDTypeKind<DType::COMPLEX64>);
    case COMPLEX128:
      return alignof(TypeForDTypeKind<DType::COMPLEX128>);
#define DTYPE_NUMERIC(ENUM) \
  case ENUM:                \
    return alignof(TypeForDTypeKind<DType::ENUM>);
#include "tfrt/dtype/dtype.def"  // NOLINT
  }
}

// Print out a blob of memory as this dtype.
void DType::Print(const void *data, raw_ostream &os) const {
  switch (kind()) {
    case DType::Invalid:
    case DType::Unsupported:
      llvm_unreachable("can't happen");
    case DType::BF16:
      os << "Does not support printing bf16.";
      break;
    case DType::F16:
      os << "Does not support printing fp16.";
      break;
    case DType::I1:
      os << *static_cast<const i1 *>(data);
      break;
    case DType::String:
      os << *static_cast<const TypeForDTypeKind<DType::String> *>(data);
      break;
    case DType::COMPLEX64:
      os << "("
         << static_cast<const TypeForDTypeKind<DType::COMPLEX64> *>(data)
                ->real()
         << ","
         << static_cast<const TypeForDTypeKind<DType::COMPLEX64> *>(data)
                ->imag()
         << ")";
      break;
    case DType::COMPLEX128:
      os << "("
         << static_cast<const TypeForDTypeKind<DType::COMPLEX128> *>(data)
                ->real()
         << ","
         << static_cast<const TypeForDTypeKind<DType::COMPLEX128> *>(data)
                ->imag()
         << ")";
      break;
#define DTYPE_TRIVIAL(ENUM)                                           \
  case ENUM:                                                          \
    os << +*static_cast<const TypeForDTypeKind<DType::ENUM> *>(data); \
    break;
#include "tfrt/dtype/dtype.def"  // NOLINT
  }
}

void DType::PrintFullPrecision(const void *data, raw_ostream &os) const {
  switch (kind()) {
    case DType::Invalid:
    case DType::Unsupported:
      llvm_unreachable("can't happen");
    case DType::BF16:
      os << "Does not support printing bf16.";
      break;
    case DType::F16:
      os << "Does not support printing fp16.";
      break;
    case DType::I1:
      os << *static_cast<const i1 *>(data);
      break;
    case DType::String:
      os << *static_cast<const TypeForDTypeKind<DType::String> *>(data);
      break;
    case DType::BOOL:
      os << *static_cast<const TypeForDTypeKind<DType::BOOL> *>(data);
      break;
    case DType::COMPLEX64:
      os << "("
         << static_cast<const TypeForDTypeKind<DType::COMPLEX64> *>(data)
                ->real()
         << ","
         << static_cast<const TypeForDTypeKind<DType::COMPLEX64> *>(data)
                ->imag()
         << ")";
      break;
    case DType::COMPLEX128:
      os << "("
         << static_cast<const TypeForDTypeKind<DType::COMPLEX128> *>(data)
                ->real()
         << ","
         << static_cast<const TypeForDTypeKind<DType::COMPLEX128> *>(data)
                ->imag()
         << ")";
      break;
#define DTYPE_FLOAT(ENUM)                                                 \
  case ENUM:                                                              \
    os << llvm::format(                                                   \
        "%.*g",                                                           \
        std::numeric_limits<TypeForDTypeKind<DType::ENUM>>::max_digits10, \
        *static_cast<const TypeForDTypeKind<DType::ENUM> *>(data));       \
    break;
#define DTYPE_INT(ENUM)                                               \
  case ENUM:                                                          \
    os << +*static_cast<const TypeForDTypeKind<DType::ENUM> *>(data); \
    break;
#include "tfrt/dtype/dtype.def"  // NOLINT
  }
}

// Support printing of dtype enums.
raw_ostream &operator<<(raw_ostream &os, DType dtype) {
  switch (dtype.kind()) {
    case DType::Invalid:
      os << "<invalid dtype>";
      break;
    case DType::Unsupported:
      os << "<unsupported dtype>";
      break;
#define DTYPE(ENUM) \
  case DType::ENUM: \
    os << #ENUM;    \
    break;
#include "tfrt/dtype/dtype.def"
  }
  return os;
}

}  // namespace tfrt
