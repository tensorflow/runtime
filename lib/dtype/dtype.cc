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

// This file implements DType class.

#include "tfrt/dtype/dtype.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace tfrt {

const char *DType::GetName() const {
  switch (kind_) {
    case Invalid:
      return "Invalid";
    case Unsupported:
    case Resource:
    case Variant:
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
      // TODO(b/170482990): unify I1 and BOOL, use tfrt::i1 directly.
      return "bool";
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
    case Complex64:
      return "complex64";
    case Complex128:
      return "complex128";
    case F16:
      return "f16";
    case BF16:
      return "bf16";
    case String:
      return "str";
    case QUI8:
      return "quint8";
    case QUI16:
      return "quint16";
    case QI8:
      return "qint8";
    case QI16:
      return "qint16";
    case QI32:
      return "qint32";
  }
}

// Return the size of one value of this dtype when represented on the host.
size_t DType::GetHostSize() const {
  switch (kind_) {
    case Invalid:
      assert(0 && "invalid dtype has no size");
      return ~size_t(0);
    case Unsupported:
    case Resource:
      assert(0 && "unsupported dtype has no size");
      return ~size_t(0);
    case Variant:
      // Using size of HostBuffer.
      return size_t(64);
    case String:
      assert(0 &&
             "GetHostSize() is not expected to be called with string type");
      return ~size_t(0);
    case BF16:
      return sizeof(TypeForDTypeKind<DType::BF16>);
    case I1:
      return sizeof(TypeForDTypeKind<DType::I1>);
    case Complex64:
      return sizeof(TypeForDTypeKind<DType::Complex64>);
    case Complex128:
      return sizeof(TypeForDTypeKind<DType::Complex128>);
    case QUI8:
      return sizeof(TypeForDTypeKind<DType::QUI8>);
    case QUI16:
      return sizeof(TypeForDTypeKind<DType::QUI16>);
    case QI8:
      return sizeof(TypeForDTypeKind<DType::QI8>);
    case QI16:
      return sizeof(TypeForDTypeKind<DType::QI16>);
    case QI32:
      return sizeof(TypeForDTypeKind<DType::QI32>);
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
    case Resource:
      assert(0 && "unspupported dtype has no alignment");
      return ~size_t(0);
    case Variant:
      return alignof(std::max_align_t);
    case String:
      assert(
          0 &&
          "GetHostAlignment() is not expected to be called with string type");
      return ~size_t(0);
    case BF16:
      return alignof(TypeForDTypeKind<DType::BF16>);
    case I1:
      return alignof(TypeForDTypeKind<DType::I1>);
    case Complex64:
      return alignof(TypeForDTypeKind<DType::Complex64>);
    case Complex128:
      return alignof(TypeForDTypeKind<DType::Complex128>);
    case QUI8:
      return alignof(TypeForDTypeKind<DType::QUI8>);
    case QUI16:
      return alignof(TypeForDTypeKind<DType::QUI16>);
    case QI8:
      return alignof(TypeForDTypeKind<DType::QI8>);
    case QI16:
      return alignof(TypeForDTypeKind<DType::QI16>);
    case QI32:
      return alignof(TypeForDTypeKind<DType::QI32>);
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
    case DType::Resource:
    case DType::Variant:
      llvm_unreachable("can't happen");
    case DType::BF16:
      os << "Does not support printing bf16.";
      break;
    case DType::F16:
      os << "Does not support printing fp16.";
      break;
    case DType::QUI8:
      os << *static_cast<const TypeForDTypeKind<DType::QUI8> *>(data);
      break;
    case DType::QI8:
      os << *static_cast<const TypeForDTypeKind<DType::QI8> *>(data);
      break;
    case DType::QI16:
      os << *static_cast<const TypeForDTypeKind<DType::QI16> *>(data);
      break;
    case DType::QI32:
      os << *static_cast<const TypeForDTypeKind<DType::QI32> *>(data);
      break;
    case DType::QUI16:
      os << *static_cast<const TypeForDTypeKind<DType::QUI16> *>(data);
      break;
    case DType::String:
      os << *static_cast<const TypeForDTypeKind<DType::String> *>(data);
      break;
    case DType::Complex64:
      os << "("
         << static_cast<const TypeForDTypeKind<DType::Complex64> *>(data)
                ->real()
         << ","
         << static_cast<const TypeForDTypeKind<DType::Complex64> *>(data)
                ->imag()
         << ")";
      break;
    case DType::Complex128:
      os << "("
         << static_cast<const TypeForDTypeKind<DType::Complex128> *>(data)
                ->real()
         << ","
         << static_cast<const TypeForDTypeKind<DType::Complex128> *>(data)
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
    case DType::Resource:
    case DType::Variant:
      llvm_unreachable("can't happen");
    case DType::BF16:
      os << "Does not support printing bf16.";
      break;
    case DType::F16:
      os << "Does not support printing fp16.";
      break;
    case DType::I1:
      os << *static_cast<const TypeForDTypeKind<DType::I1> *>(data);
      break;
    case DType::QUI8:
      os << *static_cast<const TypeForDTypeKind<DType::QUI8> *>(data);
      break;
    case DType::QI8:
      os << *static_cast<const TypeForDTypeKind<DType::QI8> *>(data);
      break;
    case DType::QI16:
      os << *static_cast<const TypeForDTypeKind<DType::QI16> *>(data);
      break;
    case DType::QI32:
      os << *static_cast<const TypeForDTypeKind<DType::QI32> *>(data);
      break;
    case DType::QUI16:
      os << *static_cast<const TypeForDTypeKind<DType::QUI16> *>(data);
      break;
    case DType::String:
      os << *static_cast<const TypeForDTypeKind<DType::String> *>(data);
      break;
    case DType::Complex64:
      os << "("
         << static_cast<const TypeForDTypeKind<DType::Complex64> *>(data)
                ->real()
         << ","
         << static_cast<const TypeForDTypeKind<DType::Complex64> *>(data)
                ->imag()
         << ")";
      break;
    case DType::Complex128:
      os << "("
         << static_cast<const TypeForDTypeKind<DType::Complex128> *>(data)
                ->real()
         << ","
         << static_cast<const TypeForDTypeKind<DType::Complex128> *>(data)
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
#define DTYPE(ENUM, _) \
  case DType::ENUM:    \
    os << #ENUM;       \
    break;
#include "tfrt/dtype/dtype.def"
  }
  return os;
}

}  // namespace tfrt
