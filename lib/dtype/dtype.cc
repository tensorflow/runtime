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
  return DispatchByDType(*this,
                         [](auto dtype_data) { return dtype_data.kName; });
}

// Return the size of one value of this dtype when represented on the host.
size_t DType::GetHostSize() const {
  return DispatchByDType(*this,
                         [](auto dtype_data) { return dtype_data.kByteSize; });
}

// Return the alignment of this dtype when represented on the host.
size_t DType::GetHostAlignment() const {
  return DispatchByDType(*this,
                         [](auto dtype_data) { return dtype_data.kAlignment; });
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
  return os << DispatchByDType(
             dtype, [](auto dtype_data) { return dtype_data.kEnumName; });
}

}  // namespace tfrt
