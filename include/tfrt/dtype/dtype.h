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

//===- dtype.h --------------------------------------------------*- C++ -*-===//
//
// This file defines the DType enum and helpers for working with it.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DTYPE_DTYPE_H_
#define TFRT_DTYPE_DTYPE_H_

#include <complex>
#include <cstddef>
#include <cstdint>

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/dtype/i1.h"
#include "tfrt/support/bf16.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/fp16.h"

namespace tfrt {

class DType {
 public:
  enum Kind {
    // Invalid type that is used by default constructor. A Tensor should never
    // use this type.
    Invalid,
    // Valid types that are not natively supported by TFRT.
    Unsupported,
#define DTYPE(ENUM) ENUM,
#include "tfrt/dtype/dtype.def"
  };

  explicit DType() : kind_(Invalid) {}

  explicit constexpr DType(Kind kind) : kind_(kind) {}
  DType(const DType &) = default;
  DType &operator=(const DType &) = default;
  bool operator==(DType other) const { return kind_ == other.kind_; }
  bool operator!=(DType other) const { return kind_ != other.kind_; }

  Kind kind() const { return kind_; }

  bool IsValid() const { return kind_ != Invalid; }
  bool IsInvalid() const { return kind_ == Invalid; }

  // Get the name for the dtype, e.g. i32, f32.
  const char *GetName() const;

  // Print out a blob of memory as this dtype.
  void Print(const void *data, raw_ostream &os) const;

  // Print out a blob of memory as this dtype with full precision.
  // Full precision is defined by std::numeric_limits<T>::max_digits10.
  // These many digits are enough to make sure number->text->number is
  // guaranteed to get the same number back.
  void PrintFullPrecision(const void *data, raw_ostream &os) const;

  // Return the size of one value of this dtype when represented on the host.
  size_t GetHostSize() const;

  // Return the alignment of this dtype when represented on the host.
  size_t GetHostAlignment() const;

 private:
  Kind kind_;
};

// Support printing of dtype enums.
raw_ostream &operator<<(raw_ostream &os, DType dtype);

// Provides interconversions between C++ type and DTypes at compile time.
//
// GetDType<T>() is the DType for the C++ type T.
//
// TypeForDTypeKind<DT> is the C++ type for DType kind DT. For non-trivial type,
// it can only return a storage-only type.

// Provide a way to get the DType for a specified C++ type at compile time.
template <typename T>
constexpr DType GetDType();

// Provide a way to get the C++ type for a specified DType Kind at compile
// time.
template <DType::Kind K>
struct TypeForDTypeKindInternal {};
template <DType::Kind K>
using TypeForDTypeKind = typename TypeForDTypeKindInternal<K>::Type;

// TFRT_REGISTER_DTYPE is a macro to register a non-trivial C++ type to a DType.
#define TFRT_REGISTER_DTYPE(CPP_TYPE, ENUM)     \
  template <>                                   \
  inline constexpr DType GetDType<CPP_TYPE>() { \
    return DType{DType::ENUM};                  \
  }

#define TFRT_DEFINE_DTYPE_INTERNAL(ENUM, CPP_TYPE) \
  TFRT_REGISTER_DTYPE(CPP_TYPE, ENUM)              \
  template <>                                      \
  struct TypeForDTypeKindInternal<DType::ENUM> {   \
    using Type = CPP_TYPE;                         \
  };

// LINT.IfChange
TFRT_DEFINE_DTYPE_INTERNAL(UI8, uint8_t)
TFRT_DEFINE_DTYPE_INTERNAL(UI16, uint16_t)
TFRT_DEFINE_DTYPE_INTERNAL(UI32, uint32_t)
TFRT_DEFINE_DTYPE_INTERNAL(UI64, uint64_t)
TFRT_DEFINE_DTYPE_INTERNAL(I1, i1)
TFRT_DEFINE_DTYPE_INTERNAL(I8, int8_t)
TFRT_DEFINE_DTYPE_INTERNAL(I16, int16_t)
TFRT_DEFINE_DTYPE_INTERNAL(I32, int32_t)
TFRT_DEFINE_DTYPE_INTERNAL(I64, int64_t)
TFRT_DEFINE_DTYPE_INTERNAL(BF16, bf16)
TFRT_DEFINE_DTYPE_INTERNAL(F16, fp16)
TFRT_DEFINE_DTYPE_INTERNAL(F32, float)
TFRT_DEFINE_DTYPE_INTERNAL(F64, double)
TFRT_DEFINE_DTYPE_INTERNAL(BOOL, bool)
// TODO(tfrt-devs): Consider creating a special CPP string type for TFRT.
TFRT_DEFINE_DTYPE_INTERNAL(String, std::string)
TFRT_DEFINE_DTYPE_INTERNAL(COMPLEX64,
                           std::complex<float>)  // Single precision complex.
TFRT_DEFINE_DTYPE_INTERNAL(COMPLEX128,
                           std::complex<double>)  // Double precision complex.
// LINT.ThenChange(//depot/tf_runtime/include/tfrt/dtype/dtype.def)

#undef TFRT_DEFINE_DTYPE_INTERNAL

}  // namespace tfrt

#endif  // TFRT_DTYPE_DTYPE_H_
