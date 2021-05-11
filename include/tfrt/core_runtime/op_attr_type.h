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

// This file declares OpAttrType and helpers.

#ifndef TFRT_CORE_RUNTIME_OP_ATTR_TYPE_H_
#define TFRT_CORE_RUNTIME_OP_ATTR_TYPE_H_

#include <cinttypes>
#include <complex>
#include <utility>

#include "tfrt/bef/bef_encoding.h"
#include "tfrt/dtype/quantized_types.h"
#include "tfrt/support/bf16.h"
#include "tfrt/support/fp16.h"

namespace tfrt {

class DenseAttr;
class AggregateAttr;
class ShapeAttr;

enum class OpAttrType : uint8_t {
  DTYPE,
  AGGREGATE,
  DENSE,
  SHAPE,
  // Function attribute.
  FUNC,
  BF16,
  F16,
  I1,
  COMPLEX64,
  COMPLEX128,
  // Following attribute types can only be used as value of DTYPE.
  UNSUPPORTED_QUI8,
  UNSUPPORTED_QUI16,
  UNSUPPORTED_QI8,
  UNSUPPORTED_QI16,
  UNSUPPORTED_QI32,
  UNSUPPORTED_RESOURCE,
  UNSUPPORTED_VARIANT,
#define OP_ATTR_TYPE(ENUM, CPP_TYPE) ENUM,
#include "tfrt/core_runtime/op_attr_type.def"
};

// Provide a way to get the OpAttrType for a specified C++ type at compile
// time.
template <typename T>
constexpr OpAttrType GetOpAttrType();

template <>
constexpr OpAttrType GetOpAttrType<OpAttrType>() {
  return OpAttrType::DTYPE;
}

template <>
constexpr OpAttrType GetOpAttrType<AggregateAttr>() {
  return OpAttrType::AGGREGATE;
}

template <>
constexpr OpAttrType GetOpAttrType<DenseAttr>() {
  return OpAttrType::DENSE;
}

template <>
constexpr OpAttrType GetOpAttrType<ShapeAttr>() {
  return OpAttrType::SHAPE;
}

template <>
constexpr OpAttrType GetOpAttrType<bf16>() {
  return OpAttrType::BF16;
}

template <>
constexpr OpAttrType GetOpAttrType<fp16>() {
  return OpAttrType::F16;
}

template <>
constexpr OpAttrType GetOpAttrType<std::complex<float>>() {
  return OpAttrType::COMPLEX64;
}

template <>
constexpr OpAttrType GetOpAttrType<std::complex<double>>() {
  return OpAttrType::COMPLEX128;
}

#define OP_ATTR_TYPE(ENUM, CPP_TYPE)               \
  template <>                                      \
  constexpr OpAttrType GetOpAttrType<CPP_TYPE>() { \
    return OpAttrType::ENUM;                       \
  }
#include "tfrt/core_runtime/op_attr_type.def"

// Return the size and alignment of the specified attribute type. `data` may be
// needed to decode custom scalar type to get host size and alignment.
std::pair<size_t, size_t> GetHostSizeAndAlignment(const void* data,
                                                  OpAttrType type);

// Return the name of the specified attribute type, e.g. "I32".
const char* GetNameString(OpAttrType type);

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_OP_ATTR_TYPE_H_
