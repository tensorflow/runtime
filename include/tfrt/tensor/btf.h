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

// This file contains constants and functions related to BTF (Binary Tensor
// Format).

#ifndef TFRT_TENSOR_BTF_H_
#define TFRT_TENSOR_BTF_H_

#include <cstdint>

#include "tfrt/dtype/dtype.h"
#include "tfrt/support/byte_order.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace btf {

// enum for tensor dtype. Do not change the enum values: The enum values are
// persisted in Binary Tensor Format.
enum class TensorDType : uint8_t {
  kInt8 = 0,
  kInt16 = 1,
  kInt32 = 2,
  kInt64 = 3,
  kFloat32 = 4,
  kFloat64 = 5,
  kUInt8 = 6,
  kUInt16 = 7,
  kUInt32 = 8,
  kUInt64 = 9,
};

DType ToDTypeKind(TensorDType type);

Expected<TensorDType> ToTensorDType(DType type);

raw_ostream& operator<<(raw_ostream& os, const TensorDType& dtype);

// Convert dtype to TensorDtype enum.
constexpr TensorDType GetTensorDType(int8_t) { return TensorDType::kInt8; }
constexpr TensorDType GetTensorDType(int16_t) { return TensorDType::kInt16; }
constexpr TensorDType GetTensorDType(int32_t) { return TensorDType::kInt32; }
constexpr TensorDType GetTensorDType(int64_t) { return TensorDType::kInt64; }
constexpr TensorDType GetTensorDType(float) { return TensorDType::kFloat32; }
constexpr TensorDType GetTensorDType(double) { return TensorDType::kFloat64; }
constexpr TensorDType GetTensorDType(uint8_t) { return TensorDType::kUInt8; }
constexpr TensorDType GetTensorDType(uint16_t) { return TensorDType::kUInt16; }
constexpr TensorDType GetTensorDType(uint32_t) { return TensorDType::kUInt32; }
constexpr TensorDType GetTensorDType(uint64_t) { return TensorDType::kUInt64; }

// This class should be kept in sync with the TensorLayout enum in
// utils/mnist/btf_writer.py.
enum class TensorLayout : uint8_t {
  kRMD = 0,
  kCOO_EXPERIMENTAL = 1,
};

raw_ostream& operator<<(raw_ostream& os, const TensorLayout& layout);

// Tensor header in the Binary Tensor Format. This struct has to directly map to
// the on-disk structure of BTF files.
#pragma pack(push, 1)
struct TensorHeader {
  uint64_t rank;
  // dtype, layout and the padding together occupy 64 bits.
  TensorDType dtype;
  TensorLayout layout;
  uint8_t padding[6];
};
#pragma pack(pop)

// Assure that the TenshorHead structure has the correct size and structures for
// wire compatibility.
static_assert(sizeof(TensorHeader) == 16,
              "TensorHeader packed to the wrong size.");
static_assert(offsetof(TensorHeader, dtype) == 8,
              "dtype does not start at the correct offset.");
static_assert(offsetof(TensorHeader, layout) == 9,
              "layout does not start at the correct offset.");

}  // namespace btf
}  // namespace tfrt

#endif  // TFRT_TENSOR_BTF_H_
