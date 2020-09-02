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

//===- bef_attr_encoder.cc ------------------------------------------------===//
//
// This file defines the BEF Attr Encoder library.
//
//===----------------------------------------------------------------------===//

#include "tfrt/bef_converter/bef_attr_encoder.h"

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace {

BEFShapeType GetBEFShapeType(int rank) {
  if (rank < 0) return BEFShapeType::kUnranked;

  return BEFShapeType::kRanked;
}

}  // namespace

void BEFTypedAttributeEncoder::EncodeShapeAttrBase(int byte_count, int rank) {
  EmitInt2(static_cast<uint16_t>(BEFAttributeType::kShape));
  EmitInt2(byte_count);
  EmitByte(static_cast<uint8_t>(GetBEFShapeType(rank)));
  EmitByte(kDummyByte);
  EmitInt2(rank);
}

llvm::Error BEFTypedAttributeEncoder::EncodeUnrankedShapeAttr() {
  uint16_t byte_count = AssertAttrFieldSize(sizeof(BEFShapeAttr));

  EmitAlignment(alignof(BEFShapeAttr));

  EncodeShapeAttrBase(byte_count, /*rank=*/-1);

  return llvm::Error::success();
}

llvm::Error BEFTypedAttributeEncoder::EncodeRankedShapeAttr(
    ArrayRef<int64_t> dims) {
  // Emit the shape with non-zero ranks as BEFShapeAttr. If rank is 0, the shape
  // attribute is emitted as BEFShapeAttr instead of BEFRankedShapeAttr.

  size_t rank = dims.size();

  uint16_t byte_count = AssertAttrFieldSize(sizeof(BEFShapeAttr));

  byte_count = AssertAttrFieldSize(sizeof(int64_t) * rank + byte_count);

  EmitAlignment(alignof(BEFRankedShapeAttr));

  EncodeShapeAttrBase(byte_count, rank);

  // Emit the dimensions.
  for (int i = 0; i < rank; ++i) {
    EmitInt8(dims[i]);
  }

  return llvm::Error::success();
}

llvm::Error BEFTypedAttributeEncoder::EncodeStringAttr(string_view sv) {
  size_t length = sv.size();
  uint16_t byte_count =
      AssertAttrFieldSize(sizeof(BEFAttrBase) + sizeof(uint8_t) * length);

  EmitAlignment(alignof(BEFStringAttr));
  EmitInt2(static_cast<uint16_t>(BEFDataType::kString));
  EmitInt2(byte_count);
  EmitBytes(
      llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(sv.data()), length));

  return llvm::Error::success();
}

llvm::Error BEFTypedAttributeEncoder::EncodeShapeListAttr(const int64_t** dims,
                                                          const int* num_dims,
                                                          int num_values) {
  // Reserve header space in buffer.
  size_t header_size = num_values > 0 ? sizeof(BEFAggregateAttr) +
                                            sizeof(uint16_t) * (num_values - 1)
                                      : sizeof(BEFAggregateAttr);
  EmitRepeatedByte(BEFEmitter::kDummyByte, header_size);

  BEFAggregateAttr header;
  header.base.type = BEFAttributeType::kAggregate;
  header.num_elements = num_values;

  // Append array element to buffer.
  SmallVector<uint16_t, 8> offsets;
  for (int i = 0; i < num_values; ++i) {
    BEFTypedAttributeEncoder elem_encoder;
    if (num_dims[i] < 0) {
      if (auto error = elem_encoder.EncodeUnrankedShapeAttr()) return error;
    } else if (auto error = elem_encoder.EncodeRankedShapeAttr(
                   llvm::makeArrayRef(dims[i], num_dims[i]))) {
      return error;
    }
    EmitAlignment(elem_encoder.GetRequiredAlignment());
    offsets.push_back(AssertAttrFieldSize(size()));
    EmitEmitter(elem_encoder);
  }

  // Reset byte_count in header.
  header.base.byte_count = AssertAttrFieldSize(size());

  size_t element_offset = offsetof(BEFAggregateAttr, offsets);
  OverwriteBytes(0, &header, element_offset);
  OverwriteBytes(element_offset, offsets.data(),
                 sizeof(uint16_t) * offsets.size());
  return llvm::Error::success();
}

llvm::Error BEFTypedAttributeEncoder::EncodeStringListAttr(
    const void* const* values, const size_t* lengths, int num_values) {
  // Reserve header space in buffer.
  size_t header_size = num_values > 0 ? sizeof(BEFAggregateAttr) +
                                            sizeof(uint16_t) * (num_values - 1)
                                      : sizeof(BEFAggregateAttr);
  EmitRepeatedDummyByte(header_size);

  BEFAggregateAttr header;
  header.base.type = BEFAttributeType::kAggregate;
  header.num_elements = num_values;

  // Append array element to buffer.
  SmallVector<uint16_t, 8> offsets;
  for (int i = 0; i < num_values; ++i) {
    BEFTypedAttributeEncoder elem_encoder;
    if (auto error = elem_encoder.EncodeStringAttr(
            string_view(static_cast<const char*>(values[i]), lengths[i])))
      return error;
    EmitAlignment(alignof(BEFStringAttr));
    offsets.push_back(AssertAttrFieldSize(size()));
    EmitEmitter(elem_encoder);
  }

  // Reset byte_count in header.
  header.base.byte_count = AssertAttrFieldSize(size());

  size_t element_offset = offsetof(BEFAggregateAttr, offsets);
  OverwriteBytes(0, &header, element_offset);
  OverwriteBytes(element_offset, offsets.data(),
                 sizeof(uint16_t) * offsets.size());
  return llvm::Error::success();
}

}  // namespace tfrt
