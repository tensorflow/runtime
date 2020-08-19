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

llvm::Error BEFTypedAttributeEncoder::EncodeShapeAttr(ArrayRef<int64_t> dims) {
  int num_dims = dims.size();
  if (num_dims < 0) {
    // TODO(b/156035574): Need to support unranked shape.
    return MakeStringError("does not yet support unknown rank");
  }

  uint16_t byte_count = AssertAttrFieldSize(sizeof(BEFShapeAttr));
  if (num_dims > 0) {
    byte_count =
        AssertAttrFieldSize(sizeof(int64_t) * (num_dims - 1) + byte_count);
  }

  EmitAlignment(alignof(BEFShapeAttr));  // FIXME
  EmitInt2(static_cast<uint16_t>(BEFAttributeType::kShape));
  EmitInt2(byte_count);
  EmitInt2(num_dims);
  EmitDummyByte();
  EmitDummyByte();
  for (int i = 0; i < num_dims; ++i) {
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
    if (auto error = elem_encoder.EncodeShapeAttr(
            llvm::makeArrayRef(dims[i], num_dims[i])))
      return error;
    EmitAlignment(alignof(BEFShapeAttr));
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
