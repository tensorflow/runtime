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

// This file defines the BEF Attr Encoder library.

#include "tfrt/bef_converter/bef_attr_encoder.h"

#include <cstdint>
#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"
#include "tfrt/bef_converter/bef_emitter.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// ShapeAttr encoding format:
//    <rank:vbr> [shape entry : int64]*
//     rank == 0 : unranked shape
//     rank == 1 : zero ranked shape
//     sizeof(dimension) == rank - 1
size_t BefAttrEncoder::EncodeRankedShapeAttr(ArrayRef<int64_t> dims) {
  const auto rank = dims.size() + 1;
  if (rank > 1) {
    EmitAlignment(alignof(int64_t),
                  llvm::offsetToAlignment(size() + GetSizeOfVbrInt(rank),
                                          llvm::Align(alignof(int64_t))));
  }
  const auto offset = size();
  EmitVbrInt(rank);
  for (auto shape_entry : dims) {
    EmitInt8(shape_entry);
  }
  return offset;
}

// DenseAttr encoding format:
//    <dtype:1B> <rank:vbr> <length:4B> [shape entry : int64]* <elements>
size_t BefAttrEncoder::EncodeDenseAttrHeader(DType::Kind element_type,
                                             ArrayRef<int64_t> shape,
                                             size_t element_payload_size) {
  assert(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()) *
             GetDTypeByteSize(element_type) ==
         element_payload_size);

  const auto rank = shape.size();
  EmitAlignment(
      alignof(int64_t),
      llvm::offsetToAlignment(size() + sizeof(DType::Kind) +
                                  GetSizeOfVbrInt(rank) + sizeof(uint32_t),
                              llvm::Align(alignof(int64_t))));
  const auto offset = size();
  EmitByte(element_type);
  EmitVbrInt(rank);
  EmitInt4(size() - offset + sizeof(uint32_t) + sizeof(int64_t) * rank +
           element_payload_size);
  for (auto shape_entry : shape) {
    EmitInt8(shape_entry);
  }
  return offset;
}

size_t BefAttrEncoder::EncodeDenseAttr(DType::Kind element_type,
                                       ArrayRef<int64_t> shape,
                                       ArrayRef<uint8_t> element_payload) {
  auto offset =
      EncodeDenseAttrHeader(element_type, shape, element_payload.size());
  EmitBytes(element_payload);
  return offset;
}

// ArrayAttr encoding format:
//    <EntryCount:VBR> [element payloads]*
size_t BefAttrEncoder::EncodeArrayAttrHeader(size_t element_count,
                                             size_t payload_alignment) {
  EmitAlignment(payload_alignment,
                llvm::offsetToAlignment(size() + GetSizeOfVbrInt(element_count),
                                        llvm::Align(payload_alignment)));
  const auto offset = size();
  EmitVbrInt(element_count);
  return offset;
}

// AggregateAttr encoding format:
//    <element count:VBR>
//      [element type:2B]* [element offset:4B]* <payload size:4B>
//      [element payloads]*
size_t BefAttrEncoder::EncodeAggregatedAttrHeader(size_t element_count,
                                                  size_t* type_offset,
                                                  size_t* offset_offset) {
  const size_t prefix_size =
      GetSizeOfVbrInt(element_count) +
      element_count * (sizeof(uint32_t) + sizeof(uint16_t)) + sizeof(uint32_t);
  EmitAlignment(kAttributeMaxAlignment,
                llvm::offsetToAlignment(size() + prefix_size,
                                        llvm::Align(kAttributeMaxAlignment)));
  const auto offset = size();
  EmitVbrInt(element_count);

  *type_offset = size();
  for (int i = 0; i < element_count; ++i) EmitInt2(0);

  *offset_offset = size();
  for (int i = 0; i < element_count; ++i) EmitInt4(0);

  // Reserve a space for total byte size.
  EmitInt4(0);
  return offset;
}

void BefAttrEncoder::EncodeAggregatedAttrEntryTypeAndOffset(
    size_t* type_offset, size_t* offset_offset, BEFAttributeType attribute_type,
    uint32_t element_offset) {
  OverwriteBytes(*type_offset, &attribute_type, sizeof(BEFAttributeType));
  *type_offset += sizeof(BEFAttributeType);

  OverwriteBytes(*offset_offset, &element_offset, sizeof(element_offset));
  *offset_offset += sizeof(element_offset);
}

void BefAttrEncoder::EncodeAggregatedAttrLength(size_t* offset_offset,
                                                size_t offset) {
  const uint32_t total_byte_size = static_cast<uint32_t>(size() - offset);
  OverwriteBytes(*offset_offset, &total_byte_size, sizeof(total_byte_size));
}

size_t BefAttrEncoder::EncodeStringListAttr(const void* const* values,
                                            const size_t* lengths,
                                            int num_values) {
  if (num_values == 0) return EncodeEmptyAttr();
  size_t type_offset;
  size_t offset_offset;
  const size_t offset =
      EncodeAggregatedAttrHeader(num_values, &type_offset, &offset_offset);
  for (int idx = 0; idx < num_values; ++idx) {
    const uint32_t element_offset =
        EncodeStringAttr(
            string_view(static_cast<const char*>(values[idx]), lengths[idx])) -
        offset;
    EncodeAggregatedAttrEntryTypeAndOffset(
        &type_offset, &offset_offset,
        static_cast<BEFAttributeType>(DType::String), element_offset);
  }
  EncodeAggregatedAttrLength(&offset_offset, offset);
  return offset;
}

size_t BefAttrEncoder::EncodeFuncListAttr(const void* const* values,
                                          const size_t* lengths,
                                          int num_values) {
  if (num_values == 0) return EncodeEmptyAttr();
  size_t type_offset;
  size_t offset_offset;
  const size_t offset =
      EncodeAggregatedAttrHeader(num_values, &type_offset, &offset_offset);
  for (int idx = 0; idx < num_values; ++idx) {
    const uint32_t element_offset =
        EncodeFuncAttr(
            string_view(static_cast<const char*>(values[idx]), lengths[idx])) -
        offset;
    EncodeAggregatedAttrEntryTypeAndOffset(
        &type_offset, &offset_offset, BEFAttributeType::kFunc, element_offset);
  }
  EncodeAggregatedAttrLength(&offset_offset, offset);
  return offset;
}

size_t BefAttrEncoder::EncodeShapeListAttr(const int64_t** shapes,
                                           const int* ranks, int num_values) {
  if (num_values == 0) return EncodeEmptyAttr();
  size_t type_offset;
  size_t offset_offset;
  const size_t offset =
      EncodeAggregatedAttrHeader(num_values, &type_offset, &offset_offset);
  for (int idx = 0; idx < num_values; ++idx) {
    const auto shape = shapes[idx];
    const auto rank = ranks[idx];
    const uint32_t element_offset =
        ((shape == nullptr || rank < 0)
             ? EncodeUnrankedShapeAttr()
             : EncodeRankedShapeAttr(llvm::makeArrayRef(shape, rank))) -
        offset;

    EncodeAggregatedAttrEntryTypeAndOffset(
        &type_offset, &offset_offset, BEFAttributeType::kShape, element_offset);
  }
  EncodeAggregatedAttrLength(&offset_offset, offset);
  return offset;
}

}  // namespace tfrt
