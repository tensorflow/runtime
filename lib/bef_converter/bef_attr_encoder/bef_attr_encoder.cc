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
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

size_t BefAttrEncoder::EncodeEmptyAttr() { return EncodeAttr<AttrSizeT>(0); }

size_t BefAttrEncoder::EncodeHeader(BefAttrBase* base, uint8_t alignment,
                                    DType element_type, uint16_t prefix_size,
                                    AttrSizeT byte_size, AttrSizeT emit_size) {
  base->alignment = alignment;
  base->element_type = element_type;
  base->prefix_size = prefix_size;
  base->byte_size = byte_size;

  EmitAlignment(alignment, llvm::offsetToAlignment(size() + prefix_size,
                                                   llvm::Align(alignment)));
  const auto offset = size();
  EmitBytes(llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(base),
                               sizeof(BefAttrBase) + emit_size));
  return offset;
}

size_t BefAttrEncoder::EncodeRankedShapeAttr(ArrayRef<int64_t> dims) {
  BefShapeAttr header;

  header.rank = dims.size();
  if (header.rank == 0) {
    return EncodeHeader(reinterpret_cast<BefAttrBase*>(&header.base),
                        /*alignment=*/alignof(BefAttrBase),
                        /*element_type=*/DType::Invalid,
                        /*prefix_size=*/0,
                        /*byte_size=*/sizeof(BefAttrBase) + sizeof(AttrSizeT),
                        /*emit_size=*/sizeof(AttrSizeT));
  }
  const size_t offset =
      EncodeHeader(reinterpret_cast<BefAttrBase*>(&header.base),
                   /*alignment=*/alignof(AttrShapeT),
                   /*element_type=*/DType::Invalid,
                   /*prefix_size=*/sizeof(BefAttrBase) + sizeof(AttrSizeT),
                   /*byte_size=*/sizeof(BefAttrBase) + sizeof(AttrSizeT) +
                       sizeof(AttrShapeT) * header.rank,
                   /*emit_size=*/sizeof(AttrSizeT));

  for (auto shape_entry : dims) Emit<int64_t>(shape_entry);

  return offset;
}

size_t BefAttrEncoder::EncodeDenseAttrHeader(DType element_type,
                                             ArrayRef<int64_t> dims,
                                             size_t rawdata_size) {
  BefDenseAttr header;
  header.rank = dims.size();
  header.element_count =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());

  assert(header.element_count * GetHostSize(element_type) == rawdata_size);

  header.element_offset =
      BefAttrOffsetOf(BefDenseAttr, dims) + sizeof(AttrShapeT) * header.rank;

  const size_t offset =
      EncodeHeader(reinterpret_cast<BefAttrBase*>(&header.base),
                   /*alignment=*/kAttributeTensorAlignment,
                   /*element_type=*/element_type,
                   /*prefix_size=*/header.element_offset,
                   /*byte_size=*/header.element_offset + rawdata_size,
                   /*emit_size=*/sizeof(AttrSizeT) * 3);

  for (auto shape_entry : dims) Emit<int64_t>(shape_entry);
  return offset;
}

size_t BefAttrEncoder::EncodeDenseAttr(DType element_type,
                                       ArrayRef<int64_t> dims,
                                       ArrayRef<uint8_t> element_payload) {
  auto offset =
      EncodeDenseAttrHeader(element_type, dims, element_payload.size());
  EmitBytes(element_payload);
  return offset;
}

size_t BefAttrEncoder::EncodeArrayAttrHeader(size_t element_count,
                                             size_t payload_alignment) {
  if (payload_alignment <= alignof(AttrSizeT)) {
    EmitAlignment(alignof(AttrSizeT));
  } else {
    EmitAlignment(payload_alignment,
                  llvm::offsetToAlignment(size() + sizeof(AttrSizeT),
                                          llvm::Align(payload_alignment)));
  }
  const auto offset = size();
  Emit<uint32_t>(element_count);
  return offset;
}

size_t BefAttrEncoder::EncodeAggregatedAttrHeader(size_t max_alignment,
                                                  size_t element_count,
                                                  size_t* offset_offset) {
  assert(element_count > 0);

  BefAggregateAttr header;

  header.element_count = element_count;
  const size_t offset =
      EncodeHeader(reinterpret_cast<BefAttrBase*>(&header),
                   /*alignment=*/std::max(max_alignment, sizeof(AttrSizeT)),
                   /*element_type=*/DType::Invalid,
                   /*prefix_size=*/BefAttrOffsetOf(BefAggregateAttr, offsets) +
                       element_count * sizeof(AttrSizeT),
                   /*byte_size=*/0,
                   /*emit_size=*/sizeof(AttrSizeT));

  *offset_offset = size();
  for (int i = 0; i < element_count; ++i) Emit<uint32_t>(0);
  return offset;
}

void BefAttrEncoder::EncodeAggregatedAttrEntryTypeAndOffset(
    size_t offset, size_t* offset_offset, BEFAttributeType attribute_type,
    AttrSizeT element_offset) {
  OverwriteBytes(element_offset - sizeof(BEFAttributeType), &attribute_type,
                 sizeof(BEFAttributeType));
  element_offset -= offset;
  OverwriteBytes(*offset_offset, &element_offset, sizeof(element_offset));
  *offset_offset += sizeof(element_offset);
}

void BefAttrEncoder::EncodeAggregatedAttrLength(size_t offset) {
  const auto total_byte_size = static_cast<AttrSizeT>(size() - offset);
  OverwriteBytes(offset + offsetof(BefAttrBase, byte_size), &total_byte_size,
                 sizeof(total_byte_size));
}

// Encode a list of attributes as an aggregate attribute in BEF. The `emitter`
// will be called with the indices sequentially and is expected to emit the
// bytes for this element and return the offset.
size_t BefAttrEncoder::EncodeListAttr(
    size_t num_values, BEFAttributeType type, size_t max_alignment,
    llvm::function_ref<AttrSizeT(int)> emitter) {
  if (num_values == 0) return EncodeEmptyAttr();

  size_t offset_offset;
  const size_t offset =
      EncodeAggregatedAttrHeader(max_alignment, num_values, &offset_offset);
  for (int idx = 0; idx < num_values; ++idx) {
    // Reserve a space for BEFAttributeType (1B)
    EmitByte(kDummyByte);

    EncodeAggregatedAttrEntryTypeAndOffset(offset, &offset_offset, type,
                                           emitter(idx));
  }
  EncodeAggregatedAttrLength(offset);
  return offset;
}

size_t BefAttrEncoder::EncodeShapeListAttr(const int64_t** shapes,
                                           const int* ranks, int num_values) {
  return EncodeListAttr(num_values, BEFAttributeType::kShape,
                        alignof(AttrShapeT), [&](int index) -> AttrSizeT {
                          return (ranks[index] < 0)
                                     ? EncodeUnrankedShapeAttr()
                                     : EncodeRankedShapeAttr(llvm::makeArrayRef(
                                           shapes[index], ranks[index]));
                        });
}

size_t BefAttrEncoder::EncodeStringListAttr(const void* const* values,
                                            const size_t* lengths,
                                            int num_values) {
  return EncodeListAttr(
      num_values, static_cast<BEFAttributeType>(DType::String),
      alignof(AttrSizeT), [&](int index) -> AttrSizeT {
        return EncodeStringAttr(string_view(
            static_cast<const char*>(values[index]), lengths[index]));
      });
}

size_t BefAttrEncoder::EncodeFuncListAttr(const void* const* values,
                                          const size_t* lengths,
                                          int num_values) {
  return EncodeListAttr(
      num_values, BEFAttributeType::kFunc, alignof(AttrSizeT),
      [&](int index) -> AttrSizeT {
        return EncodeFuncAttr(string_view(
            static_cast<const char*>(values[index]), lengths[index]));
      });
}

}  // namespace tfrt
