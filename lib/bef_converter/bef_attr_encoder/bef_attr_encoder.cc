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

size_t BefAttrEncoder::EncodeAttrBase(BEFAttributeType type,
                                      size_t byte_count) {
  BEFAttrBase base;
  base.type = type;
  SetBEFAttrByteCount(byte_count, &base);
  EmitAlignment(alignof(BEFAttrBase));
  const size_t offset = size();
  EmitBytes(llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(&base),
                               sizeof(base)));
  return offset;
}

size_t BefAttrEncoder::EncodeShapeAttrBase(size_t byte_count, int rank) {
  const size_t offset = EncodeAttrBase(BEFAttributeType::kShape, byte_count);
  EmitByte(static_cast<uint8_t>(GetBEFShapeType(rank)));
  EmitByte(kDummyByte);
  EmitInt2(rank);
  return offset;
}

size_t BefAttrEncoder::EncodeUnrankedShapeAttr() {
  EmitAlignment(alignof(BEFShapeAttr));
  return EncodeShapeAttrBase(/*byte_count=*/sizeof(BEFShapeAttr), /*rank=*/-1);
}

size_t BefAttrEncoder::EncodeRankedShapeAttr(ArrayRef<int64_t> dims) {
  const size_t rank = dims.size();

  // If rank is 0, the shape attribute is emitted as BEFShapeAttr instead of
  // BEFRankedShapeAttr.
  if (rank == 0) {
    EmitAlignment(alignof(BEFShapeAttr));
    return EncodeShapeAttrBase(/*byte_count=*/sizeof(BEFShapeAttr), /*rank=*/0);
  }

  // Otherwise, emit the shape with non-zero ranks as BEFRankedShapeAttr.
  EmitAlignment(alignof(BEFRankedShapeAttr));
  size_t byte_count = sizeof(BEFRankedShapeAttr) + sizeof(int64_t) * (rank - 1);
  const size_t offset = EncodeShapeAttrBase(byte_count, rank);
  // Emit the dimensions.
  for (int64_t dim : dims) {
    EmitInt8(dim);
  }
  return offset;
}

size_t BefAttrEncoder::EncodeStringAttr(string_view sv) {
  const size_t length = sv.size();
  const size_t byte_count = sizeof(BEFAttrBase) + sizeof(uint8_t) * length;
  // Here we directly cast the DType::Kind to BEFAttributeType. This is fine as
  // we explicitly reserve the entire range of valid DType::Kind values in
  // BEFAttributeType.
  //
  // TODO(tfrt-dev): Revisit the design of BEFAttributeType to avoid
  // static_cast.
  const size_t offset =
      EncodeAttrBase(static_cast<BEFAttributeType>(DType::String), byte_count);
  EmitBytes(
      llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(sv.data()), length));
  return offset;
}

size_t BefAttrEncoder::EncodeFuncAttr(string_view sv) {
  const size_t length = sv.size();
  const size_t byte_count = sizeof(BEFAttrBase) + sizeof(uint8_t) * length;
  // Here we directly cast the BEFDataType to BEFAttributeType. This is fine as
  // we explicitly reserve the entire range of valid BEFDataType values in
  // BEFAttributeType.
  //
  const size_t offset = EncodeAttrBase(BEFAttributeType::kFunc, byte_count);
  EmitBytes(
      llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(sv.data()), length));
  return offset;
}

size_t BefAttrEncoder::ReserveHeaderSpace(size_t alignment,
                                          size_t header_size) {
  EmitAlignment(alignment);
  const size_t offset = size();
  EmitRepeatedByte(BefEmitter::kDummyByte, header_size);
  return offset;
}

size_t BefAttrEncoder::ReserveAggregatedAttrHeader(size_t element_count) {
  const size_t header_size =
      element_count > 0
          ? sizeof(BEFAggregateAttr) +
                sizeof(BEFAggregateAttrOffset32_t) * (element_count - 1)
          : sizeof(BEFAggregateAttr);
  return ReserveHeaderSpace(alignof(BEFAggregateAttr), header_size);
}

void BefAttrEncoder::EncodeCompleteAggregatedAttr(
    size_t element_count, size_t offset,
    ArrayRef<BEFAggregateAttrOffset32_t> offsets) {
  BEFAggregateAttr header;
  header.base.type = BEFAttributeType::kAggregate;
  header.num_elements = AssertAttrFieldSize32(element_count);

  // Reset byte_count in header.
  SetBEFAttrByteCount(size() - offset, &header.base);

  const size_t element_offset = offsetof(BEFAggregateAttr, offsets);
  OverwriteBytes(offset, &header, element_offset);
  OverwriteBytes(offset + element_offset, offsets.data(),
                 sizeof(BEFAggregateAttrOffset32_t) * offsets.size());
}

void BefAttrEncoder::EncodeCompleteArrayAttr(size_t offset,
                                             BEFAttributeType element_type,
                                             size_t element_count,
                                             size_t element_offset) {
  BEFArrayAttr header;
  header.base.type = GetArrayAttributeType(element_type);
  header.num_elements = AssertAttrFieldSize32(element_count);
  header.element_offset = element_offset;

  SetBEFAttrByteCount(size() - offset, &header.base);
  OverwriteBytes(offset, &header, sizeof(header));
}

void BefAttrEncoder::EncodeCompleteDenseAttr(size_t offset,
                                             DType::Kind element_type,
                                             size_t rank, size_t shape_offset,
                                             size_t num_elements,
                                             size_t element_offset) {
  BEFDenseAttr header;
  header.base.type = GetDenseAttributeType(element_type);
  header.rank = AssertAttrFieldSize16(rank);
  header.shape_offset = AssertAttrFieldSize16(shape_offset);
  header.num_elements = AssertAttrFieldSize32(num_elements);
  header.element_offset = AssertAttrFieldSize32(element_offset);
  SetBEFAttrByteCount(size() - offset, &header.base);
  OverwriteBytes(offset, &header, sizeof(header));
}

// Encode a list of attributes as an aggregate attribute in BEF. The `emitter`
// will be called with the indices sequentially and is expected to emit the
// bytes for this element and return the offset.
size_t BefAttrEncoder::EncodeListAttr(
    size_t num_elements,
    llvm::function_ref<BEFAggregateAttrOffset32_t(int)> emitter) {
  const size_t offset = ReserveAggregatedAttrHeader(num_elements);
  // Append array element to buffer.
  SmallVector<BEFAggregateAttrOffset32_t, 8> offsets;
  for (int i = 0; i < num_elements; ++i) {
    offsets.push_back(emitter(i) - offset);
  }
  EncodeCompleteAggregatedAttr(num_elements, offset, offsets);
  return offset;
}

size_t BefAttrEncoder::EncodeShapeListAttr(const int64_t** dims,
                                           const int* num_dims,
                                           int num_values) {
  return EncodeListAttr(
      num_values, [&](int index) -> BEFAggregateAttrOffset32_t {
        return (num_dims[index] < 0) ? EncodeUnrankedShapeAttr()
                                     : EncodeRankedShapeAttr(llvm::makeArrayRef(
                                           dims[index], num_dims[index]));
      });
}

size_t BefAttrEncoder::EncodeStringListAttr(const void* const* values,
                                            const size_t* lengths,
                                            int num_values) {
  return EncodeListAttr(
      num_values, [&](int index) -> BEFAggregateAttrOffset32_t {
        return EncodeStringAttr(string_view(
            static_cast<const char*>(values[index]), lengths[index]));
      });
}

size_t BefAttrEncoder::EncodeFuncListAttr(const void* const* values,
                                          const size_t* lengths,
                                          int num_values) {
  return EncodeListAttr(
      num_values, [&](int index) -> BEFAggregateAttrOffset32_t {
        return EncodeFuncAttr(string_view(
            static_cast<const char*>(values[index]), lengths[index]));
      });
}

}  // namespace tfrt
