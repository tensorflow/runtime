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

void BefAttrEncoder::EncodeAttrBase(BEFAttributeType type, size_t byte_count) {
  BEFAttrBase base;
  base.type = type;
  SetBEFAttrByteCount(byte_count, &base);
  EmitAlignment(alignof(BEFAttrBase));
  EmitBytes(llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(&base),
                               sizeof(base)));
}

void BefAttrEncoder::EncodeShapeAttrBase(size_t byte_count, int rank) {
  EncodeAttrBase(BEFAttributeType::kShape, byte_count);
  EmitByte(static_cast<uint8_t>(GetBEFShapeType(rank)));
  EmitByte(kDummyByte);
  EmitInt2(rank);
}

llvm::Error BefAttrEncoder::EncodeUnrankedShapeAttr() {
  EmitAlignment(alignof(BEFShapeAttr));
  EncodeShapeAttrBase(/*byte_count=*/sizeof(BEFShapeAttr), /*rank=*/-1);
  return llvm::Error::success();
}

llvm::Error BefAttrEncoder::EncodeRankedShapeAttr(ArrayRef<int64_t> dims) {
  size_t rank = dims.size();

  // If rank is 0, the shape attribute is emitted as BEFShapeAttr instead of
  // BEFRankedShapeAttr.
  if (rank == 0) {
    EmitAlignment(alignof(BEFShapeAttr));
    EncodeShapeAttrBase(/*byte_count=*/sizeof(BEFShapeAttr), /*rank=*/0);
    return llvm::Error::success();
  }

  // Otherwise, emit the shape with non-zero ranks as BEFRankedShapeAttr.
  size_t byte_count = sizeof(BEFRankedShapeAttr) + sizeof(int64_t) * (rank - 1);
  EmitAlignment(alignof(BEFRankedShapeAttr));
  EncodeShapeAttrBase(byte_count, rank);
  // Emit the dimensions.
  for (int i = 0; i < rank; ++i) {
    EmitInt8(dims[i]);
  }
  return llvm::Error::success();
}

llvm::Error BefAttrEncoder::EncodeStringAttr(string_view sv) {
  size_t length = sv.size();
  size_t byte_count = sizeof(BEFAttrBase) + sizeof(uint8_t) * length;
  // Here we directly cast the DType::Kind to BEFAttributeType. This is fine as
  // we explicitly reserve the entire range of valid DType::Kind values in
  // BEFAttributeType.
  //
  // TODO(tfrt-dev): Revisit the design of BEFAttributeType to avoid
  // static_cast.
  EncodeAttrBase(static_cast<BEFAttributeType>(DType::String), byte_count);
  EmitBytes(
      llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(sv.data()), length));
  return llvm::Error::success();
}

llvm::Error BefAttrEncoder::EncodeFuncAttr(string_view sv) {
  size_t length = sv.size();
  size_t byte_count = sizeof(BEFAttrBase) + sizeof(uint8_t) * length;
  // Here we directly cast the BEFDataType to BEFAttributeType. This is fine as
  // we explicitly reserve the entire range of valid BEFDataType values in
  // BEFAttributeType.
  //
  // TODO(tfrt-dev): Revisit the design of BEFAttributeType to avoid
  // static_cast.
  EncodeAttrBase(static_cast<BEFAttributeType>(BEFAttributeType::kFunc),
                 byte_count);
  EmitBytes(
      llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(sv.data()), length));
  return llvm::Error::success();
}

// Encode a list of attributes as an aggregate attribute in BEF. The `emitter`
// will be called with the indices sequentially and is expected to emit the
// bytes for this element and return the offset.
llvm::Error BefAttrEncoder::EncodeListAttr(
    size_t num_elements,
    llvm::function_ref<llvm::Expected<BEFAggregateAttrOffset32_t>(int)>
        emitter) {
  // Reserve header space in buffer.
  size_t header_size =
      num_elements > 0
          ? sizeof(BEFAggregateAttr) +
                sizeof(BEFAggregateAttrOffset32_t) * (num_elements - 1)
          : sizeof(BEFAggregateAttr);
  EmitRepeatedByte(BEFEmitter::kDummyByte, header_size);

  BEFAggregateAttr header;
  header.base.type = BEFAttributeType::kAggregate;
  header.num_elements = AssertAttrFieldSize32(num_elements);

  // Append array element to buffer.
  SmallVector<BEFAggregateAttrOffset32_t, 8> offsets;
  for (int i = 0; i < num_elements; ++i) {
    auto offset = emitter(i);
    if (!offset) return offset.takeError();
    offsets.push_back(*offset);
  }

  // Reset byte_count in header.
  SetBEFAttrByteCount(size(), &header.base);

  size_t element_offset = offsetof(BEFAggregateAttr, offsets);
  OverwriteBytes(0, &header, element_offset);
  OverwriteBytes(element_offset, offsets.data(),
                 sizeof(BEFAggregateAttrOffset32_t) * offsets.size());
  return llvm::Error::success();
}

llvm::Error BefAttrEncoder::EncodeShapeListAttr(const int64_t** dims,
                                                const int* num_dims,
                                                int num_values) {
  return EncodeListAttr(
      num_values, [&](int index) -> llvm::Expected<BEFAggregateAttrOffset32_t> {
        BefAttrEncoder elem_encoder;
        if (num_dims[index] < 0) {
          if (auto error = elem_encoder.EncodeUnrankedShapeAttr())
            return std::move(error);
        } else if (auto error = elem_encoder.EncodeRankedShapeAttr(
                       llvm::makeArrayRef(dims[index], num_dims[index]))) {
          return std::move(error);
        }
        EmitAlignment(elem_encoder.GetRequiredAlignment());
        BEFAggregateAttrOffset32_t offset = AssertAttrFieldSize32(size());
        EmitEmitter(elem_encoder);
        return offset;
      });
}

llvm::Error BefAttrEncoder::EncodeStringListAttr(const void* const* values,
                                                 const size_t* lengths,
                                                 int num_values) {
  return EncodeListAttr(
      num_values, [&](int index) -> llvm::Expected<BEFAggregateAttrOffset32_t> {
        BefAttrEncoder elem_encoder;
        if (auto error = elem_encoder.EncodeStringAttr(string_view(
                static_cast<const char*>(values[index]), lengths[index]))) {
          return std::move(error);
        }
        EmitAlignment(elem_encoder.GetRequiredAlignment());
        BEFAggregateAttrOffset32_t offset = AssertAttrFieldSize32(size());
        EmitEmitter(elem_encoder);
        return offset;
      });
}

llvm::Error BefAttrEncoder::EncodeFuncListAttr(const void* const* values,
                                               const size_t* lengths,
                                               int num_values) {
  return EncodeListAttr(
      num_values, [&](int index) -> llvm::Expected<BEFAggregateAttrOffset32_t> {
        BefAttrEncoder elem_encoder;
        if (auto error = elem_encoder.EncodeFuncAttr(string_view(
                static_cast<const char*>(values[index]), lengths[index]))) {
          return std::move(error);
        }
        EmitAlignment(elem_encoder.GetRequiredAlignment());
        BEFAggregateAttrOffset32_t offset = AssertAttrFieldSize32(size());
        EmitEmitter(elem_encoder);
        return offset;
      });
}

}  // namespace tfrt
