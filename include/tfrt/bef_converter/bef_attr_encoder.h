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

// This file declares the interface to the BEF Attr Encoder library.

#ifndef TFRT_BEF_CONVERTER_BEF_ATTR_ENCODER_H_
#define TFRT_BEF_CONVERTER_BEF_ATTR_ENCODER_H_

#include "tfrt/bef_converter/bef_emitter.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/type_traits.h"

namespace tfrt {

// TODO(zhangqiaorjc): Unify with BEFTypedAttributeEmitter.
// Consider a more open design using ADL. Encoder functions can be made free
// functions inside headers for TensorShape, and other user defined types.

// This class serializes BEF attributes.
class BefAttrEncoder : public BefEmitter {
 public:
  // Encode a generic attribute.
  //   Supported attribute types:
  //     char,
  //     uint8_t, uint16_t, uint32_t, uint64_t,
  //     int8_t, int16_t, int32_t, int64_t,
  //     float, double
  template <typename T>
  size_t EncodeAttr(T attr);

  // <Array size:VBR> [array payload]*
  // Empty element representation: Array, Aggregate, UnrankedShape
  //    <0>
  template <typename T>
  size_t EncodeArrayAttr(ArrayRef<T> array);

  size_t EncodeEmptyAttribute() {
    EmitByte(0);
    return size() - 1;
  }

  // Encode a unranked shape attribute.
  size_t EncodeUnrankedShapeAttr();

  // Encode a ranked shape attribute.
  size_t EncodeRankedShapeAttr(ArrayRef<int64_t> dims);

  // Encode a list of shapes as an aggregate attribute.
  size_t EncodeShapeListAttr(const int64_t** dims, const int* num_dims,
                             int num_values);

  // Encode a string attribute.
  size_t EncodeStringAttr(string_view sv);

  // Encode a list of strings as an aggregate attribute.
  size_t EncodeStringListAttr(const void* const* values, const size_t* lengths,
                              int num_values);

  // Encode a function attribute.
  size_t EncodeFuncAttr(string_view sv);

  // Encode a list of functions as an aggregate attribute.
  size_t EncodeFuncListAttr(const void* const* values, const size_t* lengths,
                            int num_values);

  // Reserve space for the header part of an aggregate attribute.
  size_t ReserveAggregatedAttrHeader(size_t element_count);

  // Complete encoding of an aggregate attribute.
  void EncodeCompleteAggregatedAttr(
      size_t element_count, size_t offset,
      ArrayRef<BEFAggregateAttrOffset32_t> offsets);

  // Reserve space for the header part of an array attribute.
  size_t ReserveArrayAttrHeader() {
    return ReserveHeaderSpace(alignof(BEFArrayAttr), sizeof(BEFArrayAttr));
  }

  // Complete encoding of an array attribute.
  void EncodeCompleteArrayAttr(size_t offset, BEFAttributeType element_type,
                               size_t element_count, size_t element_offset);

  // Reserve space for the header part of a dense (tensor) attribute.
  size_t ReserveDenseAttrHeader() {
    return ReserveHeaderSpace(alignof(BEFDenseAttr), sizeof(BEFDenseAttr));
  }

  // Complete encoding of a densor (tensor) attribute.
  void EncodeCompleteDenseAttr(size_t offset, DType::Kind element_type,
                               size_t rank, size_t shape_offset,
                               size_t num_elements, size_t element_offset);

 private:
  size_t ReserveHeaderSpace(size_t alignment, size_t header_size);

  size_t EncodeAttrBase(BEFAttributeType type, size_t byte_count);

  // Encode a list of attributes as an aggregate attribute in BEF. The `emitter`
  // will be called with the indices sequentially and is expected to emit the
  // bytes for this element and return the offset.
  size_t EncodeListAttr(
      size_t num_elements,
      llvm::function_ref<BEFAggregateAttrOffset32_t(int)> emitter);

  // A helper function to emit the common header for both ranked and unranked
  // shape attributes. If `rank` is a negative number, then this shape is
  // unranked.
  size_t EncodeShapeAttrBase(size_t byte_count, int rank);

  template <typename T>
  static constexpr bool kSupportedScalarAttributeType =
      IsOneOfTypes<T, char, uint8_t, uint16_t, uint32_t, uint64_t, int8_t,
                   int16_t, int32_t, int64_t, float, double>();
};

template <typename T>
size_t BefAttrEncoder::EncodeAttr(T attr) {
  static_assert(kSupportedScalarAttributeType<T>);

  const auto entry_size = sizeof(attr);

  // A shortcut for 1 byte sized attribute.
  if (entry_size == 1) {
    EmitByte(static_cast<uint8_t>(attr));
    return size() - 1;
  }

  EmitAlignment(alignof(T));
  const auto offset = size();
  EmitBytes(
      llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(&attr), entry_size));
  return offset;
}

// <Array size:VBR> [array payload]*
template <typename T>
size_t BefAttrEncoder::EncodeArrayAttr(ArrayRef<T> array) {
  static_assert(kSupportedScalarAttributeType<T>);

  const auto element_count = array.size();

  // Empty array attribute representation should be matched with
  // empty aggregate attribute representation: <0>
  if (element_count == 0) {
    return EncodeEmptyAttribute();
  }

  EmitAlignment(alignof(T),
                llvm::offsetToAlignment(size() + GetSizeOfVbrInt(element_count),
                                        llvm::Align(alignof(T))));
  const auto offset = size();
  EmitVbrInt(element_count);
  assert(size() % alignof(T) == 0);
  EmitBytes(llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(array.data()),
                               element_count * sizeof(T)));
  return offset;
}

}  // namespace tfrt

#endif  // TFRT_BEF_CONVERTER_BEF_ATTR_ENCODER_H_
