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

// This class serializes BEF attributes by using BEFEmitter functions.
// The functions return the beginning offsets of the encoded attributes.
// The offsets are decided based on the alignment constraints of attributes.
class BefAttrEncoder : public BefEmitter {
 private:
  template <typename T>
  static constexpr bool kSupportedScalarAttributeType =
      IsOneOfTypes<T, char, uint8_t, uint16_t, uint32_t, uint64_t, int8_t,
                   int16_t, int32_t, int64_t, float, double>();

 public:
  // Encode a generic attribute.
  //   Support the types listed by kSupportedScalarAttributeType
  template <typename T>
  size_t EncodeAttr(T attr);

  // Encode the length of an array attribute.
  size_t EncodeArrayAttrHeader(size_t element_count, size_t payload_alignment);

  // Encode an array of generic type attributes.
  // Support the types listed by kSupportedScalarAttributeType
  //   Encoded format: <Array size:VBR> [array payload]*
  template <typename T>
  size_t EncodeArrayAttr(ArrayRef<T> array);

  // Encode an empty attribute (Array, Aggregate, UnrankedShape).
  //   Encoded format: <0>
  size_t EncodeEmptyAttr() {
    EmitByte(0);
    return size() - 1;
  }

  // Encode a string attribute.
  //   Encoded format: <length:vbr> <string payload>
  size_t EncodeStringAttr(string_view sv) {
    return EncodeArrayAttr(llvm::makeArrayRef(
        reinterpret_cast<const char*>(sv.data()), sv.size()));
  }

  // Encode a function attribute.
  //   Encoded format: <length:vbr> <function name payload>
  size_t EncodeFuncAttr(string_view sv) {
    return EncodeArrayAttr(llvm::makeArrayRef(
        reinterpret_cast<const char*>(sv.data()), sv.size()));
  }

  // Encode a ranked shape attribute.
  //   Encoded format:
  //    <rank:vbr> [shape entry : int64]*
  //     rank == 0 : unranked shape
  //     rank == 1 : zero ranked shape
  //     sizeof(dimension) == rank - 1
  size_t EncodeRankedShapeAttr(ArrayRef<int64_t> dims);

  // Encode an unranked shape attribute.
  //   Encoded format: <0>
  size_t EncodeUnrankedShapeAttr() { return EncodeEmptyAttr(); }

  // Encode a dense attribute header.
  //   Encoded format:
  //    <dtype:1B> <rank:vbr> <length:4B> [shape entry : int64]* <elements>
  size_t EncodeDenseAttrHeader(DType::Kind element_type,
                               ArrayRef<int64_t> shape,
                               size_t element_payload_size);

  // Encode a dense attribute.
  //   Encoded format:
  //    <dtype:1B> <rank:vbr> <length:4B> [shape entry : int64]* <elements>
  size_t EncodeDenseAttr(DType::Kind element_type, ArrayRef<int64_t> shape,
                         ArrayRef<uint8_t> element_payload);

  // Encode an aggregate attribute header.
  //   Encoded format:
  //    <EntryCount:VBR>
  //      [element type:2B]* [element offset:4B]* <payload size:4B>
  //      [element payloads]*
  size_t EncodeAggregatedAttrHeader(size_t element_count, size_t* type_offset,
                                    size_t* offset_offset);

  // Encode type information and element offset to an aggregate attribute header
  void EncodeAggregatedAttrEntryTypeAndOffset(size_t* type_offset,
                                              size_t* offset_offset,
                                              BEFAttributeType attribute_type,
                                              uint32_t element_offset);

  // Encode total byte size to an aggregate attribute header.
  void EncodeAggregatedAttrLength(size_t* offset_offset, size_t offset);

  // Encode a list of string attributes as an aggregate attribute.
  size_t EncodeStringListAttr(const void* const* values, const size_t* lengths,
                              int num_values);

  // Encode a list of function attributes as an aggregate attribute.
  size_t EncodeFuncListAttr(const void* const* values, const size_t* lengths,
                            int num_values);

  // Encode a list of shape attributes as an aggregate attribute.
  size_t EncodeShapeListAttr(const int64_t** shapes, const int* ranks,
                             int num_values);
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
    return EncodeEmptyAttr();
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
