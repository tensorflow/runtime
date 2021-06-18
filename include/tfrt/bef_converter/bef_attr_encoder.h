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

#include "tfrt/bef/bef_encoding.h"
#include "tfrt/bef_converter/bef_emitter.h"
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
  template <typename T>
  size_t EncodeArrayAttr(ArrayRef<T> array);

  // Encode an empty attribute (Array, Aggregate, UnrankedShape).
  size_t EncodeEmptyAttr();

  // Encode a string attribute.
  size_t EncodeStringAttr(string_view sv) {
    return EncodeArrayAttr(llvm::makeArrayRef(
        reinterpret_cast<const char*>(sv.data()), sv.size()));
  }

  // Encode a function attribute.
  size_t EncodeFuncAttr(string_view sv) {
    return EncodeArrayAttr(llvm::makeArrayRef(
        reinterpret_cast<const char*>(sv.data()), sv.size()));
  }

  // Encode a ranked shape attribute.
  size_t EncodeRankedShapeAttr(ArrayRef<AttrShapeT> dims);

  // Encode an unranked shape attribute.
  size_t EncodeUnrankedShapeAttr() { return EncodeEmptyAttr(); }

  // Encode a dense attribute header.
  size_t EncodeDenseAttrHeader(DType element_type, ArrayRef<AttrShapeT> dims,
                               size_t rawdata_size);

  // Encode a dense attribute.
  size_t EncodeDenseAttr(DType element_type, ArrayRef<AttrShapeT> dims,
                         ArrayRef<uint8_t> element_payload);

  // Encode an aggregate attribute header.
  size_t EncodeAggregatedAttrHeader(size_t max_alignment, size_t element_count,
                                    size_t* offset_offset);

  // Encode type information and element offset to an aggregate attribute header
  void EncodeAggregatedAttrEntryTypeAndOffset(size_t offset,
                                              size_t* offset_offset,
                                              BEFAttributeType attribute_type,
                                              AttrSizeT element_offset);

  // Encode total byte size to an aggregate attribute header.
  void EncodeAggregatedAttrLength(size_t offset);

  // Encode a list of string attributes as an aggregate attribute.
  size_t EncodeStringListAttr(const void* const* values, const size_t* lengths,
                              int num_values);

  // Encode a list of function attributes as an aggregate attribute.
  size_t EncodeFuncListAttr(const void* const* values, const size_t* lengths,
                            int num_values);

  // Encode a list of shape attributes as an aggregate attribute.
  size_t EncodeShapeListAttr(const int64_t** shapes, const int* ranks,
                             int num_values);

  // A helper function to encode an AggregateAttr having same type entries.
  size_t EncodeListAttr(size_t num_values, BEFAttributeType type,
                        size_t max_alignment,
                        llvm::function_ref<AttrSizeT(int)> emitter);

 private:
  // Fill and encode BefAttrBase struct content.
  size_t EncodeHeader(BefAttrBase* base, uint8_t alignment, DType element_type,
                      uint16_t prefix_size, AttrSizeT byte_size,
                      AttrSizeT emit_size);
};

template <typename T>
size_t BefAttrEncoder::EncodeAttr(T attr) {
  static_assert(kSupportedScalarAttributeType<T>, "unsupported attribute");

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
  static_assert(kSupportedScalarAttributeType<T>, "unsupported attribute");

  const auto element_count = array.size();

  if (element_count == 0) {
    return EncodeEmptyAttr();
  }

  const auto offset = EncodeArrayAttrHeader(element_count, alignof(T));
  assert(size() % alignof(T) == 0);
  EmitBytes(llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(array.data()),
                               element_count * sizeof(T)));
  return offset;
}

}  // namespace tfrt

#endif  // TFRT_BEF_CONVERTER_BEF_ATTR_ENCODER_H_
