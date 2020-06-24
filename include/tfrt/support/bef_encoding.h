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

//===- bef_encoding.h -------------------------------------------*- C++ -*-===//
//
// This file declares constants used when interfacing with the "Binary Executor
// Format" (BEF) files.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_SUPPORT_BEF_ENCODING_H_
#define TFRT_SUPPORT_BEF_ENCODING_H_

#include <cstddef>
#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// Magic numbers for the file header.  These are the first two bytes of the
// file.
enum : uint8_t {
  kBEFMagic1 = 0x0B,
  kBEFMagic2 = 0xEF,

  // This is the only known version of BEF files at the moment, new numbers
  // should be used when/if a format break is introduced.
  kBEFVersion0 = 0,
};

// These are the section ID's for the standard sections.  Each section is
// encoded with an ID, followed by a length, followed by the contents of the
// section:
//
// <BEFSectionID> <length> <... data ...>
//
enum class BEFSectionID : uint8_t {
  // The FormatVersion section has a single byte payload which is the version
  // number.
  kFormatVersion = 0,

  // This is a list of filenames used for location information.  This is kept
  // in a separate section from other strings because we don't expect it to be
  // frequently accessed.
  kLocationFilenames = 1,

  // This is a list of positions referred to by location information.  Each
  // position is a triple of FilenameIndex, line, column, and offsets into this
  // section are used by location references.
  kLocationPositions = 2,

  // The strings section contains NUL terminated strings, indexed by the offset
  // into the table. This is used for type references and function names.
  kStrings = 3,

  // The attributes section contains the attributes referenced by the program.
  kAttributes = 4,

  // The kernels section defines a dense numbering for kernels.  It is a
  // count of the number of kernels present, followed by a list of indices
  // into the string table.
  kKernels = 5,

  // The types section defines a dense numbering for types.  It is the count of
  // types present, followed by a list of indices into the string table.
  kTypes = 6,

  // The functions section contains the bodies of executable code fragments.
  kFunctions = 7,

  // The function index section provides a symbol table and metadata about the
  // functions in this BEFFile.
  kFunctionIndex = 8,

  // The attribute types section provides type information for each attribute in
  // attributes section. It is an optional section and will be ignored by
  // executor. It will be used for converting BEF back to mlir.
  kAttributeTypes = 9,

  // The attribute names section provides names of attributes for each kernel.
  // It is an optional section and will be ignored by executor. It will be used
  // for converting BEF back to mlir.
  kAttributeNames = 10,

  // The register types section provides type information for each register in
  // each function. It is an optional section and will be ignored by executor.
  // It will be used for converting BEF back to mlir.
  kRegisterTypes = 11,

  // kNumSectionIDs is the number of section ids in a BEF file including
  // optional sections.
  kNumSectionIDs,
};

enum : size_t {
  // Kernels in BEF are 4-byte aligned.
  kKernelEntryAlignment = 4,
};

// SpecialAttribute describes the special BEF attributes of a kernel. It is a
// bitfield, each bit of which represent one kind of such attribute.
enum class SpecialAttribute : uint8_t {
  kUnknown = 0,

  // This is the bef.nonstrict attribute, which indicates a kernel is runnable
  // when one of its operands becomes available.
  kNonStrict = 1,
};

// This enum defined the function kind.
enum class FunctionKind : uint8_t {
  // This is the normal BEF function that defines registers and kernels in BEF.
  kBEFFunction = 0,

  // This is the native function that invokes executable code directly.
  kNativeFunction = 1,
};

// Below constants defines bit positions and bit sizes for different category of
// attributes.
enum {
  kScalarAttributeTypeSize = 8,
  kArrayAttributeTypeSize = 1,
  kDenseAttributeTypeSize = 1,
  kAggregateAttributeTypeSize = 1,

  kScalarAttributeTypeShift = 0,
  kArrayAttributeTypeShift = kScalarAttributeTypeSize,
  kDenseAttributeTypeShift = kScalarAttributeTypeSize + kArrayAttributeTypeSize,
  kAggregateAttributeTypeShift = kScalarAttributeTypeSize +
                                 kArrayAttributeTypeSize +
                                 kDenseAttributeTypeSize,

  kScalarAttributeTypeMask = ((1 << kScalarAttributeTypeSize) - 1)
                             << kScalarAttributeTypeShift,
  kArrayAttributeTypeMask = ((1 << kArrayAttributeTypeSize) - 1)
                            << kArrayAttributeTypeShift,
  kDenseAttributeTypeMask = ((1 << kDenseAttributeTypeSize) - 1)
                            << kDenseAttributeTypeShift,
  kAggregateAttributeTypeMask = ((1 << kAggregateAttributeTypeSize) - 1)
                                << kAggregateAttributeTypeShift,

  kArrayAttributeType = 1 << kArrayAttributeTypeShift,
  kDenseAttributeType = 1 << kDenseAttributeTypeShift,
  kAggregateAttributeType = 1 << kAggregateAttributeTypeShift,
};

// BEFDataType defines the data types supported in BEF. Note that the enum
// values here should be kept consistent with BEFAttributeType.
//
// TODO(tfrt-devs): Consider having a single centralized definition for
// all data types supported by TFRT.
enum class BEFDataType : uint8_t {
  kUnsupported,

  kBool,

  kI1,
  kI8,
  kI16,
  kI32,
  kI64,
  kUI8,
  kUI16,
  kUI32,
  kUI64,
  kBF16,
  kF16,
  kF32,
  kF64,
  kComplex64,
  kComplex128,
  kString,

  // TODO(tfrt-devs): Consider adding support for resource, variant
  // and quantized integers.

  kFirstDataType = kBool,
  kLastDataType = kString,
};

// This enum defines the attribute type.
enum class BEFAttributeType : uint16_t {
  kUnsupported,

  // Reserve entries for data types.
  kFirstDataType = static_cast<uint8_t>(BEFDataType::kFirstDataType),
  kLastDataType = static_cast<uint8_t>(BEFDataType::kLastDataType),

  kType,

  kShape,

  kFirstScalarType = kFirstDataType,
  kLastScalarType = kShape,

  kEmptyArray = static_cast<uint8_t>(BEFDataType::kI32) | kArrayAttributeType,

  kI8Array = static_cast<uint8_t>(BEFDataType::kI8) | kArrayAttributeType,
  kI32Array = static_cast<uint8_t>(BEFDataType::kI32) | kArrayAttributeType,
  kI64Array = static_cast<uint8_t>(BEFDataType::kI64) | kArrayAttributeType,
  kBF16Array = static_cast<uint8_t>(BEFDataType::kBF16) | kArrayAttributeType,
  kF16Array = static_cast<uint8_t>(BEFDataType::kF16) | kArrayAttributeType,
  kF32Array = static_cast<uint8_t>(BEFDataType::kF32) | kArrayAttributeType,
  kF64Array = static_cast<uint8_t>(BEFDataType::kF64) | kArrayAttributeType,

  kTypeArray = kType | kArrayAttributeType,

  kI8Dense = static_cast<uint8_t>(BEFDataType::kI8) | kDenseAttributeType,
  kI32Dense = static_cast<uint8_t>(BEFDataType::kI32) | kDenseAttributeType,
  kI64Dense = static_cast<uint8_t>(BEFDataType::kI64) | kDenseAttributeType,
  kBF16Dense = static_cast<uint8_t>(BEFDataType::kBF16) | kDenseAttributeType,
  kF16Dense = static_cast<uint8_t>(BEFDataType::kF16) | kDenseAttributeType,
  kF32Dense = static_cast<uint8_t>(BEFDataType::kF32) | kDenseAttributeType,
  kF64Dense = static_cast<uint8_t>(BEFDataType::kF64) | kDenseAttributeType,

  kAggregate = kAggregateAttributeType,
};
static_assert(static_cast<uint16_t>(BEFAttributeType::kLastScalarType) <=
                  kScalarAttributeTypeMask,
              "Scalar attributes can only use one byte.");

inline bool IsArrayAttribute(BEFAttributeType type) {
  return static_cast<uint16_t>(type) & kArrayAttributeType;
}

inline bool IsDenseAttribute(BEFAttributeType type) {
  return static_cast<uint16_t>(type) & kDenseAttributeType;
}

inline bool IsScalarAttribute(BEFAttributeType type) {
  return type >= BEFAttributeType::kFirstScalarType &&
         type <= BEFAttributeType::kLastScalarType;
}

inline bool IsDataTypeAttribute(BEFAttributeType type) {
  return type >= BEFAttributeType::kFirstDataType &&
         type <= BEFAttributeType::kLastDataType;
}

inline BEFAttributeType GetDenseAttributeType(BEFDataType element_type) {
  return static_cast<BEFAttributeType>(static_cast<uint16_t>(element_type) |
                                       kDenseAttributeType);
}

inline BEFAttributeType GetElementAttributeType(BEFAttributeType type) {
  auto r = static_cast<BEFAttributeType>(static_cast<uint16_t>(type) &
                                         kScalarAttributeTypeMask);
  return r;
}

inline BEFDataType GetDataType(BEFAttributeType type) {
  auto r = GetElementAttributeType(type);
  assert(IsDataTypeAttribute(r));
  return static_cast<BEFDataType>(r);
}

inline bool IsFixedAttribute(BEFAttributeType type) {
  return (type == BEFAttributeType::kType) ||
         (IsDataTypeAttribute(type) &&
          GetDataType(type) != BEFDataType::kString);
}

inline BEFAttributeType GetArrayAttributeType(BEFAttributeType element_type) {
  assert(IsFixedAttribute(element_type));
  return static_cast<BEFAttributeType>(static_cast<uint16_t>(element_type) |
                                       kArrayAttributeType);
}

// Belows are helper functions for retrieving BEFAttributeType for scalar types.
template <typename T>
BEFAttributeType GetBEFAttributeType();

template <>
inline BEFAttributeType GetBEFAttributeType<int8_t>() {
  return static_cast<BEFAttributeType>(BEFDataType::kI8);
}

template <>
inline BEFAttributeType GetBEFAttributeType<int16_t>() {
  return static_cast<BEFAttributeType>(BEFDataType::kI16);
}

template <>
inline BEFAttributeType GetBEFAttributeType<int32_t>() {
  return static_cast<BEFAttributeType>(BEFDataType::kI32);
}

template <>
inline BEFAttributeType GetBEFAttributeType<int64_t>() {
  return static_cast<BEFAttributeType>(BEFDataType::kI64);
}

template <>
inline BEFAttributeType GetBEFAttributeType<uint8_t>() {
  return static_cast<BEFAttributeType>(BEFDataType::kUI8);
}

template <>
inline BEFAttributeType GetBEFAttributeType<uint16_t>() {
  return static_cast<BEFAttributeType>(BEFDataType::kUI16);
}

template <>
inline BEFAttributeType GetBEFAttributeType<uint32_t>() {
  return static_cast<BEFAttributeType>(BEFDataType::kUI32);
}

template <>
inline BEFAttributeType GetBEFAttributeType<uint64_t>() {
  return static_cast<BEFAttributeType>(BEFDataType::kUI64);
}

template <>
inline BEFAttributeType GetBEFAttributeType<float>() {
  return static_cast<BEFAttributeType>(BEFDataType::kF32);
}

template <>
inline BEFAttributeType GetBEFAttributeType<double>() {
  return static_cast<BEFAttributeType>(BEFDataType::kF64);
}

template <>
inline BEFAttributeType GetBEFAttributeType<BEFDataType>() {
  return BEFAttributeType::kType;
}

// Given a pointer to the start of an array of BEF attributes, decode the
// size information that is stored in reverse order immediately preceding it.
static inline size_t DecodeArraySizeFromBEFAttributes(const void* data) {
  const uint8_t* len_ptr = static_cast<const uint8_t*>(data) - 1;
  size_t size = 0;
  bool more_bytes;
  do {
    more_bytes = (len_ptr[0] & 0x80) != 0;
    size <<= 7;
    size |= len_ptr[0] & 0x7F;
    --len_ptr;
  } while (more_bytes);
  return size;
}

template <typename T>
static inline ArrayRef<T> DecodeArrayFromBEFAttributes(const void* ptr) {
  const uint8_t* data = reinterpret_cast<const uint8_t*>(ptr);
  size_t size = DecodeArraySizeFromBEFAttributes(data);
  return ArrayRef<T>(reinterpret_cast<const T*>(data), size);
}

// Return the number of bytes preceding the data pointer that correspond to the
// BEF array size.  This is the size of the size of the array - the number of
// bytes occupied by the VBR encoded array size.
static inline size_t GetBEFArraySizeSize(const void* data) {
  const uint8_t* len_ptr = static_cast<const uint8_t*>(data) - 1;
  size_t size = 1;
  while (true) {
    // Scan backwards until we find a byte with the high bit clear.
    if ((len_ptr[0] & 0x80) == 0) return size;
    ++size;
    --len_ptr;
  }
}

// Emit the specified array length into the indicated vector, which should be
// a SmallVector<uint8_t>, vector<uint8_t> or equivalent.
template <typename VectorType>
static inline void EmitBEFArrayLength(size_t value, VectorType* byte_vector) {
  byte_vector->push_back(uint8_t(value & 0x7F));
  value >>= 7;
  while (value != 0) {
    byte_vector->push_back(uint8_t((value & 0x7F) | 0x80));
    value >>= 7;
  }
}

struct BEFAttrBase {
  BEFAttributeType type;
  // The byte count for the entire content including `BEFAttrBase` and necessary
  // alignment paddings.
  uint16_t byte_count;
};

struct BEFFixed8Attr {
  BEFAttrBase base;
  uint8_t data;
};

struct BEFFixed16Attr {
  BEFAttrBase base;
  uint16_t data;
};

struct BEFFixed32Attr {
  BEFAttrBase base;
  uint32_t data;
};

struct BEFFixed64Attr {
  BEFAttrBase base;
  uint8_t paddings[4];
  uint64_t data;
};

struct BEFStringAttr {
  BEFAttrBase base;
  // `data` is the start of the string.
  uint8_t data[1];
};

struct BEFArrayAttr {
  BEFAttrBase base;
  uint16_t num_elements;
  // `offset` is byte offset from &base for the elements.
  uint16_t element_offset;
};

// Shape attributes in TFRT must be ranked.
struct BEFShapeAttr {
  BEFAttrBase base;
  uint16_t rank;
  uint8_t paddings[2];
  int64_t dims[1];
};

struct BEFDenseAttr {
  BEFAttrBase base;
  uint16_t rank;
  uint16_t num_elements;
  // `shape_offset` is the offset from &base for the shape dimensions. It is
  // aligned to 8-byte as dimensions are always signed 64bit integers in TFRT.
  uint16_t shape_offset;
  // `element_offset` is the byte offset from &base for the elements. It should
  // be sufficiently aligned according to data type, though it cannot be more
  // than 8-byte aligned.
  uint16_t element_offset;
};

struct BEFAggregateAttr {
  BEFAttrBase base;
  uint16_t num_elements;
  // `offsets` is the start of `num_elements` 16bit integers offsets, which are
  // immediately followed by the corresponding elements. These elements are also
  // typed (ie. start with BEFAttrBase).
  uint16_t offsets[1];
};

inline uint16_t AssertAttrFieldSize(size_t size) {
  assert(size <= ((1ul << 16) - 1));
  return static_cast<uint16_t>(size);
}

// AttributeTag is used in optional attribute types section to indicate the
// encoding (currently typed or untyped) of this attribute.
struct AttributeTag {
  AttributeTag() = default;
  AttributeTag(BEFAttributeType attribute_type, bool typed) {
    data = static_cast<size_t>(attribute_type) << 1;
    if (typed) data = data | 1;
  }
  bool IsTyped() const { return data & 1; }
  BEFAttributeType GetAttributeType() const {
    return static_cast<BEFAttributeType>(data >> 1);
  }

  size_t data = 0;
};

}  // namespace tfrt

#endif  // TFRT_SUPPORT_BEF_ENCODING_H_
