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

// This file declares constants used when interfacing with the "Binary Executor
// Format" (BEF) files.

#ifndef TFRT_SUPPORT_BEF_ENCODING_H_
#define TFRT_SUPPORT_BEF_ENCODING_H_

#include <cstddef>
#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
#include "tfrt/dtype/dtype.h"
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
  // This is a list of filenames used for location information.  This is kept
  // in a separate section from other strings because we don't expect it to be
  // frequently accessed.
  kLocationFilenames = 0,

  // This is a list of positions referred to by location information.  Each
  // position is a triple of FilenameIndex, line, column, and offsets into this
  // section are used by location references.
  kLocationPositions = 1,

  // The strings section contains NUL terminated strings, indexed by the offset
  // into the table. This is used for type references and function names.
  kStrings = 2,

  // The attributes section contains the attributes referenced by the program.
  kAttributes = 3,

  // The kernels section defines a dense numbering for kernels.  It is a
  // count of the number of kernels present, followed by a list of indices
  // into the string table.
  kKernels = 4,

  // The types section defines a dense numbering for types.  It is the count of
  // types present, followed by a list of indices into the string table.
  kTypes = 5,

  // The function index section provides a symbol table and metadata about the
  // functions in this BEFFile.
  kFunctionIndex = 6,

  // The functions section contains the bodies of executable code fragments.
  kFunctions = 7,

  // The attribute types section provides type information for each attribute in
  // attributes section. It is an optional section and will be ignored by
  // executor. It will be used for converting BEF back to mlir.
  kAttributeTypes = 8,

  // The attribute names section provides names of attributes for each kernel.
  // It is an optional section and will be ignored by executor. It will be used
  // for converting BEF back to mlir.
  kAttributeNames = 9,

  // The register types section provides type information for each register in
  // each function. It is an optional section and will be ignored by executor.
  // It will be used for converting BEF back to mlir.
  kRegisterTypes = 10,

  // The debug info section contains extra metadata for tracing and debugging
  // purposes only.
  kDebugInfo = 11,

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

  // This attribute indicates whether a kernel has a debug info available.
  kHasDebugInfo = 2,
};

// This enum defined the function kind.
enum class FunctionKind : uint8_t {
  // This is the async BEF function that defines registers and kernels in BEF.
  // TODO(jingdong): Rename kBEFFunction to kAsyncBEFFunction after the code for
  // SyncBEFFunction stabilizes.
  kBEFFunction = 0,

  // This is the native function that invokes executable code directly.
  kNativeFunction = 1,

  // This is the sync BEF function that defines registers and kernels in BEF.
  kSyncBEFFunction = 2,
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

// This enum defines the attribute type.
enum class BEFAttributeType : uint16_t {
  kUnsupported = 0,

  // Reserve entries for data types.
  kFirstDataType = static_cast<uint8_t>(DType::FirstDType),
  kLastDataType = static_cast<uint8_t>(DType::LastDType),

  kType,

  kShape,

  kSymbolRef,

  kFirstScalarType = kFirstDataType,
  kLastScalarType = kShape,

  kEmptyArray = static_cast<uint8_t>(DType::I32) | kArrayAttributeType,

  kI8Array = static_cast<uint8_t>(DType::I8) | kArrayAttributeType,
  kI32Array = static_cast<uint8_t>(DType::I32) | kArrayAttributeType,
  kI64Array = static_cast<uint8_t>(DType::I64) | kArrayAttributeType,
  kBF16Array = static_cast<uint8_t>(DType::BF16) | kArrayAttributeType,
  kF16Array = static_cast<uint8_t>(DType::F16) | kArrayAttributeType,
  kF32Array = static_cast<uint8_t>(DType::F32) | kArrayAttributeType,
  kF64Array = static_cast<uint8_t>(DType::F64) | kArrayAttributeType,

  kTypeArray = kType | kArrayAttributeType,

  kI1Dense = static_cast<uint8_t>(DType::I1) | kDenseAttributeType,
  kI8Dense = static_cast<uint8_t>(DType::I8) | kDenseAttributeType,
  kI32Dense = static_cast<uint8_t>(DType::I32) | kDenseAttributeType,
  kI64Dense = static_cast<uint8_t>(DType::I64) | kDenseAttributeType,
  kBF16Dense = static_cast<uint8_t>(DType::BF16) | kDenseAttributeType,
  kF16Dense = static_cast<uint8_t>(DType::F16) | kDenseAttributeType,
  kF32Dense = static_cast<uint8_t>(DType::F32) | kDenseAttributeType,
  kF64Dense = static_cast<uint8_t>(DType::F64) | kDenseAttributeType,
  kComplex64Dense =
      static_cast<uint8_t>(DType::Complex64) | kDenseAttributeType,
  kComplex128Dense =
      static_cast<uint8_t>(DType::Complex128) | kDenseAttributeType,

  kAggregate = kAggregateAttributeType,

  kFunc,
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
         type < BEFAttributeType::kLastDataType;
}

inline bool IsFuncAttribute(BEFAttributeType type) {
  return type == BEFAttributeType::kFunc;
}

inline bool IsSymbolRefAttribute(BEFAttributeType type) {
  return type == BEFAttributeType::kSymbolRef;
}

inline BEFAttributeType GetDenseAttributeType(DType::Kind element_type) {
  return static_cast<BEFAttributeType>(static_cast<uint16_t>(element_type) |
                                       kDenseAttributeType);
}

inline BEFAttributeType GetElementAttributeType(BEFAttributeType type) {
  auto r = static_cast<BEFAttributeType>(static_cast<uint16_t>(type) &
                                         kScalarAttributeTypeMask);
  return r;
}

inline DType::Kind GetDataType(BEFAttributeType type) {
  auto r = GetElementAttributeType(type);
  assert(IsDataTypeAttribute(r));
  return static_cast<DType::Kind>(r);
}

inline bool IsFixedAttribute(BEFAttributeType type) {
  return (type == BEFAttributeType::kType) ||
         (IsDataTypeAttribute(type) && GetDataType(type) != DType::String);
}

inline BEFAttributeType GetArrayAttributeType(BEFAttributeType element_type) {
  assert(IsFixedAttribute(element_type));
  return static_cast<BEFAttributeType>(static_cast<uint16_t>(element_type) |
                                       kArrayAttributeType);
}

// Belows are helper functions for retrieving BEFAttributeType for scalar types.
template <typename T>
BEFAttributeType GetBEFAttributeType() {
  return static_cast<BEFAttributeType>(GetDType<T>().kind());
}
template <>
inline BEFAttributeType GetBEFAttributeType<DType::Kind>() {
  return BEFAttributeType::kType;
}

// Read an integer encoded in VBR format from the given pointer.
// It returns the updated pointer after reading a VBR integer.
inline const uint8_t* ReadVbrInt(const uint8_t* ptr, size_t* out) {
  *out = 0;
  uint8_t onebyte;
  do {
    onebyte = *ptr++;
    *out <<= 7;
    *out |= onebyte & 0x7f;
  } while (onebyte & 0x80);
  return ptr;
}

// Check if the given alignment is valid. Valid alignments are 1, 2, 4, 8, ...
inline bool IsValidAlignment(unsigned alignment) {
  return llvm::isPowerOf2_32(alignment);
}

// Calculate required byte size of alignment padding for the given offset when
// there is a prefix.
//
// Examples,
//   CalculateAlignmentPaddingSize(0, 1, 4) should return 3.
//   CalculateAlignmentPaddingSize(1, 1, 4) should return 2.
//   CalculatePaddingSize(3, 2, 8) should return 3.
inline size_t CalculateAlignmentPaddingSize(size_t offset, unsigned prefix_size,
                                            unsigned alignment) {
  return llvm::offsetToAlignment(offset + prefix_size, llvm::Align(alignment));
}

// Return the expected length of VBR integer.
//   E.g., 1 when 0   <= value < 128
//         2 when 128 <= value < 16384
inline size_t GetSizeOfVbrInt(size_t value) {
  return (value < 0x80) ? 1 : GetSizeOfVbrInt(value >> 7) + 1;
}

// An array starts with its length encoded in fixed32 integer format.
template <typename T>
ArrayRef<T> DecodeArrayFromBEFAttributes(const void* ptr) {
  size_t size;
  const uint8_t* data = ReadVbrInt(static_cast<const uint8_t*>(ptr), &size);
  return ArrayRef<T>(reinterpret_cast<const T*>(data), size);
}

// Return the number of bytes preceding the data pointer that correspond to the
// BEF array size.  This is the size of the size of the array - the number of
// bytes occupied by the VBR encoded array size.
inline size_t GetBEFArraySizeSize(const void* data) {
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
void EmitBEFArrayLength(size_t value, VectorType* byte_vector) {
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
  uint16_t byte_count_high;
  uint32_t byte_count_low;
};
static_assert(sizeof(BEFAttrBase) == 8, "Unexpected size of BEFAttrBase");
static_assert(alignof(BEFAttrBase) == 4, "Unexpected alignment of BEFAttrBase");
static_assert(std::is_standard_layout<BEFAttrBase>::value,
              "BEFAttrBase must have standard layout");

inline uint64_t GetBEFAttrByteCount(const BEFAttrBase& base) {
  return (static_cast<uint64_t>(base.byte_count_high) << 32) |
         base.byte_count_low;
}

inline void SetBEFAttrByteCount(uint64_t byte_count, BEFAttrBase* base) {
  assert((byte_count >> 48) == 0);
  base->byte_count_high = static_cast<uint16_t>(byte_count >> 32);
  base->byte_count_low = static_cast<uint32_t>(byte_count & ((1ull << 32) - 1));
}

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
  uint64_t data;
};

struct BEFStringAttr {
  BEFAttrBase base;
  // `data` is the start of the string.
  uint8_t data[1];
};

struct BEFArrayAttr {
  BEFAttrBase base;
  uint32_t num_elements;
  // `offset` is byte offset from &base for the elements.
  uint32_t element_offset;
};

enum class BEFShapeType : uint8_t {
  kUnranked = 0,
  kRanked,
};

struct BEFShapeAttr {
  BEFAttrBase base;
  BEFShapeType shape_type;
  uint8_t padding;
  uint16_t rank;
};

// Shape attributes in TFRT must be ranked.
// TODO(tfrt-dev): Consider a better binary representation for ranked shape
// attributes. Currently shapes with 0-rank are not emitted as
// BEFRankedShapeAttr as it has at least one trailing integer for dimensions.
struct BEFRankedShapeAttr {
  BEFShapeAttr shape_base;
  uint8_t paddings[4];
  int64_t dims[1];
};

struct BEFDenseAttr {
  BEFAttrBase base;
  uint16_t rank;
  // `shape_offset` is the offset from &base for the shape dimensions. It is
  // aligned to 8-byte as dimensions are always signed 64bit integers in TFRT.
  uint16_t shape_offset;
  uint32_t num_elements;
  // `element_offset` is the byte offset from &base for the elements. It should
  // be sufficiently aligned according to data type, though it cannot be more
  // than 8-byte aligned.
  uint32_t element_offset;
};

using BEFAggregateAttrOffset32_t = uint32_t;
struct BEFAggregateAttr {
  BEFAttrBase base;
  uint32_t num_elements;
  // `offsets` is the start of `num_elements` 32bit-integer offsets, which are
  // immediately followed by the corresponding elements. These elements are also
  // typed (ie. start with BEFAttrBase).
  BEFAggregateAttrOffset32_t offsets[1];
};

// TODO(b/168505010): The size error should be handled properly by callers
// instead of using assert().
inline uint16_t AssertAttrFieldSize16(size_t size) {
  assert(size <= ((1ul << 16) - 1));
  return static_cast<uint16_t>(size);
}

inline uint32_t AssertAttrFieldSize32(size_t size) {
  assert(size <= ((1ul << 32) - 1));
  return static_cast<uint32_t>(size);
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
