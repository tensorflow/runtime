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

  // Maximum attribute alignment.
  // TODO(b/184278791): apply 64byte alignment to DenseAttr raw data for eigen
  // library.
  kAttributeMaxAlignment = 8,
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

  kArray = kArrayAttributeType,
  kEmptyArray = static_cast<uint8_t>(DType::I32) | kArrayAttributeType,

  kI8Array = static_cast<uint8_t>(DType::I8) | kArrayAttributeType,
  kI32Array = static_cast<uint8_t>(DType::I32) | kArrayAttributeType,
  kI64Array = static_cast<uint8_t>(DType::I64) | kArrayAttributeType,
  kBF16Array = static_cast<uint8_t>(DType::BF16) | kArrayAttributeType,
  kF16Array = static_cast<uint8_t>(DType::F16) | kArrayAttributeType,
  kF32Array = static_cast<uint8_t>(DType::F32) | kArrayAttributeType,
  kF64Array = static_cast<uint8_t>(DType::F64) | kArrayAttributeType,

  kTypeArray = kType | kArrayAttributeType,

  kDense = kDenseAttributeType,

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

inline size_t GetAttributeDataTypeByteSize(BEFAttributeType type) {
  assert(IsArrayAttribute(type) || IsDenseAttribute(type));
  auto data_type = GetElementAttributeType(type);
  if (data_type == BEFAttributeType::kType) return 1;
  assert(IsDataTypeAttribute(data_type));
  return GetDTypeByteSize(static_cast<DType::Kind>(data_type));
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

// Return the expected length of VBR integer.
//   E.g., 1 when 0   <= value < 128
//         2 when 128 <= value < 16384
inline size_t GetSizeOfVbrInt(size_t value) {
  return (value < 0x80) ? 1 : GetSizeOfVbrInt(value >> 7) + 1;
}

}  // namespace tfrt

#endif  // TFRT_SUPPORT_BEF_ENCODING_H_
