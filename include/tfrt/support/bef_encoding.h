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

enum : uint8_t {
  kScalarAttributeTypeSize = 7,
  kArrayAttributeTypeSize = 1,

  kScalarAttributeTypeShift = 0,
  kArrayAttributeTypeShift = kScalarAttributeTypeSize,

  kScalarAttributeTypeMask = ((1 << kScalarAttributeTypeSize) - 1)
                             << kScalarAttributeTypeShift,
  kArrayAttributeTypeMask = ((1 << kArrayAttributeTypeSize) - 1)
                            << kArrayAttributeTypeShift,

  kArrayAttributeType = 1 << kArrayAttributeTypeShift,
};

// This enum defines the attribute type.
enum class BEFAttributeType : uint8_t {
  kUnsupported,

  kBool,

  kI1,
  kI32,
  kI64,
  kF16,
  kF32,
  kF64,

  kType,

  kFirstFixedType = kBool,
  kLastFixedType = kType,

  kString,
  kDenseElements,

  kFirstScalarType = kBool,
  kLastScalarType = kDenseElements,

  kAggregate = 0x7f,

  kEmptyArray = kI32 | kArrayAttributeType,
  kI1Array = kI1 | kArrayAttributeType,
  kI32Array = kI32 | kArrayAttributeType,
  kI64Array = kI64 | kArrayAttributeType,
  kF16Array = kF16 | kArrayAttributeType,
  kF32Array = kF32 | kArrayAttributeType,
  kF64Array = kF64 | kArrayAttributeType,

  kTypeArray = kType | kArrayAttributeType,
};

inline bool IsArrayAttribute(BEFAttributeType type) {
  return static_cast<uint8_t>(type) & kArrayAttributeType;
}

inline bool IsScalarAttribute(BEFAttributeType type) {
  return type >= BEFAttributeType::kFirstScalarType &&
         type <= BEFAttributeType::kLastScalarType;
}

inline bool IsFixedAttribute(BEFAttributeType type) {
  return type >= BEFAttributeType::kFirstFixedType &&
         type <= BEFAttributeType::kLastFixedType;
}

inline BEFAttributeType GetArrayAttributeType(BEFAttributeType element_type) {
  assert(IsFixedAttribute(element_type));
  return static_cast<BEFAttributeType>(static_cast<uint8_t>(element_type) |
                                       kArrayAttributeType);
}

inline BEFAttributeType GetArrayAttributeElementType(
    BEFAttributeType array_type) {
  auto r = static_cast<BEFAttributeType>(static_cast<uint8_t>(array_type) &
                                         kScalarAttributeTypeMask);
  assert(IsFixedAttribute(r));
  return r;
}

// Return the byte size of attributes in BEF. It will return 0 if the size is
// not fixed.
inline size_t GetBEFAttributeSize(BEFAttributeType type) {
  switch (type) {
    case BEFAttributeType::kType:
    case BEFAttributeType::kBool:
    case BEFAttributeType::kI1:
      return 1;
    case BEFAttributeType::kI32:
    case BEFAttributeType::kF32:
      return 4;
    case BEFAttributeType::kI64:
    case BEFAttributeType::kF64:
      return 8;
    default:
      return 0;
  }
}

// Belows are helper functions for retrieving BEFAttributeType for scalar types.
template <typename T>
BEFAttributeType GetBEFAttributeType();

template <>
inline BEFAttributeType GetBEFAttributeType<uint8_t>() {
  return BEFAttributeType::kI1;
}

template <>
inline BEFAttributeType GetBEFAttributeType<int32_t>() {
  return BEFAttributeType::kI32;
}

template <>
inline BEFAttributeType GetBEFAttributeType<int64_t>() {
  return BEFAttributeType::kI64;
}

template <>
inline BEFAttributeType GetBEFAttributeType<float>() {
  return BEFAttributeType::kF32;
}

template <>
inline BEFAttributeType GetBEFAttributeType<double>() {
  return BEFAttributeType::kF64;
}

// AttributeDescriptor describes the attribute location and type information. It
// is currently only used in AggregateAttribute.
struct AttributeDescriptor {
  // `offset` is the offset to the attribute in BEF Attributes section.
  uint32_t offset;
  BEFAttributeType type;
  uint8_t padding[3];
};
static_assert(sizeof(AttributeDescriptor) == 8,
              "AttributeDescriptor should have a size of 8 bytes.");
static_assert(alignof(AttributeDescriptor) == 4,
              "AttributeDescriptor should be 4-byte aligned.");

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

struct BEFDenseAttrHeader {
  uint64_t rank : 56;
  uint64_t dtype : 8;
  uint64_t size;
};
static_assert(sizeof(BEFDenseAttrHeader) == 16,
              "unexpected size of DenseAttrHeader");

struct BEFDenseAttr {
  BEFDenseAttrHeader header;

  // payload indicates the start of shape and elements. It is 8-byte aligned.
  uint64_t payload[1];
};

// DenseAttr is used to represent dense elements (tensor) attribute for handling
// op and kernel attributes. Attribute types are first-class in host runtime but
// DenseHostTensor isn't, so we need some first-class c++ class to represent a
// dense elements attribute type rather using DenseHostTensor directly to avoid
// introducing a dependency from host_runtime lib on tensor lib. To avoid
// unnecessary copying, DenseAttr can either own or not own the underlying data.
// If a BEF kernel has a DenseElementsAttribute, the underlying bytes are in BEF
// file and thus DenseAttr doesn't need to own the bytes, and only points to the
// BEF attribute bytes.
class DenseAttr {
 public:
  static size_t Alignment() { return alignof(uint64_t); }

  DenseAttr() : bef_dense_attr_(nullptr) {}
  explicit DenseAttr(const void* data)
      : bef_dense_attr_(static_cast<const BEFDenseAttr*>(data)) {}
  ~DenseAttr() = default;

  DenseAttr(const DenseAttr&) = default;
  DenseAttr& operator=(const DenseAttr&) = default;

  explicit operator bool() const { return bef_dense_attr_ != nullptr; }

  BEFAttributeType dtype() const {
    return static_cast<BEFAttributeType>(header().dtype);
  }

  ArrayRef<int64_t> shape() const {
    assert(bef_dense_attr_ != nullptr);
    return llvm::makeArrayRef(
        reinterpret_cast<const int64_t*>(bef_dense_attr_->payload),
        header().rank);
  }

  size_t size() const { return header().size; }

  const void* data() const { return static_cast<const void*>(bef_dense_attr_); }

  const void* elements() const {
    assert(bef_dense_attr_ != nullptr);
    return static_cast<const void*>(bef_dense_attr_->payload + header().rank);
  }

  size_t SizeInBytes() {
    return sizeof(BEFDenseAttrHeader) + header().rank * sizeof(uint64_t) +
           DataSizeInBytes();
  }

  size_t DataSizeInBytes() const {
    auto element_size = GetBEFAttributeSize(dtype());
    assert(element_size != 0);
    return size() * element_size;
  }

 private:
  const BEFDenseAttrHeader& header() const {
    assert(bef_dense_attr_ != nullptr);
    return bef_dense_attr_->header;
  }

  const BEFDenseAttr* bef_dense_attr_;
};

}  // namespace tfrt

#endif  // TFRT_SUPPORT_BEF_ENCODING_H_
