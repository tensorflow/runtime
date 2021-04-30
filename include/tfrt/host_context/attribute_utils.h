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

// Helpers for BEF Attributes
//
// This file declares helper routines for reading BEF Attributes.

#ifndef TFRT_HOST_CONTEXT_ATTRIBUTE_UTILS_H_
#define TFRT_HOST_CONTEXT_ATTRIBUTE_UTILS_H_

#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Alignment.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/byte_order.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class StringAttribute;
template <typename T>
class ArrayAttribute;
class AggregateAttribute;

namespace internal {

template <typename>
struct is_array_attribute : std::false_type {};
template <typename T>
struct is_array_attribute<ArrayAttribute<T>> : std::true_type {};

}  // namespace internal

// Kernels should use this so we know they have an attribute input.
template <typename T>
class Attribute {
 public:
  explicit Attribute(const void* value)
      : value_(*reinterpret_cast<const T*>(value)) {
    ASSERT_LITTLE_ENDIAN();
  }

  const T& get() const { return value_; }
  const T* operator->() const { return &value_; }
  const T& operator*() const { return value_; }

 private:
  static_assert(!std::is_same<T, std::string>(),
                "Use StringAttribute instead of Attribute<std::string>");
  static_assert(
      !std::is_same<T, StringAttribute>(),
      "Use StringAttribute directly instead of Attribute<StringAttribute>");
  static_assert(!std::is_same<T, AggregateAttribute>(),
                "Use AggregateAttribute directly instead of "
                "Attribute<AggregateAttribute>");
  static_assert(!internal::is_array_attribute<T>(),
                "Use ArrayAttribute directly instead of "
                "Attribute<ArrayAttribute<T>>");

  const T& value_;
};

// Kernels should use this so we know it has an array attribute.
template <typename T>
class ArrayAttribute {
 public:
  explicit ArrayAttribute(const void* data) {
    if (data) {
      size_t element_count;
      auto ptr =
          ReadVbrInt(reinterpret_cast<const uint8_t*>(data), &element_count);
      data_ =
          llvm::makeArrayRef(reinterpret_cast<const T*>(ptr), element_count);
    }
  }

  ArrayRef<T> data() const { return data_; }
  size_t size() const { return data_.size(); }
  const T& operator[](size_t i) const { return data_[i]; }

 private:
  ArrayRef<T> data_;
};

// Like Attribute, but specifically for strings. We use this instead of
// Attribute<std::string> because strings are stored as character arrays and we
// don't want unnecessary deep copies.
//
// StringAttribute is equivalent to ArrayAttribute<char>, but
// StringAttribute provides a string_view, while ArrayAttribute<char>
// provides an ArrayRef<char>.
class StringAttribute {
 public:
  StringAttribute() = default;

  explicit StringAttribute(const void* value) {
    ASSERT_LITTLE_ENDIAN();
    if (value) {
      size_t string_length;
      auto ptr =
          ReadVbrInt(reinterpret_cast<const uint8_t*>(value), &string_length);
      value_ = string_view(reinterpret_cast<const char*>(ptr), string_length);
    }
  }

  string_view get() const { return value_; }
  operator string_view() const { return value_; }
  std::string str() const { return std::string(value_); }

 protected:
  string_view value_;
};

// Compilation unit attribute decodes serialized MLIR module and a compilation
// target symbol (function name).
class CompilationUnitAttribute {
 public:
  explicit CompilationUnitAttribute(const void* value) {
    ASSERT_LITTLE_ENDIAN();
    const auto* ptr = static_cast<const uint8_t*>(value);

    size_t root_symbol_len;
    ptr = ReadVbrInt(ptr, &root_symbol_len);

    size_t num_nested_symbols;
    ptr = ReadVbrInt(ptr, &num_nested_symbols);

    llvm::SmallVector<size_t, 4> nested_symbols_len(num_nested_symbols);
    for (int i = 0; i < num_nested_symbols; ++i) {
      ptr = ReadVbrInt(ptr, &nested_symbols_len[i]);
    }

    size_t serialized_operation_len;
    ptr = ReadVbrInt(ptr, &serialized_operation_len);

    // The base of the attribute payload.
    const char* base = reinterpret_cast<const char*>(ptr);
    root_symbol_ = {base, root_symbol_len};
    size_t offset = root_symbol_len;

    for (int i = 0; i < num_nested_symbols; ++i) {
      size_t len = nested_symbols_len[i];
      nested_symbols_.emplace_back(base + offset, len);
      offset += len;
    }

    serialized_operation_ = {base + offset, serialized_operation_len};
  }

  string_view root_symbol() const { return root_symbol_; }
  ArrayRef<string_view> nested_symbols() const { return nested_symbols_; }
  string_view serialized_operation() const { return serialized_operation_; }

 private:
  string_view root_symbol_;
  llvm::SmallVector<string_view, 4> nested_symbols_;
  string_view serialized_operation_;
};

// FunctionAttribute holds the function name. Can be extended in the future.
struct FunctionAttribute {
  string_view func_name;
};

// TypedAttrBase is the base class for all typed attributes below. It provides
// llvm style cast (isa, cast, dyn_cast, etc) for efficient down-casting to
// subclasses.
class TypedAttrBase {
 public:
  TypedAttrBase() : data_(nullptr) {}

  TypedAttrBase(BEFAttributeType type, const void* data)
      : type_(type), data_(static_cast<const uint8_t*>(data)) {}

  BEFAttributeType type() const {
    if (type_ != BEFAttributeType::kArray) return type_;
    size_t embedded_type;
    ReadVbrInt(data_ - 2, &embedded_type);
    return static_cast<BEFAttributeType>(embedded_type);
  }

  const void* data() const { return data_; }

  template <typename T>
  bool isa() const {
    return T::classof(*this);
  }

  template <typename T>
  T dyn_cast() const {
    return isa<T>() ? T(type_, data_) : T();
  }

  template <typename T>
  T cast() const {
    assert(isa<T>());
    return T(type_, data_);
  }

  explicit operator bool() const { return data_ != nullptr; }

 protected:
  BEFAttributeType type_;
  const uint8_t* data_ = nullptr;
};

namespace internal {

// An intermediate class template for all fixed-width attributes. It provides
// the common GetValue() method for all fixed-width attributes.
template <DType::Kind DataTypeEnum, typename DataType>
class DataTypeAttrBase : public TypedAttrBase {
 public:
  using TypedAttrBase::TypedAttrBase;
  explicit DataTypeAttrBase(const void* data)
      : TypedAttrBase(static_cast<BEFAttributeType>(DataTypeEnum), data) {}

  DataType GetValue() const {
    return *reinterpret_cast<const DataType*>(data_);
  }

  size_t GetByteSize() const { return sizeof(DataType); }

  static bool classof(TypedAttrBase base) {
    const auto attr_type = base.type();
    return IsDataTypeAttribute(attr_type) &&
           GetDataType(attr_type) == DataTypeEnum;
  }
};

}  // namespace internal

using UI8Attr = internal::DataTypeAttrBase<DType::UI8, uint8_t>;
using UI16Attr = internal::DataTypeAttrBase<DType::UI16, uint16_t>;
using UI32Attr = internal::DataTypeAttrBase<DType::UI32, uint32_t>;
using UI64Attr = internal::DataTypeAttrBase<DType::UI64, uint64_t>;
using I8Attr = internal::DataTypeAttrBase<DType::I8, uint8_t>;
using I16Attr = internal::DataTypeAttrBase<DType::I16, int16_t>;
using I32Attr = internal::DataTypeAttrBase<DType::I32, int32_t>;
using I64Attr = internal::DataTypeAttrBase<DType::I64, int64_t>;
using F32Attr = internal::DataTypeAttrBase<DType::F32, float>;
using F64Attr = internal::DataTypeAttrBase<DType::F64, double>;
using BF16Attr = internal::DataTypeAttrBase<DType::BF16, int16_t>;
using F16Attr = internal::DataTypeAttrBase<DType::F16, int16_t>;

class I1Attr : public internal::DataTypeAttrBase<DType::I1, uint8_t> {
 public:
  using DataTypeAttrBase::DataTypeAttrBase;

  bool GetValue() const {
    return static_cast<bool>(DataTypeAttrBase::GetValue());
  }
};

class TypeAttr : public TypedAttrBase {
 public:
  using TypedAttrBase::TypedAttrBase;

  DType::Kind GetValue() const {
    return *(reinterpret_cast<const DType::Kind*>(data_));
  }

  static bool classof(TypedAttrBase base) {
    return base.type() == BEFAttributeType::kType;
  }
};

class ArrayAttr : public TypedAttrBase {
 public:
  using TypedAttrBase::TypedAttrBase;
  explicit ArrayAttr(const void* data)
      : ArrayAttr(BEFAttributeType::kArray, data) {}

  ArrayAttr(BEFAttributeType type, const void* data)
      : TypedAttrBase(type, data) {
    if (data) {
      element_base_ =
          ReadVbrInt(reinterpret_cast<const uint8_t*>(data_), &element_count_);
    }
  }

  BEFAttributeType GetElementType() const {
    return GetElementAttributeType(type_);
  }

  const void* GetElements() const { return element_base_; }

  template <typename T>
  ArrayRef<T> GetValue() const {
    // For empty arrays, we don't care the element type.
    if (GetNumElements() == 0) return {};
    return llvm::makeArrayRef(static_cast<const T*>(GetElements()),
                              GetNumElements());
  }

  size_t GetNumElements() const { return element_count_; }

  static bool classof(TypedAttrBase base) {
    return IsArrayAttribute(base.type());
  }

  size_t GetByteSize() const {
    return GetSizeOfVbrInt(static_cast<size_t>(element_count_)) +
           GetAttributeDataTypeByteSize(GetElementAttributeType(type_)) *
               element_count_;
  }

 protected:
  const void* element_base_;
  size_t element_count_;
};

class StringAttr : public TypedAttrBase, public StringAttribute {
 public:
  using TypedAttrBase::TypedAttrBase;
  explicit StringAttr(const void* data)
      : StringAttr(static_cast<BEFAttributeType>(DType::String), data) {}

  StringAttr(BEFAttributeType type, const void* data)
      : TypedAttrBase(type, data), StringAttribute(data_) {}

  string_view GetValue() const { return get(); }

  static bool classof(TypedAttrBase base) {
    return IsDataTypeAttribute(base.type()) &&
           GetDataType(base.type()) == DType::String;
  }
};

// FuncAttr holds the function names as strings. This attribute is separated
// from StringAttr so that clients (such as TensorFlow runtime fallback)
// can handle separately.
//
// Currently we ignore the attributes in a TensorFlow function op, which is
// different from current TensorFlow runtime. This is acceptable since these
// attributes are unused.
class FuncAttr : private StringAttr {
 public:
  explicit FuncAttr(const void* data)
      : StringAttr(BEFAttributeType::kFunc, data) {}

  FuncAttr(BEFAttributeType type, const void* data) : StringAttr(type, data) {}

  string_view GetFunctionName() const { return GetValue(); }

  static bool classof(TypedAttrBase base) {
    return base.type() == BEFAttributeType::kFunc;
  }
};

class ShapeAttr : public TypedAttrBase {
 public:
  using TypedAttrBase::TypedAttrBase;

  explicit ShapeAttr(const void* data)
      : ShapeAttr(BEFAttributeType::kShape, data) {}

  ShapeAttr(BEFAttributeType type, const void* data)
      : TypedAttrBase(type, data) {
    if (data) {
      shape_base_ = ReadVbrInt(reinterpret_cast<const uint8_t*>(data), &rank_);
    }
  }

  // Return the prefix size of ShapeAttr.
  //
  // In DiamondPacking, the sub-elements should be placed to have only one
  // peak alignment constraint; element size should increase then decrease.
  // Both ascending packing (element size should always increase) and
  // descending packing (element size should always decrease) are covered by
  // DiamondPacking.
  //
  // Prefix size is defined as the total bytes before the sub entry having
  // the peak alignment constraint.
  //
  //   e.g., [uint8_t, uint16_t, uint64_t, uint64_t, uint32_t]
  //
  //          peak_alignment = alignof(uint64_t) = 8
  //          prefix_size    = sizeof(uint8_t) + sizeof(uint16_t) = 3
  //
  // The DiamondPacking provides a nice property. When we need to place
  // an object (packed in DiamondPacking method) in memory for an arbitrary
  // address A, we could calculate padding size P, which satisfies the following
  // equation:
  //
  //       (A + P + prefix_size) % peak_alignment == 0
  //
  // When the object is placed at (A + P), it guarantees that
  // all the sub element alignment constraints meet as well.
  size_t GetPrefixSize() const {
    return (rank_ <= 1) ? 0 : GetSizeOfVbrInt(rank_);
  }

  // Return the peak alignment constraint of ShapeAttr.
  size_t Alignment() const { return (rank_ <= 1) ? 1 : alignof(int64_t); }

  int GetRank() const { return static_cast<int>(rank_) - 1; }

  bool HasRank() const { return rank_ > 0; }

  ArrayRef<int64_t> GetShape() const {
    const auto shape_size = (rank_) ? rank_ - 1 : 0;
    return llvm::makeArrayRef(reinterpret_cast<const int64_t*>(shape_base_),
                              shape_size);
  }

  static bool classof(TypedAttrBase base) {
    return base.type() == BEFAttributeType::kShape;
  }

  size_t GetByteSize() const {
    return (rank_ <= 1)
               ? 1
               : (rank_ - 1) * sizeof(int64_t) + GetSizeOfVbrInt(rank_);
  }

 protected:
  // To distinguish unranked shapes from zero ranked shapes,
  // rank_ is set as follows:
  //   rank_ == 0: unranked shape
  //   rank_ == 1: zero ranked shape
  //   rank_ >= 2: ranked shape ( sizeof(dimension) + 1 ).
  size_t rank_;
  const uint8_t* shape_base_;
};

class DenseAttr : public TypedAttrBase {
 public:
  using TypedAttrBase::TypedAttrBase;

  explicit DenseAttr(const void* data)
      : DenseAttr(BEFAttributeType::kDense, data) {}

  DenseAttr(BEFAttributeType type, const void* data)
      : TypedAttrBase(BEFAttributeType::kDense, data) {
    // For an invalid/uninitialized instance, data could be set to null.
    if (data) {
      auto ptr = reinterpret_cast<const uint8_t*>(data);
      dtype_ = static_cast<DType::Kind>(*ptr++);
      ptr = ReadVbrInt(ptr, &rank_);
      total_byte_size_ = *(reinterpret_cast<const uint32_t*>(ptr));
      ptr += sizeof(uint32_t);
      shape_ = llvm::makeArrayRef(reinterpret_cast<const int64_t*>(ptr), rank_);
      element_base_ = ptr + rank_ * sizeof(int64_t);
      header_size_ = reinterpret_cast<const uint8_t*>(element_base_) -
                     reinterpret_cast<const uint8_t*>(data);
    }
  }

  // Return the prefix size of DenseAttr.
  size_t GetPrefixSize() const {
    return sizeof(DType::Kind) + GetSizeOfVbrInt(rank_) + sizeof(uint32_t);
  }

  // Return the peak alignment constraint of DenseAttr.
  size_t Alignment() const {
    return std::max(alignof(int64_t), DType(dtype_).GetHostAlignment());
  }

  size_t GetHeaderSize() const { return header_size_; }

  DType::Kind dtype() const { return dtype_; }

  llvm::ArrayRef<int64_t> shape() const { return shape_; }

  size_t GetNumElements() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1,
                           std::multiplies<>());
  }

  const void* GetElements() const { return element_base_; }

  ArrayRef<char> GetRawData() const {
    return llvm::makeArrayRef(reinterpret_cast<const char*>(element_base_),
                              total_byte_size_ - header_size_);
  }

  template <typename T>
  const T& GetElement(size_t index) const {
    assert(GetDType<T>().kind() == dtype_);
    return *(reinterpret_cast<const T*>(element_base_) + index);
  }

  static bool classof(TypedAttrBase base) {
    return IsDenseAttribute(base.type());
  }

  size_t GetByteSize() const { return total_byte_size_; }

 protected:
  DType::Kind dtype_;
  size_t rank_;
  size_t total_byte_size_;
  ArrayRef<int64_t> shape_;
  const void* element_base_;
  size_t header_size_;
};

class AggregateAttr : public TypedAttrBase {
 public:
  using TypedAttrBase::TypedAttrBase;

  explicit AggregateAttr(const void* data)
      : AggregateAttr(BEFAttributeType::kAggregate, data) {}

  AggregateAttr(BEFAttributeType type, const void* data)
      : TypedAttrBase(BEFAttributeType::kAggregate, data) {
    if (data) {
      size_t element_count;
      auto ptr =
          ReadVbrInt(reinterpret_cast<const uint8_t*>(data), &element_count);
      if (element_count == 0) {
        total_byte_size_ = 1;
      } else {
        element_types_ = llvm::makeArrayRef(
            reinterpret_cast<const BEFAttributeType*>(ptr), element_count);
        ptr += sizeof(uint16_t) * element_count;
        element_offsets_ = llvm::makeArrayRef(
            reinterpret_cast<const uint32_t*>(ptr), element_count);
        ptr += sizeof(uint32_t) * element_count;
        total_byte_size_ = *(reinterpret_cast<const uint32_t*>(ptr));
      }
    }
  }

  // Return the prefix size of AggregateAttr.
  size_t GetPrefixSize() const {
    const size_t element_count = element_offsets_.size();
    return (total_byte_size_ > 1)
               ? GetSizeOfVbrInt(element_count) + sizeof(uint32_t) +
                     element_count * (sizeof(uint32_t) + sizeof(uint16_t))
               : 0;
  }

  // Return the peak alignment constraint of AggregateAttr.
  size_t Alignment() const { return kAttributeMaxAlignment; }

  size_t GetNumElements() const {
    return (total_byte_size_ > 1) ? element_offsets_.size() : 0;
  }

  // Usage example;
  //   string_view sv = agg_attr.GetElement<StringAttr>(0).GetValue();
  template <typename T>
  T GetElement(int index) const {
    assert(total_byte_size_ > 1 && index < element_offsets_.size());
    return T(data_ + element_offsets_[index]);
  }

  BEFAttributeType GetElementType(int index) const {
    assert(total_byte_size_ > 1 && index < element_offsets_.size());
    return element_types_[index];
  }

  size_t GetElementOffset(int index) const {
    assert(total_byte_size_ > 1 && index < element_offsets_.size());
    return element_offsets_[index];
  }

  TypedAttrBase GetAttribute(int index) const {
    assert(index < GetNumElements());
    return TypedAttrBase(element_types_[index],
                         data_ + element_offsets_[index]);
  }

  template <typename AttrClass>
  AttrClass GetAttributeOfType(int index) const {
    return GetElement<AttrClass>(index);
  }

  static bool classof(TypedAttrBase base) {
    return base.type() == BEFAttributeType::kAggregate;
  }

  size_t GetByteSize() const { return total_byte_size_; }

  size_t GetAlgnmentPadding(size_t offset) const {
    const size_t element_count = element_offsets_.size();
    return (total_byte_size_ > 1)
               ? llvm::offsetToAlignment(
                     offset + GetSizeOfVbrInt(element_count) +
                         sizeof(uint32_t) +
                         element_count * (sizeof(uint32_t) + sizeof(uint16_t)),
                     llvm::Align(kAttributeMaxAlignment))
               : 0;
  }

 protected:
  uint32_t total_byte_size_;
  ArrayRef<uint32_t> element_offsets_;
  ArrayRef<BEFAttributeType> element_types_;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_ATTRIBUTE_UTILS_H_
