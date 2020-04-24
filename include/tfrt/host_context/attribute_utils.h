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

//===- attribute_utils.h - Helpers for BEF Attributes -----------*- C++ -*-===//
//
// This file declares helper routines for reading BEF Attributes.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_ATTRIBUTE_UTILS_H_
#define TFRT_HOST_CONTEXT_ATTRIBUTE_UTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/byte_order.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class StringAttribute;
template <typename T>
class FlatArrayAttribute;
class OffsetArrayAttribute;

namespace internal {

template <typename>
struct is_flat_array_attribute : std::false_type {};
template <typename T>
struct is_flat_array_attribute<FlatArrayAttribute<T>> : std::true_type {};

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
  static_assert(!std::is_same<T, OffsetArrayAttribute>(),
                "Use OffsetArrayAttribute directly instead of "
                "Attribute<OffsetArrayAttribute>");
  static_assert(!internal::is_flat_array_attribute<T>(),
                "Use FlatArrayAttribute directly instead of "
                "Attribute<FlatArrayAttribute<T>>");

  const T& value_;
};

// Like Attribute, but specifically for strings. We use this instead of
// Attribute<std::string> because strings are stored as character arrays and we
// don't want unnecessary deep copies.
//
// StringAttribute is equivalent to FlatArrayAttribute<char>, but
// StringAttribute provides a string_view, while FlatArrayAttribute<char>
// provides an ArrayRef<char>.
class StringAttribute {
 public:
  explicit StringAttribute(const void* value) {
    ASSERT_LITTLE_ENDIAN();
    auto char_array = DecodeArrayFromBEFAttributes<char>(value);
    value_ = string_view(char_array.data(), char_array.size());
  }

  string_view get() const { return value_; }
  operator string_view() const { return value_; }
  std::string str() const { return std::string(value_); }

 private:
  string_view value_;
};

// Kernels should use this so we know it has an attribute flat array.
template <typename T>
class FlatArrayAttribute {
 public:
  explicit FlatArrayAttribute(const void* data)
      : data_(DecodeArrayFromBEFAttributes<T>(data)) {
    ASSERT_LITTLE_ENDIAN();
  }

  ArrayRef<T> data() const { return data_; }
  size_t size() const { return data_.size(); }
  const T& operator[](size_t i) const { return data_[i]; }

 private:
  ArrayRef<T> data_;
};

// OffsetArrayAttribute is an array of pointers/offsets to its elements. Kernels
// should use this so we know they have an attribute array input.
class OffsetArrayAttribute {
 public:
  OffsetArrayAttribute(ArrayRef<uint8_t> attribute_section, const void* value)
      : attribute_section_(attribute_section),
        element_descriptors_(
            DecodeArrayFromBEFAttributes<AttributeDescriptor>(value)) {
    ASSERT_LITTLE_ENDIAN();
  }

  template <typename T>
  Attribute<T> GetAttribute(int index) const {
    const auto& descriptor = GetDescriptor(index);
    assert(descriptor.second.kind == GetAttributeKind<T>());
    return Attribute<T>(descriptor.first);
  }

  StringAttribute GetStringAttribute(int index) const {
    const auto& descriptor = GetDescriptor(index);
    assert(descriptor.second.kind == AttributeKind::kString);
    return StringAttribute(descriptor.first);
  }

  template <typename T>
  FlatArrayAttribute<T> GetFlatArrayAttribute(int index) const {
    const auto& descriptor = GetDescriptor(index);
    assert(descriptor.second.kind == AttributeKind::kFlatArray);
    assert(descriptor.second.array_element_kind == GetAttributeKind<T>());
    return FlatArrayAttribute<T>(descriptor.first);
  }

  OffsetArrayAttribute GetOffsetArrayAttribute(int index) const {
    const auto& descriptor = GetDescriptor(index);
    assert(descriptor.second.kind == AttributeKind::kOffsetArray);
    return OffsetArrayAttribute(attribute_section_, descriptor.first);
  }

  std::pair<const void*, AttributeKindDescriptor> GetRawAttribute(
      int index) const {
    return GetDescriptor(index);
  }

  size_t size() const { return element_descriptors_.size(); }

 private:
  std::pair<const void*, AttributeKindDescriptor> GetDescriptor(
      int index) const {
    assert(index < element_descriptors_.size());
    const auto& descriptor = element_descriptors_[index];
    assert(descriptor.offset < attribute_section_.size());
    return {static_cast<const void*>(&attribute_section_[descriptor.offset]),
            descriptor.kind_descriptor};
  }

  ArrayRef<uint8_t> attribute_section_;
  ArrayRef<AttributeDescriptor> element_descriptors_;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_ATTRIBUTE_UTILS_H_
