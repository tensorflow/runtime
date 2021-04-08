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

// This file declares the OpAttrs interface.  OpAttrs is the primary class used
// by clients of the CoreRuntime API when executing ops.  It is intended to live
// on the stack, and includes significant internal storage to make op execution
// efficient in the common case.

#ifndef TFRT_CORE_RUNTIME_OP_ATTRS_H_
#define TFRT_CORE_RUNTIME_OP_ATTRS_H_

#include <cstdint>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/core_runtime/op_attr_type.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

class OpAttrsRef;
class ImmutableOpAttrs;

// This describes a single attribute entry maintained by OpAttrs. It is a 'raw'
// attribute, meaning that it is type erased and may or may not be an array.
struct OpAttrsRawEntry final {
  // Pointer to a null terminated string for the attribute name.
  const char* name;

  union {
    // Pointer to the start of the data for this element, which is the start
    // of an array of elements. IsScalarAndNotInlined() must return true.
    const void* data;

    // Buffer that stores the copy of the scalar value. IsScalarAndInlined()
    // must return true.
    char buffer[sizeof(void*)];
  };

  // The value of array_size encodes either the number of array elements, or
  // whether this raw entry contains a scalar or an object that's stored in its
  // inline buffer or not. Note that both primitive types (int, float, bool) and
  // composite types (DenseAttr, ShapeAttr and AggregateAttr) are inlinable
  // depending on its size. In particular, a small composite (e.g., empty
  // AggregateAttr) may be inlined too.
  // 1. if array_size >=0, this attribute is an array, and array_size = num of
  //    elements in array.
  // 2. if array_size == kNotInlinedScalarSentinel, then this attribute is
  //    scalar, but not stored inline.
  // 3. if array_size == kInlinedScalarSentinel, then this attribute is scalar
  //    and stored inline.
  // This is sized to 56 bits to fit nicely on 64-bit systems.
  int64_t array_size : (sizeof(void*) == 8 ? 56 : sizeof(void*) * 8);

  // This indicates the type of the entry.
  OpAttrType type : 8;

  // OpAttrs::SetRaw uses kScalarSentinel to indicate a scalar/non-array entry.
  // This is user visible, in contrast to kNotInlinedScalarSentinel and
  // kInlinedScalarSentinel, which indicate the inlining implementation detail.
  static constexpr int64_t kScalarSentinel = -1;

  // array_size is set to kNotInlinedScalarSentinel if the union is out of line
  // data. Not user-facing. Implementation detail.
  static const int64_t kNotInlinedScalarSentinel = -2;

  // array_size is set to kInlinedScalarSentinel if the union is inlined buffer.
  // Not user-facing. Implementation detail.
  static const int64_t kInlinedScalarSentinel = -3;

  // Get the actual number of elements. num_elements could be
  // OpAttrsRawEntry::kScalarSentinel.
  static int64_t GetNumElements(int64_t num_elements);

  // Decode the array size.
  static int64_t DecodeArraySize(int64_t encoded_array_size);

  // Encode the array size.
  static int64_t EncodeArraySize(int64_t num_elements, bool inlined);

  bool IsArray() const { return array_size >= 0; }

  bool IsScalarAndNotInlined() const {
    return array_size == kNotInlinedScalarSentinel;
  }

  bool IsScalarAndInlined() const {
    return array_size == kInlinedScalarSentinel;
  }

  // Return the pointer to the underlying data.
  const void* GetData() const { return IsScalarAndInlined() ? buffer : data; }

  template <typename T,
            typename std::enable_if_t<(sizeof(T) <= sizeof(void*))>* = nullptr>
  const T& GetScalarData() const {
    assert(IsScalarAndInlined());
    assert(type == GetOpAttrType<T>());
    return *reinterpret_cast<const T*>(buffer);
  }

  template <typename T,
            typename std::enable_if_t<(sizeof(T) > sizeof(void*))>* = nullptr>
  const T& GetScalarData() const {
    assert(IsScalarAndNotInlined());
    assert(type == GetOpAttrType<T>());
    return *static_cast<const T*>(data);
  }
};

// This class maintains a mutable set of attributes for a Op. This is
// implemented as an efficient in-place representation which backs off to an
// out-of-line representation when needed. This makes it efficient in the
// normal case of zero or a few attributes, but supports the general case of
// many large attributes when needed.
//
// OpAttrs differentiates between two different kinds of attributes: scalar
// and array attributes, and the getter will only see the attribute if it is
// used with the corresponding accessor. OpAttrs also has a 'raw' interface
// which can be used for dynamic reflection use-cases and you can also iterate
// over all of the attributes in the set. Iteration order is
// non-deterministic.
class OpAttrs final {
 public:
  OpAttrs();
  ~OpAttrs();

  // Clear all attributes. After calling Reset(), the client can reuse the
  // OpAttrs for another op.
  void Reset();

  // Set an attribute to the specified value, returning true on success or
  // false if an attribute with the specified name already exists.
  template <typename T>
  bool Set(string_view attr_name, const T& value) {
    return SetRaw(attr_name, &value, OpAttrsRawEntry::kScalarSentinel,
                  GetOpAttrType<T>());
  }

  // Overload for DenseAttr.
  bool Set(string_view attr_name, DenseAttr value) {
    return SetRaw(attr_name, value.data(), OpAttrsRawEntry::kScalarSentinel,
                  OpAttrType::DENSE);
  }

  // Overload for ShapeAttr.
  bool Set(string_view attr_name, ShapeAttr value) {
    return SetRaw(attr_name, value.data(), OpAttrsRawEntry::kScalarSentinel,
                  OpAttrType::SHAPE);
  }

  // Overload for AggregateAttr.
  bool Set(string_view attr_name, AggregateAttr value) {
    return SetRaw(attr_name, value.data(), OpAttrsRawEntry::kScalarSentinel,
                  OpAttrType::AGGREGATE);
  }

  // Read an attribute with the specified value, returning false on failure or
  // true on success.
  template <typename T>
  bool Get(string_view attr_name, T* value) const {
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || result->IsArray() || result->type != GetOpAttrType<T>())
      return false;
    *value = result->GetScalarData<T>();
    return true;
  }

  // Overload for DenseAttr.
  bool Get(string_view attr_name, DenseAttr* value) const {
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || result->IsArray() || result->type != OpAttrType::DENSE)
      return false;
    *value = DenseAttr(result->GetData());
    return true;
  }

  // Overload for ShapeAttr.
  bool Get(string_view attr_name, ShapeAttr* value) const {
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || result->IsArray() || result->type != OpAttrType::SHAPE)
      return false;
    *value = ShapeAttr(result->GetData());
    return true;
  }

  // Overload for AggregateAttr.
  bool Get(string_view attr_name, AggregateAttr* value) const {
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || result->IsArray() || result->type != OpAttrType::AGGREGATE)
      return false;
    *value = AggregateAttr(result->GetData());
    return true;
  }

  // Read a scalar attribute when it is known to exist. This asserts on
  // failure.
  template <typename T>
  T GetAsserting(string_view attr_name) const {
    T value;
    bool success = Get(attr_name, &value);
    assert(success && "OpAttrs::GetAsserting() failed");
    (void)success;
    return value;
  }

  template <typename T>
  Optional<T> GetOptional(string_view attr_name) const {
    T value;
    bool success = Get(attr_name, &value);
    if (success) {
      return value;
    } else {
      return llvm::None;
    }
  }

  template <typename T>
  bool SetArray(string_view attr_name, ArrayRef<T> value) {
    return SetRaw(attr_name, value.data(), value.size(), GetOpAttrType<T>());
  }

  template <typename T>
  bool GetArray(string_view attr_name, ArrayRef<T>* value) const {
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || !result->IsArray() || result->type != GetOpAttrType<T>())
      return false;
    *value = ArrayRef<T>(reinterpret_cast<const T*>(result->data),
                         result->array_size);
    return true;
  }

  // Read an array attribute when it is known to exist. This asserts on
  // failure.
  template <typename T>
  ArrayRef<T> GetArrayAsserting(string_view attr_name) const {
    ArrayRef<T> values;
    bool success = GetArray(attr_name, &values);
    assert(success);
    (void)success;
    return values;
  }

  template <typename T>
  ArrayRef<T> GetArrayOptional(string_view attr_name) const {
    ArrayRef<T> values;
    GetArray(attr_name, &values);
    return values;
  }

  // Support string_views as aliases of ArrayRef<char> in Get/Set.
  bool SetString(string_view attr_name, string_view value) {
    return SetArray(attr_name, ArrayRef<char>(value.data(), value.size()));
  }

  bool GetString(string_view attr_name, string_view* value) const {
    ArrayRef<char> value_ar;
    if (!GetArray(attr_name, &value_ar)) return false;
    *value = string_view(value_ar.data(), value_ar.size());
    return true;
  }

  // Read a string attribute when it is known to exist. This asserts on failure.
  string_view GetStringAsserting(string_view attr_name) const {
    string_view value;
    bool success = GetString(attr_name, &value);
    assert(success);
    (void)success;
    return value;
  }

  Optional<string_view> GetStringOptional(string_view attr_name) const {
    string_view value;
    bool success = GetString(attr_name, &value);
    if (success) {
      return value;
    } else {
      return llvm::None;
    }
  }

  // Support string_views as aliases of ArrayRef<char> in Get/Set.
  bool SetFunc(string_view attr_name, FunctionAttribute value) {
    return SetRaw(attr_name, value.func_name.data(), value.func_name.size(),
                  OpAttrType::FUNC);
  }

  bool GetFuncName(string_view attr_name, string_view* value) const {
    ArrayRef<char> value_ar;
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || !result->IsArray() || result->type != OpAttrType::FUNC)
      return false;

    value_ar = ArrayRef<char>(reinterpret_cast<const char*>(result->data),
                              result->array_size);
    *value = string_view(value_ar.data(), value_ar.size());
    return true;
  }

  // Read a function attribute when it is known to exist. This asserts on
  // failure.
  string_view GetFuncNameAsserting(string_view attr_name) const {
    string_view value;
    bool success = GetFuncName(attr_name, &value);
    assert(success);
    (void)success;
    return value;
  }

  Optional<string_view> GetFuncNameOptional(string_view attr_name) const {
    string_view value;
    bool success = GetFuncName(attr_name, &value);
    if (success) {
      return value;
    } else {
      return llvm::None;
    }
  }

  // Look up an attribute by name, regardless of its underlying type.
  // On lookup failure, pointer is null.
  const OpAttrsRawEntry* GetRaw(string_view attr_name) const;

  // Look up an attribute by name, regardless of its underlying type.
  const OpAttrsRawEntry& GetRawAsserting(string_view attr_name) const {
    auto* result = GetRaw(attr_name);
    assert(result);
    return *result;
  }

  // Set the specified attribute.  When num_elements is equal to
  // OpAttrsRawEntry::kScalarSentinel, this sets a scalar entry.  This
  // returns true on success or false if an attribute with that name already
  // exists.
  bool SetRaw(string_view attr_name, const void* data, int64_t num_elements,
              OpAttrType type);

  // Print the state of this attribute set, this is only intended for
  // debugging.
  void Print(raw_ostream& os) const;
  void Dump() const;

  size_t GetNumEntries() const;

  // Iterate over all of the entries in the attribute set, allowing dynamic
  // reflection.  Note that this produces the attributes in a
  // non-deterministic order.
  void IterateEntries(
      const std::function<void(const OpAttrsRawEntry& entry)>& fn) const;

  // Produce an immutable copy of this OpAttrs set on the heap and return a
  // reference to it.  This is the primary way to extend the lifetime of an
  // attribute set.
  OpAttrsRef freeze() const;

  bool IsOutOfLine() const { return out_of_line_representation_ != nullptr; }

 private:
  // This class is large and intended to live on the stack only.  It also
  // maintains interior pointers, so moving or copying isn't easy.
  //
  OpAttrs(const OpAttrs&) = delete;
  OpAttrs(const OpAttrs&&) = delete;
  void operator=(const OpAttrs&) = delete;

  void MoveOutOfLine();

  class OutOfLineRepresentation;
  friend class OutOfLineRepresentation;

  // Most op invocations have a small number of attributes, and we want them
  // to be formed without an allocation.  This array holds the string and
  // value data and is sized to be large enough to handle the common cases.
  //
  // Note that all of the inline state is undefined if OpAttrs has grown to
  // use an out-of-line representation.
  static constexpr uint8_t kInlineBufferSize = 128;
  alignas(void*) char inline_buffer_[kInlineBufferSize];

  // This is the number of bytes used in inline_buffer_.
  size_t inline_buffer_used_ = 0;

  // This is the inline representation of the entries index.  Ops should have
  // a small number of attributes, so we just do linear searches through them
  // in lookups.  We automatically switch to an indexed structure when this
  // number of entries is exceeded.
  static constexpr char kInlineEntriesSize = 6;
  OpAttrsRawEntry inline_entries_[kInlineEntriesSize];

  // This is the number of attribute entries in OpAttrs.
  size_t num_inline_entries_ = 0;

  // If non-null, OpAttrs overflowed the stack representation and all entries
  // got moved to the heap. All of the above entries are undefined when this
  // pointer is non-null.
  std::unique_ptr<OutOfLineRepresentation> out_of_line_representation_;

  // If non-null, then this OpAttrs set has been copied out to a compact
  // representation that can be closed over.  Subsequent mutations of this set
  // will drop this reference.
  mutable RCReference<ImmutableOpAttrs> frozen_representation_;
};

// An instance of this class is returned by OpAttrs::freeze() and may be used as
// an abstraction over clients that want read-only access to "either an
// immutable or a mutable OpAttrs".
//
// Notably, this avoids allocating in freeze() when there is a zero-element set.
class OpAttrsRef {
 public:
  // Form an empty OpAttrsRef, which has zero attributes in it.
  explicit OpAttrsRef() {}

  OpAttrsRef(OpAttrsRef&& other);

  // Form an OpAttrsRef with a mutable OpAttrs.
  explicit OpAttrsRef(const OpAttrs& attrs);

  // Form an OpAttrsRef with an ImmutableOpAttrs.
  explicit OpAttrsRef(RCReference<ImmutableOpAttrs> attrs);

  OpAttrsRef(const OpAttrsRef&) = delete;
  OpAttrsRef& operator=(const OpAttrsRef&) = delete;

  ~OpAttrsRef();

  // Read an attribute with the specified value, returning false on failure or
  // true on success.
  template <typename T>
  bool Get(string_view attr_name, T* value) const {
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || result->IsArray() || result->type != GetOpAttrType<T>())
      return false;
    *value = result->GetScalarData<T>();
    return true;
  }

  // Overload for DenseAttr.
  bool Get(string_view attr_name, DenseAttr* value) const {
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || result->IsArray() || result->type != OpAttrType::DENSE)
      return false;
    *value = DenseAttr(result->GetData());
    return true;
  }

  // Overload for ShapeAttr.
  bool Get(string_view attr_name, ShapeAttr* value) const {
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || result->IsArray() || result->type != OpAttrType::SHAPE)
      return false;
    *value = ShapeAttr(result->GetData());
    return true;
  }

  // Overload for AggregateAttr.
  bool Get(string_view attr_name, AggregateAttr* value) const {
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || result->IsArray() || result->type != OpAttrType::AGGREGATE)
      return false;
    *value = AggregateAttr(result->GetData());
    return true;
  }

  // Read a scalar attribute when it is known to exist. This asserts on failure.
  template <typename T>
  T GetAsserting(string_view attr_name) const {
    T value;
    bool success = Get(attr_name, &value);
    assert(success && "OpAttrs::GetAsserting() failed");
    (void)success;
    return value;
  }

  template <typename T>
  Optional<T> GetOptional(string_view attr_name) const {
    T value;
    bool success = Get(attr_name, &value);
    if (success) {
      return value;
    } else {
      return llvm::None;
    }
  }

  template <typename T>
  bool GetArray(string_view attr_name, ArrayRef<T>* value) const {
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || !result->IsArray() || result->type != GetOpAttrType<T>())
      return false;
    *value = ArrayRef<T>(reinterpret_cast<const T*>(result->data),
                         result->array_size);
    return true;
  }

  // Read an array attribute when it is known to exist. This asserts on failure.
  template <typename T>
  ArrayRef<T> GetArrayAsserting(string_view attr_name) const {
    ArrayRef<T> values;
    bool success = GetArray(attr_name, &values);
    assert(success);
    (void)success;
    return values;
  }

  template <typename T>
  ArrayRef<T> GetArrayOptional(string_view attr_name) const {
    ArrayRef<T> values;
    GetArray(attr_name, &values);
    return values;
  }

  bool GetString(string_view attr_name, string_view* value) const {
    ArrayRef<char> value_ar;
    if (!GetArray(attr_name, &value_ar)) return false;
    *value = string_view(value_ar.data(), value_ar.size());
    return true;
  }

  // Read a string attribute when it is known to exist. This asserts on failure.
  string_view GetStringAsserting(string_view attr_name) const {
    string_view value;
    bool success = GetString(attr_name, &value);
    assert(success);
    (void)success;
    return value;
  }

  Optional<string_view> GetStringOptional(string_view attr_name) const {
    string_view value;
    bool success = GetString(attr_name, &value);
    if (success) {
      return value;
    } else {
      return llvm::None;
    }
  }

  bool GetFuncName(string_view attr_name, string_view* value) const {
    ArrayRef<char> value_ar;
    const OpAttrsRawEntry* result = GetRaw(attr_name);
    if (!result || !result->IsArray() || result->type != OpAttrType::FUNC)
      return false;

    value_ar = ArrayRef<char>(reinterpret_cast<const char*>(result->data),
                              result->array_size);
    *value = string_view(value_ar.data(), value_ar.size());
    return true;
  }

  // Read a function attribute when it is known to exist. This asserts on
  // failure.
  string_view GetFuncNameAsserting(string_view attr_name) const {
    string_view value;
    bool success = GetFuncName(attr_name, &value);
    assert(success);
    (void)success;
    return value;
  }

  Optional<string_view> GetFuncNameOptional(string_view attr_name) const {
    string_view value;
    bool success = GetFuncName(attr_name, &value);
    if (success) {
      return value;
    } else {
      return llvm::None;
    }
  }

  // Return the number of entries in this set.
  size_t GetNumEntries() const;

  // Iterate over all of the entries in the attribute set, allowing dynamic
  // reflection.  This returns the entries in a determinstic order if the
  // underlying representation is frozen, otherwise not.
  void IterateEntries(
      const std::function<void(const OpAttrsRawEntry& entry)>& fn) const;

  const OpAttrsRawEntry* GetRaw(string_view attr_name) const;

  // Look up an attribute by name, regardless of its underlying type.
  const OpAttrsRawEntry& GetRawAsserting(string_view attr_name) const {
    auto* result = GetRaw(attr_name);
    assert(result);
    return *result;
  }

  // Return a reference that is guaranteed stable on the heap.
  OpAttrsRef freeze() const;

  // Print the state of this attribute set, this is only intended for debugging.
  void Print(raw_ostream& os) const;
  void Dump() const;

 private:
  // If the pointer is null, then the represented attribute set is empty.
  llvm::PointerUnion<const OpAttrs*, ImmutableOpAttrs*> attrs_;
};

// Return the OpAttrType converted from DType.
OpAttrType GetOpAttrTypeFromDType(DType::Kind kind);

// Return the OpAttrType converted from BEFAttributeType in BEF.
OpAttrType GetOpAttrTypeFromBEFAttributeType(BEFAttributeType kind);

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_OP_ATTR_H_
