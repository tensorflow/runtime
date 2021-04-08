// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements the OpAttrs class.

#include "tfrt/core_runtime/op_attrs.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/support/alloc.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {

const int64_t OpAttrsRawEntry::kNotInlinedScalarSentinel;
const int64_t OpAttrsRawEntry::kInlinedScalarSentinel;

int64_t OpAttrsRawEntry::DecodeArraySize(int64_t encoded_array_size) {
  if (encoded_array_size >= 0) return encoded_array_size;
  if (encoded_array_size == OpAttrsRawEntry::kNotInlinedScalarSentinel ||
      encoded_array_size == OpAttrsRawEntry::kInlinedScalarSentinel)
    return 1;
  llvm_unreachable("encoded_array_size should not be negative");
}

int64_t OpAttrsRawEntry::EncodeArraySize(int64_t num_elements, bool inlined) {
  if (num_elements >= 0) return num_elements;
  if (num_elements == OpAttrsRawEntry::kScalarSentinel && inlined)
    return OpAttrsRawEntry::kInlinedScalarSentinel;
  if (num_elements == OpAttrsRawEntry::kScalarSentinel && !inlined)
    return OpAttrsRawEntry::kNotInlinedScalarSentinel;
  if (num_elements == OpAttrsRawEntry::kInlinedScalarSentinel ||
      num_elements == OpAttrsRawEntry::kNotInlinedScalarSentinel)
    return num_elements;
  llvm_unreachable("num_elements not encodable");
}

int64_t OpAttrsRawEntry::GetNumElements(int64_t num_elements) {
  if (num_elements >= 0) return num_elements;
  if (num_elements == OpAttrsRawEntry::kScalarSentinel ||
      num_elements == OpAttrsRawEntry::kNotInlinedScalarSentinel ||
      num_elements == OpAttrsRawEntry::kInlinedScalarSentinel)
    return 1;
  llvm_unreachable("num_elements value error");
}

OpAttrType GetOpAttrTypeFromDType(DType::Kind kind) {
  // TODO(tfrt-devs): Unify BEFAttributeType, OpAttrType and tfrt::DType.
  switch (kind) {
    case DType::I1:
      return OpAttrType::BOOL;
    case DType::I8:
      return OpAttrType::I8;
    case DType::I16:
      return OpAttrType::I16;
    case DType::I32:
      return OpAttrType::I32;
    case DType::I64:
      return OpAttrType::I64;
    case DType::UI8:
      return OpAttrType::UI8;
    case DType::UI16:
      return OpAttrType::UI16;
    case DType::UI32:
      return OpAttrType::UI32;
    case DType::UI64:
      return OpAttrType::UI64;
    case DType::BF16:
      return OpAttrType::BF16;
    case DType::F16:
      return OpAttrType::F16;
    case DType::F32:
      return OpAttrType::F32;
    case DType::F64:
      return OpAttrType::F64;
    case DType::Complex64:
      return OpAttrType::COMPLEX64;
    case DType::Complex128:
      return OpAttrType::COMPLEX128;
    case DType::String:
      return OpAttrType::CHAR;
    case DType::Resource:
      return OpAttrType::UNSUPPORTED_RESOURCE;
    case DType::Variant:
      return OpAttrType::UNSUPPORTED_VARIANT;
    default:
      break;
  }

  llvm_unreachable("unsupported DType in Core Runtime.");
}

// Return the OpAttrType converted from BEFAttributeType in BEF.
OpAttrType GetOpAttrTypeFromBEFAttributeType(BEFAttributeType kind) {
  if (IsDenseAttribute(kind)) return OpAttrType::DENSE;

  if (IsDataTypeAttribute(kind))
    return GetOpAttrTypeFromDType(GetDataType(kind));

  switch (kind) {
    case BEFAttributeType::kAggregate:
      return OpAttrType::AGGREGATE;
    case BEFAttributeType::kType:
      return OpAttrType::DTYPE;
    case BEFAttributeType::kShape:
      return OpAttrType::SHAPE;
    default:
      break;
  }

  llvm_unreachable("unsupported BEFAttributeType in Core Runtime.");
}

// This class is an immutable copy of OpAttrs, designed for Op implementations
// where it is convenient to capture an entire attribute set. This is designed
// to do at most one allocation to store the entire attribute set - no matter
// how small or large the attribute set is.
//
class ImmutableOpAttrs : public ReferenceCounted<ImmutableOpAttrs> {
 public:
  // Note: users should not directly interface with this class, they should
  // generally use OpAttrsRef instead.
  const OpAttrsRawEntry *GetRaw(string_view attr_name) const;
  size_t GetNumEntries() const { return num_entries_; }
  void IterateEntries(
      const std::function<void(const OpAttrsRawEntry &entry)> &fn) const;

 private:
  friend class ReferenceCounted<ImmutableOpAttrs>;
  friend class OpAttrs;

  static RCReference<ImmutableOpAttrs> create(const OpAttrs &attrs);
  explicit ImmutableOpAttrs(size_t num_entries) : num_entries_(num_entries) {}

  void Destroy();

  // This is the number of entries in this set.
  size_t num_entries_;

  // The entries_ array is tail allocated here, and followed by the payload
  // data for the attributes.
  OpAttrsRawEntry entries_[];
};

// Return the size and alignment of the specified attribute type.
std::pair<size_t, size_t> GetHostSizeAndAlignment(const void *data,
                                                  OpAttrType type) {
  switch (type) {
    case OpAttrType::DTYPE:
      return {sizeof(OpAttrType), alignof(OpAttrType)};
    case OpAttrType::AGGREGATE: {
      AggregateAttr aggregate_attr(data);
      return {aggregate_attr.size(), sizeof(void *)};
    }
    case OpAttrType::DENSE: {
      DenseAttr dense_attr(data);
      return {dense_attr.size(), DenseAttr::Alignment()};
    }
    case OpAttrType::SHAPE: {
      ShapeAttr shape_attr(data);
      return {shape_attr.size(), ShapeAttr::Alignment()};
    }
    case OpAttrType::FUNC:
      return {sizeof(char), alignof(char)};
    case OpAttrType::BF16:
      return {sizeof(bf16), alignof(bf16)};
    case OpAttrType::F16:
      return {sizeof(fp16), alignof(fp16)};
    case OpAttrType::I1:
      return {sizeof(i1), alignof(i1)};
    case OpAttrType::COMPLEX64:
      return {sizeof(std::complex<float>), alignof(std::complex<float>)};
    case OpAttrType::COMPLEX128:
      return {sizeof(std::complex<double>), alignof(std::complex<double>)};
    case OpAttrType::UNSUPPORTED_RESOURCE:
    case OpAttrType::UNSUPPORTED_VARIANT:
      llvm_unreachable("unsupported attribute type");
#define OP_ATTR_TYPE(ENUM, CPP_TYPE) \
  case OpAttrType::ENUM:             \
    return {sizeof(CPP_TYPE), alignof(CPP_TYPE)};
#include "tfrt/core_runtime/op_attr_type.def"
  }
}

// Return the name of the specified attribute type, e.g. "I32".
const char *GetNameString(OpAttrType type) {
  switch (type) {
    default:
      llvm_unreachable("unsupported attribute type");
      break;
    case OpAttrType::DTYPE:
      return "DTYPE";
    case OpAttrType::AGGREGATE:
      return "AGGREGATE";
    case OpAttrType::DENSE:
      return "DENSE";
    case OpAttrType::SHAPE:
      return "SHAPE";
    case OpAttrType::FUNC:
      return "FUNC";
    case OpAttrType::BF16:
      return "BF16";
    case OpAttrType::F16:
      return "F16";
    case OpAttrType::I1:
      return "I1";
    case OpAttrType::COMPLEX64:
      return "COMPLEX64";
    case OpAttrType::COMPLEX128:
      return "COMPLEX128";
    // following two types are not natively supported in TFRT.
    case OpAttrType::UNSUPPORTED_RESOURCE:
      return "RESOURCE";
    case OpAttrType::UNSUPPORTED_VARIANT:
      return "VARIANT";
#define OP_ATTR_TYPE(ENUM, CPP_TYPE) \
  case OpAttrType::ENUM:             \
    return #ENUM;
#include "tfrt/core_runtime/op_attr_type.def"
  }
}

static void GetSortedAttrs(const OpAttrsRef &attrs,
                           SmallVectorImpl<const OpAttrsRawEntry *> *result) {
  // Collect the attributes in non-determinstic order.
  attrs.IterateEntries(
      [&](const OpAttrsRawEntry &entry) { result->push_back(&entry); });

  // Sort the elements by attribute name.
  llvm::array_pod_sort(result->begin(), result->end(),
                       [](const OpAttrsRawEntry *const *lhs,
                          const OpAttrsRawEntry *const *rhs) -> int {
                         return strcmp((*lhs)->name, (*rhs)->name);
                       });
}

//===----------------------------------------------------------------------===//
// OpAttrs::OutOfLineRepresentation implementation
//===----------------------------------------------------------------------===//

class OpAttrs::OutOfLineRepresentation {
 public:
  explicit OutOfLineRepresentation(OpAttrs *orig_attrs);

  void IterateEntries(
      const std::function<void(const OpAttrsRawEntry &entry)> &fn) const {
    for (auto &entry : entries_) fn(entry.second);
  }

  const OpAttrsRawEntry *GetRaw(string_view attr_name) const {
    auto it = entries_.find(attr_name);
    return it == entries_.end() ? nullptr : &it->second;
  }

  bool SetRaw(string_view attr_name, const void *data, int64_t num_elements,
              OpAttrType type);

  size_t GetNumEntries() const { return entries_.size(); }

 public:
  llvm::StringMap<OpAttrsRawEntry> entries_;
  llvm::BumpPtrAllocatorImpl<llvm::MallocAllocator, 256, 256> allocator_;
};

// The constructor for the out-of-line representation starts by moving the data
// out of the in-line representation.
OpAttrs::OutOfLineRepresentation::OutOfLineRepresentation(OpAttrs *orig_attrs) {
  // We move to an out-of-line representation by copying all the entries over.
  orig_attrs->IterateEntries([&](const OpAttrsRawEntry &entry) {
    bool success =
        this->SetRaw(entry.name, entry.GetData(), entry.array_size, entry.type);
    assert(success && "input cannot have dupes, so collisions aren't possible");
    (void)success;
  });
}

bool OpAttrs::OutOfLineRepresentation::SetRaw(string_view attr_name,
                                              const void *data,
                                              int64_t num_elements,
                                              OpAttrType type) {
  // Figure out what entry we need to fill in.
  auto entry_it_pair =
      entries_.insert(std::make_pair(attr_name, OpAttrsRawEntry()));

  // If it is already present, then return false to indicate a conflict.
  if (entry_it_pair.second == false) return false;

  auto type_size_and_alignment = GetHostSizeAndAlignment(data, type);

  // TODO(clattner): consider unifying the logic here with OpAttrs::SetRaw.
  auto type_size = type_size_and_alignment.first;
  auto type_alignment = type_size_and_alignment.second;

  auto &entry = entry_it_pair.first->second;
  auto bytes_to_copy =
      OpAttrsRawEntry::GetNumElements(num_elements) * type_size;
  bool is_array = num_elements >= 0;
  bool is_small = type_size <= sizeof(void *);

  // bytes_to_copy can be 0 for an empty array.
  if ((is_array && bytes_to_copy > 0) || !is_small) {
    void *our_data = allocator_.Allocate(bytes_to_copy, type_alignment);
    void *element_ptr = static_cast<char *>(our_data);

    // Copy the element(s) themselves.
    memcpy(element_ptr, data, bytes_to_copy);
    entry.array_size =
        OpAttrsRawEntry::EncodeArraySize(num_elements, /*inlined=*/false);
    entry.data = element_ptr;
  } else {
    // If it is a small scalar, copy the data to the inlined buffer.
    assert(type_alignment <= alignof(void *));
    assert(bytes_to_copy <= sizeof(void *));
    entry.array_size =
        OpAttrsRawEntry::EncodeArraySize(num_elements, /*inlined=*/true);
    if (bytes_to_copy > 0) {
      memcpy(entry.buffer, data, bytes_to_copy);
    }
  }

  entry.name = entry_it_pair.first->first().data();
  entry.type = type;
  return true;
}

//===----------------------------------------------------------------------===//
// OpAttrs implementation
//===----------------------------------------------------------------------===//

OpAttrs::OpAttrs() {}

OpAttrs::~OpAttrs() {}

size_t OpAttrs::GetNumEntries() const {
  // If we are using an out of line representation, delegate to it.
  if (auto *out_of_line = out_of_line_representation_.get())
    return out_of_line->GetNumEntries();

  return num_inline_entries_;
}

void OpAttrs::IterateEntries(
    const std::function<void(const OpAttrsRawEntry &entry)> &fn) const {
  // If we are using an out of line representation, delegate to it.
  if (auto *out_of_line = out_of_line_representation_.get())
    return out_of_line->IterateEntries(fn);

  for (size_t i = 0, e = num_inline_entries_; i != e; ++i)
    fn(inline_entries_[i]);
}

// This method copies the inline representation to the out-of-line form.
void OpAttrs::MoveOutOfLine() {
  assert(!out_of_line_representation_ && "is already out of line");
  out_of_line_representation_ = std::make_unique<OutOfLineRepresentation>(this);
}

const OpAttrsRawEntry *OpAttrs::GetRaw(string_view attr_name) const {
  // If we are using an out of line representation, delegate to it.
  if (auto *out_of_line = out_of_line_representation_.get())
    return out_of_line->GetRaw(attr_name);

  for (size_t i = 0, e = num_inline_entries_; i != e; ++i) {
    auto &entry = inline_entries_[i];
    if (strcmp(attr_name.data(), entry.name) == 0) return &entry;
  }
  return nullptr;
}

void OpAttrs::Reset() {
  inline_buffer_used_ = 0;
  num_inline_entries_ = 0;
  if (out_of_line_representation_) out_of_line_representation_.reset();
  frozen_representation_.reset();
}

bool OpAttrs::SetRaw(string_view attr_name, const void *data,
                     int64_t num_elements, OpAttrType type) {
  // If we're mutating the set, then drop any frozen representation that may be
  // formed.
  if (frozen_representation_) frozen_representation_.reset();

  // If we are using an out of line representation, delegate to it.
  if (auto *out_of_line = out_of_line_representation_.get())
    return out_of_line->SetRaw(attr_name, data, num_elements, type);

  // Otherwise, we need to find out if this entry has already been installed.
  // If so, we return failure.
  for (size_t i = 0, e = num_inline_entries_; i != e; ++i) {
    auto &entry = inline_entries_[i];
    if (strcmp(attr_name.data(), entry.name) == 0) return false;
  }

  // Ok, we're going to do an insertion.  If we are out of space in
  // inline_entries_  then we have to move out of line.
  if (num_inline_entries_ == kInlineEntriesSize) {
    MoveOutOfLine();
    return out_of_line_representation_->SetRaw(attr_name, data, num_elements,
                                               type);
  }

  // We also need space in inline_buffer_.
  auto *name_pointer = inline_buffer_ + inline_buffer_used_;
  auto attr_name_size = attr_name.size();
  inline_buffer_used_ += attr_name_size + 1;

  // If we are out of space, then switch to an out-of-line representation.
  if (inline_buffer_used_ > kInlineBufferSize) {
    MoveOutOfLine();
    return out_of_line_representation_->SetRaw(attr_name, data, num_elements,
                                               type);
  }

  // Otherwise, we have space, so copy over the string.
  memcpy(name_pointer, attr_name.data(), attr_name_size);
  name_pointer[attr_name_size] = 0;  // Null terminate C string.

  auto type_size_and_alignment = GetHostSizeAndAlignment(data, type);
  auto type_size = type_size_and_alignment.first;
  auto type_alignment = type_size_and_alignment.second;

  bool is_array = num_elements >= 0;
  bool is_small = type_size <= sizeof(void *);
  // If it is a small scalar, we can just copy the data to OpAttrsRawEntry's
  // inlined buffer.
  if (is_small && !is_array) {
    // Fill in the attribute entry.
    auto &entry = inline_entries_[num_inline_entries_++];
    entry.name = name_pointer;

    assert(type_alignment <= alignof(void *));
    memcpy(entry.buffer, data, type_size);

    // Set to inline scalar.
    entry.array_size =
        OpAttrsRawEntry::EncodeArraySize(num_elements, /*inlined=*/true);
    entry.type = type;
    return true;
  }

  // Otherwise we have non-standard attribute, then we need to copy the array to
  // OpAttrs' inlined buffer.

  // Then we hold the data. Round the element pointer up to the alignment
  // boundary required by type so we can figure out where we will be inserting
  // it.
  inline_buffer_used_ =
      static_cast<size_t>(llvm::alignTo(inline_buffer_used_, type_alignment));
  auto *dest_pointer = inline_buffer_ + inline_buffer_used_;

  // The number of bytes is equal to the number of elements times its size.
  auto bytes_to_copy =
      OpAttrsRawEntry::GetNumElements(num_elements) * type_size;
  inline_buffer_used_ += bytes_to_copy;

  // If we are out of space, then switch to an out-of-line representation.
  if (inline_buffer_used_ > kInlineBufferSize) {
    MoveOutOfLine();
    return out_of_line_representation_->SetRaw(attr_name, data, num_elements,
                                               type);
  }
  memcpy(dest_pointer, data, bytes_to_copy);

  // Fill in the attribute entry.
  auto &entry = inline_entries_[num_inline_entries_++];
  entry.name = name_pointer;
  entry.data = dest_pointer;
  entry.array_size =
      OpAttrsRawEntry::EncodeArraySize(num_elements, /*inlined=*/false);
  entry.type = type;
  return true;
}

// Print the state of this attribute set, this is only intended for debugging.
void OpAttrs::Print(raw_ostream &os) const { OpAttrsRef(*this).Print(os); }

void OpAttrs::Dump() const { Print(llvm::errs()); }

// Produce an immutable copy of this OpAttrs set on the heap and return a
// reference to it.  This is the primary way to extend the lifetime of an
// attribute set.
OpAttrsRef OpAttrs::freeze() const {
  // Empty sets are very common - avoid allocating for them, just return an
  // empty set reference.
  if (GetNumEntries() == 0) return OpAttrsRef();

  // We may have already computed a frozen representation - if not, create one.
  if (!frozen_representation_)
    frozen_representation_ = ImmutableOpAttrs::create(*this);

  return OpAttrsRef(frozen_representation_.CopyRef());
}

RCReference<ImmutableOpAttrs> ImmutableOpAttrs::create(const OpAttrs &attrs) {
  // Sort the elements by attribute name.
  SmallVector<const OpAttrsRawEntry *, 16> sorted_attrs;
  GetSortedAttrs(OpAttrsRef(attrs), &sorted_attrs);

  // Figure out how much space we need to hold these attributes.
  size_t alloc_size =
      sizeof(ImmutableOpAttrs) + sizeof(OpAttrsRawEntry) * sorted_attrs.size();

  // TODO(clattner): When coming from an inlined representation (the vastly most
  // common case, we should be able to memcpy over one big block of memory and
  // update pointers, instead of doing several small memcpy's.

  // Figure out how much space we need for each entry, which is the space for
  // the name and the payload together:
  for (auto *entry : sorted_attrs) {
    // Space for the name and null terminator.
    alloc_size += strlen(entry->name) + 1;

    // For an inlined entry, no additional memory is needed.
    if (entry->IsScalarAndInlined()) continue;

    // Round up to the required alignment.
    auto size_type_alignment =
        GetHostSizeAndAlignment(entry->data, entry->type);
    alloc_size = static_cast<size_t>(
        llvm::alignTo(alloc_size, size_type_alignment.second));

    // Add space for the elements.
    alloc_size += OpAttrsRawEntry::DecodeArraySize(entry->array_size) *
                  size_type_alignment.first;
  }

  // Now that we know the size, create the result.
  auto *raw_memory = AlignedAlloc(alignof(ImmutableOpAttrs), alloc_size);
  auto *result = new (raw_memory) ImmutableOpAttrs(sorted_attrs.size());

  char *data_ptr = static_cast<char *>(raw_memory) + sizeof(ImmutableOpAttrs) +
                   sizeof(OpAttrsRawEntry) * sorted_attrs.size();

  // Copy all of the attributes over.
  for (size_t i = 0, e = sorted_attrs.size(); i != e; ++i) {
    const auto &src_entry = *sorted_attrs[i];
    auto &result_entry = result->entries_[i];

    // Copy simple properties.
    result_entry.array_size = src_entry.array_size;
    result_entry.type = src_entry.type;

    // Copy the name over.
    result_entry.name = data_ptr;
    auto name_len = strlen(src_entry.name);
    memcpy(data_ptr, src_entry.name, name_len + 1);
    data_ptr += name_len + 1;

    auto size_and_alignment =
        GetHostSizeAndAlignment(src_entry.data, src_entry.type);
    auto type_size = size_and_alignment.first;
    auto type_alignment = size_and_alignment.second;

    if (src_entry.IsArray() || type_size > sizeof(void *)) {
      // Round up to the required alignment.
      data_ptr = reinterpret_cast<char *>(
          llvm::alignTo(reinterpret_cast<uint64_t>(data_ptr), type_alignment));

      // Copy over the elements, including the array_size if present.
      size_t elements_size =
          OpAttrsRawEntry::DecodeArraySize(src_entry.array_size) * type_size;
      memcpy(data_ptr, static_cast<const char *>(src_entry.data),
             elements_size);

      // Remember that this is where the element is.
      result_entry.data = data_ptr;
      data_ptr += elements_size;
    } else {
      memcpy(result_entry.buffer, src_entry.buffer, sizeof(void *));
    }
  }

  // Make sure the memory usage is same as expected.
  assert(data_ptr == static_cast<char *>(raw_memory) + alloc_size);

  return TakeRef(result);
}

//===----------------------------------------------------------------------===//
// ImmutableOpAttrs implementation
//===----------------------------------------------------------------------===//

// We just use malloc to allocate this, so we need to provide a custom destroy
// hook for ReferenceCounted to use.
void ImmutableOpAttrs::Destroy() {
  this->~ImmutableOpAttrs();
  free(this);
}

// Look up an attribute by name, regardless of its underlying type.
// On lookup failure, the result is null.
const OpAttrsRawEntry *ImmutableOpAttrs::GetRaw(string_view attr_name) const {
  // If we only have a few entries, do a linear search for the name.
  // TODO(tf_runtime_team): implement a binary search for more elements.
  for (size_t i = 0, e = num_entries_; i != e; ++i) {
    if (!strcmp(entries_[i].name, attr_name.data())) return &entries_[i];
  }
  return nullptr;
}

// Iterate over all of the entries in the attribute set, allowing dynamic
// reflection.  Note that this produces the attributes in a *deterministic*
// order.
void ImmutableOpAttrs::IterateEntries(
    const std::function<void(const OpAttrsRawEntry &entry)> &fn) const {
  for (size_t i = 0, e = num_entries_; i != e; ++i) fn(entries_[i]);
}

//===----------------------------------------------------------------------===//
// OpAttrsRef implementation
//===----------------------------------------------------------------------===//

OpAttrsRef::OpAttrsRef(const OpAttrs &attrs) { attrs_ = &attrs; }

OpAttrsRef::OpAttrsRef(RCReference<ImmutableOpAttrs> attrs) {
  attrs_ = attrs.release();
}

OpAttrsRef::OpAttrsRef(OpAttrsRef &&other) : attrs_(other.attrs_) {
  other.attrs_ = nullptr;
}

OpAttrsRef::~OpAttrsRef() {
  if (auto *ptr = attrs_.dyn_cast<ImmutableOpAttrs *>()) ptr->DropRef();
}

// Return the number of entries in this set.
size_t OpAttrsRef::GetNumEntries() const {
  if (auto *ptr = attrs_.dyn_cast<const OpAttrs *>())
    return ptr->GetNumEntries();
  if (auto *ptr = attrs_.dyn_cast<ImmutableOpAttrs *>())
    return ptr->GetNumEntries();

  return 0;
}

// Iterate over all of the entries in the attribute set, allowing dynamic
// reflection.  This returns the entries in a determinstic order if the
// underlying representation is frozen, otherwise not.
void OpAttrsRef::IterateEntries(
    const std::function<void(const OpAttrsRawEntry &entry)> &fn) const {
  if (auto *ptr = attrs_.dyn_cast<const OpAttrs *>())
    return ptr->IterateEntries(fn);
  if (auto *ptr = attrs_.dyn_cast<ImmutableOpAttrs *>())
    return ptr->IterateEntries(fn);
}

const OpAttrsRawEntry *OpAttrsRef::GetRaw(string_view attr_name) const {
  if (auto *ptr = attrs_.dyn_cast<const OpAttrs *>())
    return ptr->GetRaw(attr_name);
  if (auto *ptr = attrs_.dyn_cast<ImmutableOpAttrs *>())
    return ptr->GetRaw(attr_name);
  return nullptr;
}

// Print a single element of an attribute out.
static void PrintElement(const void *ptr, OpAttrType type, raw_ostream &os) {
  switch (type) {
    case OpAttrType::DTYPE: {
      auto dtype = *static_cast<const OpAttrType *>(ptr);
      assert(dtype != OpAttrType::DTYPE);
      os << GetNameString(dtype);
      break;
    }
    case OpAttrType::AGGREGATE: {
      AggregateAttr aggregate_attr(ptr);
      size_t num_elements = aggregate_attr.GetNumElements();
      os << "elt_count=" << num_elements << " [";
      for (int i = 0; i < num_elements; ++i) {
        auto base = aggregate_attr.GetAttribute(i);
        os << "{";
        if (IsDenseAttribute(base.type()) ||
            base.type() == BEFAttributeType::kAggregate) {
          PrintElement(base.data(),
                       GetOpAttrTypeFromBEFAttributeType(base.type()), os);
        } else {
          // TODO(chky): Support other types.
          os << "unknown";
        }
        os << "}";
        if (i < num_elements - 1) os << ", ";
      }
      os << "]";
      break;
    }
    case OpAttrType::DENSE: {
      DenseAttr dense_attr(ptr);
      os << "dtype="
         << GetNameString(GetOpAttrTypeFromBEFAttributeType(
                static_cast<BEFAttributeType>(dense_attr.dtype())))
         << ", rank=" << dense_attr.shape().size()
         << ", elt_count=" << dense_attr.GetNumElements();
      break;
    }
    case OpAttrType::SHAPE: {
      ShapeAttr shape_attr(ptr);
      os << "<";
      if (auto ranked_shape = shape_attr.dyn_cast<RankedShapeAttr>())
        llvm::interleave(ranked_shape.GetShape(), os, "x");
      else
        os << "*";
      os << ">";
      break;
    }
    case OpAttrType::FUNC:
      os << GetNameString(OpAttrType::FUNC);
      os << " function_name: " << *static_cast<const char *>(ptr);
      break;
    case OpAttrType::BF16:
      assert(0 && "cannot print bf16 yet.");
      break;
    case OpAttrType::F16:
      // TODO(b/149063226): Support FP16.
      assert(0 && "cannot print fp16 yet.");
      break;
    case OpAttrType::I1:
      os << *static_cast<const uint8_t *>(ptr);
      break;
    case OpAttrType::COMPLEX64:
      os << "(" << static_cast<const std::complex<float> *>(ptr)->real() << ","
         << static_cast<const std::complex<float> *>(ptr)->imag() << ")";
      break;
    case OpAttrType::COMPLEX128:
      os << "(" << static_cast<const std::complex<double> *>(ptr)->real() << ","
         << static_cast<const std::complex<double> *>(ptr)->imag() << ")";
      break;
    case OpAttrType::UNSUPPORTED_RESOURCE:
    case OpAttrType::UNSUPPORTED_VARIANT:
      llvm_unreachable("unsupported attribute type");
#define OP_ATTR_TYPE(ENUM, CPP_TYPE)           \
  case OpAttrType::ENUM:                       \
    os << *static_cast<const CPP_TYPE *>(ptr); \
    break;
#include "tfrt/core_runtime/op_attr_type.def"
  }
}

OpAttrsRef OpAttrsRef::freeze() const {
  if (auto *ptr = attrs_.dyn_cast<const OpAttrs *>()) return ptr->freeze();
  if (auto *ptr = attrs_.dyn_cast<ImmutableOpAttrs *>())
    return OpAttrsRef(FormRef(ptr));

  return OpAttrsRef();
}

// Print the state of this attribute set, this is only intended for debugging.
void OpAttrsRef::Print(raw_ostream &os) const {
  if (GetNumEntries() == 0) {
    os << "OpAttrs is empty\n";
    return;
  }

  os << "OpAttrs contains " << GetNumEntries() << " entries:\n";

  // Sort the elements by attribute name.
  SmallVector<const OpAttrsRawEntry *, 16> sorted_attrs;
  GetSortedAttrs(*this, &sorted_attrs);

  // Print out the attributes in stable order.
  for (auto *attr : sorted_attrs) {
    const OpAttrsRawEntry &entry = *attr;
    os << "  '" << entry.name << "'"
       << " type=" << GetNameString(entry.type) << " value=";

    auto type_size = GetHostSizeAndAlignment(entry.data, entry.type).first;
    if (entry.IsArray()) {
      const char *data = static_cast<const char *>(entry.data);
      os << '[';

      for (size_t i = 0, e = OpAttrsRawEntry::DecodeArraySize(entry.array_size);
           i != e; ++i) {
        // Only print the first elements of large arrays.
        if (i == 5) {
          os << "...";
          break;
        }
        if (i != 0) os << ", ";
        PrintElement(data + i * type_size, entry.type, os);
      }
      os << ']';
    } else if (type_size <= sizeof(void *)) {
      PrintElement(entry.buffer, entry.type, os);
    } else {
      PrintElement(entry.data, entry.type, os);
    }
    os << '\n';
  }
  os << '\n';
}

void OpAttrsRef::Dump() const { Print(llvm::errs()); }

}  // namespace tfrt
