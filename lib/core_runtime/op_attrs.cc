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
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/core_runtime/op_attr_type.h"
#include "tfrt/dtype/quantized_types.h"
#include "tfrt/support/alloc.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {

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
    case DType::QUI8:
      return OpAttrType::UNSUPPORTED_QUI8;
    case DType::QUI16:
      return OpAttrType::UNSUPPORTED_QUI16;
    case DType::QI8:
      return OpAttrType::UNSUPPORTED_QI8;
    case DType::QI16:
      return OpAttrType::UNSUPPORTED_QI16;
    case DType::QI32:
      return OpAttrType::UNSUPPORTED_QI32;
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
      return {aggregate_attr.GetByteSize(), aggregate_attr.Alignment()};
    }
    case OpAttrType::DENSE: {
      DenseAttr dense_attr(data);
      return {dense_attr.GetByteSize(), dense_attr.Alignment()};
    }
    case OpAttrType::SHAPE: {
      ShapeAttr shape_attr(data);
      return {shape_attr.GetByteSize(), shape_attr.Alignment()};
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
    case OpAttrType::UNSUPPORTED_QUI8:
    case OpAttrType::UNSUPPORTED_QUI16:
    case OpAttrType::UNSUPPORTED_QI8:
    case OpAttrType::UNSUPPORTED_QI16:
    case OpAttrType::UNSUPPORTED_QI32:
      llvm_unreachable("unsupported attribute type");
#define OP_ATTR_TYPE(ENUM, CPP_TYPE) \
  case OpAttrType::ENUM:             \
    return {sizeof(CPP_TYPE), alignof(CPP_TYPE)};
#include "tfrt/core_runtime/op_attr_type.def"
  }
}

// Return the required alignment padding size to place an Op attribute.
size_t GetAlignmentPaddingSize(const void *data, OpAttrType type,
                               unsigned offset) {
  size_t peak_alignment;
  size_t prefix_size = 0;
  switch (type) {
    case OpAttrType::DTYPE:
      peak_alignment = alignof(OpAttrType);
      break;

    case OpAttrType::AGGREGATE: {
      AggregateAttr attr(data);
      prefix_size = attr.GetPrefixSize();
      peak_alignment = attr.Alignment();
      break;
    }

    case OpAttrType::DENSE: {
      DenseAttr attr(data);
      prefix_size = attr.GetPrefixSize();
      peak_alignment = attr.Alignment();
      break;
    }

    case OpAttrType::SHAPE: {
      ShapeAttr attr(data);
      prefix_size = attr.GetPrefixSize();
      peak_alignment = attr.Alignment();
      break;
    }

    case OpAttrType::FUNC:
      peak_alignment = alignof(char);
      break;

    case OpAttrType::BF16:
      peak_alignment = alignof(bf16);
      break;

    case OpAttrType::F16:
      peak_alignment = alignof(fp16);
      break;

    case OpAttrType::I1:
      peak_alignment = alignof(i1);
      break;

    case OpAttrType::COMPLEX64:
      peak_alignment = alignof(std::complex<float>);
      break;

    case OpAttrType::COMPLEX128:
      peak_alignment = alignof(std::complex<double>);
      break;

    case OpAttrType::UNSUPPORTED_RESOURCE:
    case OpAttrType::UNSUPPORTED_VARIANT:
    case OpAttrType::UNSUPPORTED_QUI8:
    case OpAttrType::UNSUPPORTED_QUI16:
    case OpAttrType::UNSUPPORTED_QI8:
    case OpAttrType::UNSUPPORTED_QI16:
    case OpAttrType::UNSUPPORTED_QI32:
      llvm_unreachable("unsupported attribute type");

#define OP_ATTR_TYPE(ENUM, CPP_TYPE)    \
  case OpAttrType::ENUM:                \
    peak_alignment = alignof(CPP_TYPE); \
    break;
#include "tfrt/core_runtime/op_attr_type.def"
  }
  return llvm::offsetToAlignment(offset + prefix_size,
                                 llvm::Align(peak_alignment));
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
    case OpAttrType::UNSUPPORTED_QUI8:
      return "QUI8";
    case OpAttrType::UNSUPPORTED_QUI16:
      return "QUI16";
    case OpAttrType::UNSUPPORTED_QI8:
      return "QI8";
    case OpAttrType::UNSUPPORTED_QI16:
      return "QI16";
    case OpAttrType::UNSUPPORTED_QI32:
      return "QI32";
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

  bool SetRaw(string_view attr_name, const void *data, OpAttrType type,
              uint32_t element_count, OpAttrsRawEntryType entry_type);

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
    bool success = this->SetRaw(entry.name, entry.GetData(), entry.type,
                                entry.element_count, entry.entry_type);
    assert(success && "input cannot have dupes, so collisions aren't possible");
    (void)success;
  });
}

bool OpAttrs::OutOfLineRepresentation::SetRaw(string_view attr_name,
                                              const void *data, OpAttrType type,
                                              uint32_t element_count,
                                              OpAttrsRawEntryType entry_type) {
  // If element_count > 1, it must be an array.
  assert(element_count <= 1 || entry_type == OpAttrsRawEntryType::kArray ||
         entry_type == OpAttrsRawEntryType::kExternalArray);

  // Figure out what entry we need to fill in.
  auto entry_it_pair =
      entries_.insert(std::make_pair(attr_name, OpAttrsRawEntry()));

  // If it is already present, then return false to indicate a conflict.
  if (entry_it_pair.second == false) return false;

  // TODO(clattner): consider unifying the logic here with OpAttrs::SetRaw.
  auto &entry = entry_it_pair.first->second;
  entry.name = entry_it_pair.first->first().data();
  entry.type = type;
  entry.element_count = element_count;
  entry.entry_type = entry_type;
  entry.is_inlined = false;

  // For an entry with externally allocated buffer, we can simply keep the
  // pointer of the buffer.
  if (entry.IsExternal()) {
    entry.data = data;
    return true;
  }

  const auto type_size_and_alignment = GetHostSizeAndAlignment(data, type);
  const auto type_size = type_size_and_alignment.first;
  const auto type_alignment = type_size_and_alignment.second;
  const auto alignment_padding_size =
      GetAlignmentPaddingSize(data, type, type_alignment);

  const auto payload_size = type_size * element_count;

  // If it is a small scalar, copy the data to the inlined buffer.
  if (payload_size <= sizeof(void *) && alignment_padding_size == 0) {
    assert(type_alignment <= alignof(void *));
    if (payload_size > 0) memcpy(entry.buffer, data, payload_size);
    entry.is_inlined = true;
    return true;
  }

  // TODO(hyojun): Optimize memory allocator for DiamondPacking.
  void *our_data = allocator_.Allocate(alignment_padding_size + payload_size,
                                       type_alignment);
  void *element_ptr = static_cast<char *>(our_data) + alignment_padding_size;

  // Copy the element(s) themselves.
  memcpy(element_ptr, data, payload_size);
  entry.data = element_ptr;
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

bool OpAttrs::SetRaw(string_view attr_name, const void *data, OpAttrType type,
                     uint32_t element_count, OpAttrsRawEntryType entry_type) {
  // If element_count > 1, the entry must be an array.
  assert(element_count <= 1 || entry_type == OpAttrsRawEntryType::kArray ||
         entry_type == OpAttrsRawEntryType::kExternalArray);

  // If we're mutating the set, then drop any frozen representation that may be
  // formed.
  if (frozen_representation_) frozen_representation_.reset();

  // If we are using an out of line representation, delegate to it.
  if (auto *out_of_line = out_of_line_representation_.get())
    return out_of_line->SetRaw(attr_name, data, type, element_count,
                               entry_type);

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
    return out_of_line_representation_->SetRaw(attr_name, data, type,
                                               element_count, entry_type);
  }

  // We also need space in inline_buffer_.
  auto *name_pointer = inline_buffer_ + inline_buffer_used_;
  auto attr_name_size = attr_name.size();
  inline_buffer_used_ += attr_name_size + 1;

  // If we are out of space, then switch to an out-of-line representation.
  if (inline_buffer_used_ > kInlineBufferSize) {
    MoveOutOfLine();
    return out_of_line_representation_->SetRaw(attr_name, data, type,
                                               element_count, entry_type);
  }

  // Otherwise, we have space, so copy over the string.
  memcpy(name_pointer, attr_name.data(), attr_name_size);
  name_pointer[attr_name_size] = 0;  // Null terminate C string.

  auto &entry = inline_entries_[num_inline_entries_++];
  entry.name = name_pointer;
  entry.type = type;
  entry.element_count = element_count;
  entry.entry_type = entry_type;
  entry.is_inlined = false;

  // For an entry with externally allocated buffer, we can simply keep the
  // pointer of the buffer.
  if (entry.IsExternal()) {
    entry.data = data;
    return true;
  }

  const auto type_size_and_alignment = GetHostSizeAndAlignment(data, type);
  const auto type_size = type_size_and_alignment.first;
  const auto type_alignment = type_size_and_alignment.second;
  const auto alignment_padding_size =
      GetAlignmentPaddingSize(data, type, type_alignment);

  const auto payload_size = type_size * element_count;

  // An attribute fits in the inlined buffer.
  if (payload_size <= sizeof(void *) && alignment_padding_size == 0) {
    assert(type_alignment <= alignof(void *));
    if (payload_size > 0) memcpy(entry.buffer, data, payload_size);
    entry.is_inlined = true;
    return true;
  }

  // Otherwise, need to allocate buffer.
  inline_buffer_used_ +=
      GetAlignmentPaddingSize(data, type, inline_buffer_used_);
  auto *dest_pointer = inline_buffer_ + inline_buffer_used_;

  inline_buffer_used_ += payload_size;

  // If we are out of space, then switch to an out-of-line representation.
  if (inline_buffer_used_ > kInlineBufferSize) {
    --num_inline_entries_;
    MoveOutOfLine();
    return out_of_line_representation_->SetRaw(attr_name, data, type,
                                               element_count, entry_type);
  }

  memcpy(dest_pointer, data, payload_size);
  entry.data = dest_pointer;
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

  size_t out_offset = alloc_size;

  // TODO(clattner): When coming from an inlined representation (the vastly most
  // common case, we should be able to memcpy over one big block of memory and
  // update pointers, instead of doing several small memcpy's.

  // Figure out how much space we need for each entry, which is the space for
  // the name and the payload together:
  for (auto *entry : sorted_attrs) {
    // Space for the name and null terminator.
    alloc_size += strlen(entry->name) + 1;

    if (entry->IsInternal()) {
      const auto type_size =
          GetHostSizeAndAlignment(entry->GetData(), entry->type).first;
      alloc_size +=
          GetAlignmentPaddingSize(entry->GetData(), entry->type, alloc_size);
      alloc_size += entry->element_count * type_size;
    }
  }

  // Now that we know the size, create the result.
  auto *raw_memory = AlignedAlloc(alignof(ImmutableOpAttrs), alloc_size);
  auto *result = new (raw_memory) ImmutableOpAttrs(sorted_attrs.size());

  char *data_ptr = static_cast<char *>(raw_memory);

  // Copy all of the attributes over.
  for (size_t i = 0, e = sorted_attrs.size(); i != e; ++i) {
    const auto &src_entry = *sorted_attrs[i];
    auto &result_entry = result->entries_[i];

    // Copy simple properties.
    result_entry.element_count = src_entry.element_count;
    result_entry.entry_type = src_entry.entry_type;
    result_entry.is_inlined = src_entry.is_inlined;
    result_entry.type = src_entry.type;

    // Copy the name over.
    result_entry.name = data_ptr + out_offset;
    auto name_len = strlen(src_entry.name);
    memcpy(data_ptr + out_offset, src_entry.name, name_len + 1);
    out_offset += name_len + 1;

    // For inlined buffer and externally allocated buffer,
    // copying buffer content is enough.
    if (src_entry.IsInlined() || src_entry.IsExternal()) {
      memcpy(result_entry.buffer, src_entry.buffer, sizeof(void *));
      continue;
    }

    // Handle internally allocated buffer case.
    out_offset +=
        GetAlignmentPaddingSize(src_entry.data, src_entry.type, out_offset);

    // Copy over the elements.
    const auto payload_size =
        GetHostSizeAndAlignment(src_entry.data, src_entry.type).first *
        src_entry.element_count;

    memcpy(data_ptr + out_offset, static_cast<const char *>(src_entry.data),
           payload_size);

    // Remember that this is where the element is.
    result_entry.data = data_ptr + out_offset;
    out_offset += payload_size;
  }

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
      if (shape_attr.HasRank())
        llvm::interleave(shape_attr.GetShape(), os, "x");
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
    case OpAttrType::UNSUPPORTED_QUI8:
    case OpAttrType::UNSUPPORTED_QUI16:
    case OpAttrType::UNSUPPORTED_QI8:
    case OpAttrType::UNSUPPORTED_QI16:
    case OpAttrType::UNSUPPORTED_QI32:
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

    if (entry.IsArray()) {
      const char *data = static_cast<const char *>(entry.GetData());
      os << '[';
      auto type_size = GetHostSizeAndAlignment(data, entry.type).first;

      for (size_t i = 0, e = entry.element_count; i != e; ++i) {
        // Only print the first elements of large arrays.
        if (i == 5) {
          os << "...";
          break;
        }
        if (i != 0) os << ", ";
        PrintElement(data + i * type_size, entry.type, os);
      }
      os << ']';
    } else {
      PrintElement(entry.GetData(), entry.type, os);
    }
    os << '\n';
  }
  os << '\n';
}

void OpAttrsRef::Dump() const { Print(llvm::errs()); }

}  // namespace tfrt
