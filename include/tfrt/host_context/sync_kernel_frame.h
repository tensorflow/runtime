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

//===- sync_kernel_frame.h - Data for sync kernel invocation ----*- C++ -*-===//
//
// This file implements SyncKernelFrame which captures argument, result, and
// other related information provided to synchronous kernels on kernel
// invocation.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_SYNC_KERNEL_FRAME_H_
#define TFRT_HOST_CONTEXT_SYNC_KERNEL_FRAME_H_

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/value.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/string_util.h"

namespace tfrt {

// SyncKernelFrame captures the states associated with a kernel invocation,
// including the input arguments, attributes, result values, and the execution
// context. SyncKernelFrame is constructed by the kernel caller (currently only
// BEFInterpreter) using the SyncKernelFrameBuilder subclass. The kernel
// implementation is passed a pointer to a SyncKernelFrame object for them to
// access the inputs and attributes, and return result values.
class SyncKernelFrame {
 public:
  explicit SyncKernelFrame(ExecutionContext* exec_ctx) : exec_ctx_{exec_ctx} {}

  const ExecutionContext& GetExecutionContext() const { return *exec_ctx_; }
  HostContext* GetHostContext() const { return exec_ctx_->host(); }

  // Get the location.
  Location GetLocation() const { return exec_ctx_->location(); }

  // Get the number of arguments.
  int GetNumArgs() const { return num_arguments_; }

  // Get the argument at the given index as type T.
  template <typename T>
  T& GetArgAt(int index) const {
    return GetArgAt(index)->get<T>();
  }

  // Get the argument at the given index as Value*.
  Value* GetArgAt(int index) const {
    assert(index < GetNumArgs());
    return value_or_attrs_[index].value;
  }

  // Get all arguments.
  ArrayRef<Value*> GetArguments() const { return GetValues(0, num_arguments_); }

  // Get all attributes.
  ArrayRef<const void*> GetAttributes() const {
    if (value_or_attrs_.empty()) return {};

    return llvm::makeArrayRef(
        &value_or_attrs_[num_arguments_ + num_results_].attr,
        GetNumAttributes());
  }

  // Get the number of attributes.
  int GetNumAttributes() const {
    return value_or_attrs_.size() - num_arguments_ - num_results_;
  }

  // Get the attribute at the given index as type T.
  // TODO(jingdong): Disable const char*.
  template <typename T>
  Attribute<T> GetAttributeAt(int index) const {
    assert(index < GetNumAttributes());
    return Attribute<T>(GetAttributes()[index]);
  }

  AggregateAttr GetAggregateAttr(int index) const {
    assert(index < GetNumAttributes());
    return AggregateAttr(GetAttributes()[index]);
  }

  // Get the array attribute at the given index as type T.
  template <typename T>
  ArrayAttribute<T> GetArrayAttributeAt(int index) const {
    assert(index < GetNumAttributes());
    return ArrayAttribute<T>(GetAttributes()[index]);
  }

  // Get array attribute as a string. Equivalent to
  // GetArrayAttributeAt<char>, except that this returns StringRef instead
  // of ArrayRef<char>.
  StringAttribute GetStringAttribute(int index) const {
    return StringAttribute(GetAttributes()[index]);
  }

  // Get the number of results.
  int GetNumResults() const { return num_results_; }

  // Emplace construct the result at given index.
  template <typename T, typename... Args>
  void EmplaceResultAt(int index, Args&&... args) {
    assert(index < num_results_ && "Invalid result index");
    Value* result = value_or_attrs_[num_arguments_ + index].value;
    assert(!result->HasValue() && "Result value is non-empty.");
    result->emplace<T>(std::forward<Args>(args)...);
  }

  // Get all results as an immutable ArrayRef.
  ArrayRef<Value*> GetResults() const {
    return GetValues(num_arguments_, num_results_);
  }

  // Assert the size of arguments, attributes, and results are as expected.
  void AssertArity(int num_arguments, int num_attributes,
                   int num_results) const;

  // Report error from the kernel execution.
  void SetError(Error error) {
    assert(!error_ && "Error is already set.");
    error_ = std::move(error);
  }

  // This should only be called once.
  Error TakeError() { return std::move(error_); }

 protected:
  union ValueOrAttribute {
    explicit ValueOrAttribute(Value* value) : value{value} {}
    explicit ValueOrAttribute(const void* attr) : attr{attr} {}

    Value* value;
    const void* attr;
  };

  ArrayRef<Value*> GetValues(size_t from, size_t length) const {
    assert((from + length) <= (num_arguments_ + num_results_));

    if (length == 0) return {};

    return llvm::makeArrayRef(&(value_or_attrs_[from].value), length);
  }

  MutableArrayRef<Value*> GetMutableValues(size_t from, size_t length) {
    assert((from + length) <= (num_arguments_ + num_results_));

    if (length == 0) return {};

    return llvm::makeMutableArrayRef(&(value_or_attrs_[from].value), length);
  }

  // This SmallVector stores the kernel argument Values, result Values, and
  // attributes in order.
  SmallVector<ValueOrAttribute, 8> value_or_attrs_;
  int num_arguments_ = 0;
  int num_results_ = 0;
  ExecutionContext* exec_ctx_ = nullptr;
  Error error_ = Error::success();
};

// SyncKernelFrameBuilder is used by the kernel caller to construct a
// SyncKernelFrame object without exposing the builder methods to the kernel
// implementation.
//
// As an optimization, SyncKernelFrame stores arguments, attributes, and results
// in a single SmallVector. As a result, to initialize a SyncKernelFrame, this
// class requires that the client performs the following actions in order:
// 1. Adds the arguments (using AddArg())
// 2. Add the results (using AddResult())
// 3. Add the attributes (using AddAttribute())
class SyncKernelFrameBuilder : public SyncKernelFrame {
 public:
  explicit SyncKernelFrameBuilder(ExecutionContext* exec_ctx)
      : SyncKernelFrame{exec_ctx} {}

  // Get result Value at the given index.
  Value* GetResultAt(int index) const { return GetResults()[index]; }

  // Add a new argument to the SyncKernelFrame.
  void AddArg(Value* value) {
    assert(num_results_ == 0 && "Must call AddArg before calling AddResult");
    value_or_attrs_.emplace_back(value);
    ++num_arguments_;
  }

  // Add a new result to the SyncKernelFrame.
  void AddResult(Value* value) {
    assert(GetNumAttributes() == 0 &&
           "Must call AddResult before calling AddAttribute");
    value_or_attrs_.emplace_back(value);
    ++num_results_;
  }

  // Add a new attribute to the SyncKernelFrame.
  void AddAttribute(const void* attr) { value_or_attrs_.emplace_back(attr); }

  // Set the location.
  void SetLocation(const Location& location) {
    exec_ctx_->set_location(location);
  }

  // Clear all fields.
  void Reset() {
    value_or_attrs_.clear();
    num_arguments_ = 0;
    num_results_ = 0;
  }
};

// Implementation details

inline void SyncKernelFrame::AssertArity(int num_arguments, int num_attributes,
                                         int num_results) const {
  assert(num_arguments_ == num_arguments);
  assert(GetNumAttributes() == num_attributes);
  assert(GetNumResults() == num_results);
}

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_KERNEL_FRAME_H_
