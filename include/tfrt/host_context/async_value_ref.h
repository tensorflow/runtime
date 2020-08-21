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

//===- async_value_ref.h - RCReference<AsyncValue> wrapper ------*- C++ -*-===//
//
// AsyncValueRef<T> is an alias for RCReference<AsyncValue> that carries payload
// type information. The user does not need to pass the payload data type to
// get() or emplace().
//
// Like RCReference<AsyncValue>, it represents one reference on the underlying
// AsyncValue. When a callee returns an AsyncValueRef to a caller, the callee
// also transfers their ownership of a reference on the underlying AsyncValue.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_ASYNC_VALUE_REF_H_
#define TFRT_HOST_CONTEXT_ASYNC_VALUE_REF_H_

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Error.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/location.h"

namespace tfrt {

class ExecutionContext;

// Forward declaration from host_context.h.
template <typename T, typename... Args>
T* HostContextConstruct(HostContext* host, Args&&... args);

template <typename T>
class AsyncValueRef {
 public:
  AsyncValueRef() = default;

  // Support implicit conversion from AsyncValueRef<Derived> to
  // AsyncValueRef<Base>.
  template <typename DerivedT,
            std::enable_if_t<std::is_base_of<T, DerivedT>::value, int> = 0>
  AsyncValueRef(AsyncValueRef<DerivedT>&& u) : value_(u.ReleaseRCRef()) {}

  explicit AsyncValueRef(RCReference<AsyncValue> value)
      : value_(std::move(value)) {}

  // Support implicit conversion from RCReference<AsyncValue>.
  AsyncValueRef(RCReference<ErrorAsyncValue> value)
      : value_(std::move(value)) {}

  AsyncValueRef& operator=(RCReference<ErrorAsyncValue> new_value) {
    value_ = std::move(new_value);
    return *this;
  }

  // Allow implicit conversion to type-erased RCReference<AsyncValue>
  operator RCReference<AsyncValue>() && { return std::move(value_); }

  // Return true if the AsyncValue is resolved to a concrete value or error.
  bool IsAvailable() const { return value_->IsAvailable(); }
  bool IsUnavailable() const { return value_->IsUnavailable(); }

  // Return true if the AsyncValue contains a concrete value.
  bool IsConcrete() const { return value_->IsConcrete(); }

  // Return the stored value. The AsyncValueRef must be available.
  T& get() const { return value_->get<T>(); }

  // Return the stored value as a subclass type. The AsyncValueRef must be
  // available.
  template <typename SubclassT,
            typename = std::enable_if_t<std::is_base_of<T, SubclassT>::value>>
  SubclassT& get() const {
    return value_->get<SubclassT>();
  }

  T* operator->() const { return &get(); }

  T& operator*() const { return get(); }

  // Make the AsyncValueRef available.
  void SetStateConcrete() const { value_->SetStateConcrete(); }

  // Set the stored value. The AsyncValueRef must be unavailable. After this
  // returns, the AsyncValueRef will be available.
  template <typename... Args>
  void emplace(Args&&... args) const {
    value_->emplace<T>(std::forward<Args>(args)...);
  }

  void emplace(Expected<T> v) const {
    if (v) {
      emplace(std::move(*v));
    } else {
      SetError(v.takeError());
    }
  }

  // If the AsyncValueRef is available, run the waiter immediately. Otherwise,
  // run the waiter when the AsyncValueRef becomes available.
  template <typename WaiterT>
  void AndThen(WaiterT&& waiter) const {
    value_->AndThen(std::move(waiter));
  }

  // Return true if this AsyncValueRef represents an error.
  bool IsError() const { return value_->IsError(); }

  // Returns the underlying error. IsError() must be true.
  const DecodedDiagnostic& GetError() const { return value_->GetError(); }

  // Returns the underlying error, or nullptr if there is none.
  const DecodedDiagnostic* GetErrorIfPresent() const {
    return value_->GetErrorIfPresent();
  }

  void SetError(string_view message) const {
    return SetError(DecodedDiagnostic{message});
  }
  void SetError(DecodedDiagnostic diag) const {
    value_->SetError(std::move(diag));
  }

  void SetError(const Error& error) const {
    value_->SetError(DecodedDiagnostic(error));
  }

  explicit operator bool() const { return value_.get() != nullptr; }

  // Return a raw pointer to the AsyncValue.
  AsyncValue* GetAsyncValue() const { return value_.get(); }

  // Return true if this is the only ref to the AsyncValue.
  // This function requires the internal AsyncValue to be set (value_ !=
  // nullptr).
  bool IsUnique() const { return value_->IsUnique(); }

  // Make an explicit copy of this AsyncValueRef, increasing value_'s refcount
  // by one.
  AsyncValueRef<T> CopyRef() const { return AsyncValueRef(CopyRCRef()); }

  // Make a copy of value_, increasing value_'s refcount by one.
  RCReference<AsyncValue> CopyRCRef() const { return value_.CopyRef(); }

  // Release ownership of one reference on the AsyncValue and return a raw
  // pointer to it.
  AsyncValue* release() { return value_.release(); }

  void reset() { value_.reset(); }

  // Transfer ownership of one reference on the AsyncValue to the returned
  // RCReference<AsyncValue>.
  RCReference<AsyncValue> ReleaseRCRef() { return std::move(value_); }

 private:
  RCReference<AsyncValue> value_;
};

// For consistency, the error message should start with a lower case letter
// and not end with a period.
RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            string_view message);

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            llvm::Error error);

// Create a ConcreteAsyncValue in error state for a specified decoded
// diagnostic.
RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(
    HostContext* host, DecodedDiagnostic&& diagnostic);

// Create a ConcreteAsyncValue in error state for a specified error message.
RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(HostContext* host,
                                                    string_view message);

// Allocate an unconstructed AsyncValueRef. The AsyncValueRef should be made
// available later by invoking AsyncValueRef::emplace or
// AsyncValueRef::SetError.
template <typename T>
AsyncValueRef<T> MakeUnconstructedAsyncValueRef(HostContext* host) {
  return AsyncValueRef<T>(
      TakeRef(HostContextConstruct<internal::ConcreteAsyncValue<T>>(
          host, host,
          typename internal::ConcreteAsyncValue<T>::UnconstructedPayload{})));
}

// Allocate and construct an AsyncValueRef without making it available for
// consumption. The AsyncValueRef should be made available later by invoking
// AsyncValueRef::SetStateConcrete or AsyncValueRef::SetError.
template <typename T, typename... Args>
AsyncValueRef<T> MakeConstructedAsyncValueRef(HostContext* host,
                                              Args&&... args) {
  return AsyncValueRef<T>(
      TakeRef(HostContextConstruct<internal::ConcreteAsyncValue<T>>(
          host, host,
          typename internal::ConcreteAsyncValue<T>::ConstructedPayload{},
          std::forward<Args>(args)...)));
}

// Allocate and construct an available AsyncValueRef.
template <typename T, typename... Args>
AsyncValueRef<T> MakeAvailableAsyncValueRef(HostContext* host, Args&&... args) {
  return AsyncValueRef<T>(
      TakeRef(HostContextConstruct<internal::ConcreteAsyncValue<T>>(
          host, host,
          typename internal::ConcreteAsyncValue<T>::ConcretePayload{},
          std::forward<Args>(args)...)));
}

// Construct an empty IndirectAsyncValue, not forwarding to anything.
RCReference<IndirectAsyncValue> MakeIndirectAsyncValue(HostContext* host);

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_ASYNC_VALUE_REF_H_
