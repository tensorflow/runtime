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

//===- async_value.cc - Generic future type used by HostContext -----------===//
//
// This file implements AsyncValue.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/concurrent_vector.h"
#include "tfrt/support/string_util.h"

namespace tfrt {

// This is a singly linked list of nodes waiting for notification, hanging off
// of AsyncValue.  When the value becomes available or if an error occurs, the
// callbacks are informed.
class NotifierListNode {
 public:
  explicit NotifierListNode(llvm::unique_function<void()> notification)
      : next_(nullptr), notification_(std::move(notification)) {}

 private:
  friend class AsyncValue;
  // This is the next thing waiting on the AsyncValue.
  NotifierListNode* next_;
  llvm::unique_function<void()> notification_;
};

/*static*/ uint16_t AsyncValue::CreateTypeInfoAndReturnTypeIdImpl(
    Destructor destructor) {
  TypeInfo type_info{destructor};
  size_t type_id = GetTypeInfoTableSingleton()->emplace_back(type_info) + 1;
  // Detect overflow.
  assert(type_id < std::numeric_limits<uint16_t>::max() &&
         "Too many different AsyncValue types.");
  return type_id;
}

AsyncValue::TypeInfoTable* AsyncValue::GetTypeInfoTableSingleton() {
  const int kInitialCapacity = 64;
  static auto* type_info_table = new TypeInfoTable(kInitialCapacity);
  return type_info_table;
}

std::atomic<ssize_t> AsyncValue::total_allocated_async_values_;

const AsyncValue::TypeInfo& AsyncValue::GetTypeInfo() const {
  TypeInfoTable* type_info_table = AsyncValue::GetTypeInfoTableSingleton();
  assert(type_id_ != 0);

  // TODO(sanjoy): Once ConcurentVector supports it, we should check that
  // type_id_ - 1 is within range.
  return (*type_info_table)[type_id_ - 1];
}

void AsyncValue::Destroy() {
  if (kind() == Kind::kIndirect) {
    // Depending on what the benchmarks say, it might make sense to remove this
    // explicit check and instead make ~IndirectAsyncValue go through the
    // GetTypeInfo().destructor case below.
    static_cast<IndirectAsyncValue*>(this)->~IndirectAsyncValue();
    GetHostContext()->DeallocateBytes(this, sizeof(IndirectAsyncValue));
    return;
  }

  auto size = GetTypeInfo().destructor(this, /*destroys_object=*/true);
  GetHostContext()->DeallocateBytes(this, size);
}

// This is called when the value is set into the ConcreteAsyncValue buffer, or
// when the IndirectAsyncValue is forwarded to an available AsyncValue, and we
// need to change our state and clear out the notifications. The current state
// must be unavailable (i.e. kUnconstructed or kConstructed).
void AsyncValue::NotifyAvailable(State available_state) {
  assert((kind() == Kind::kConcrete || kind() == Kind::kIndirect) &&
         "Should only be used by ConcreteAsyncValue or IndirectAsyncValue");

  assert(available_state == State::kConcrete ||
         available_state == State::kError);

  // Mark the value as available, ensuring that new queries for the state see
  // the value that got filled in.
  auto old_value = waiters_and_state_.exchange(
      WaitersAndState(nullptr, available_state), std::memory_order_acq_rel);
  assert(old_value.getInt() == State::kUnconstructed ||
         old_value.getInt() == State::kConstructed);

  RunWaiters(old_value.getPointer());
}

void AsyncValue::RunWaiters(NotifierListNode* list) {
  HostContext* host = GetHostContext();
  while (list) {
    auto* node = list;
    // TODO(chky): pass state into notification_ so that waiters do not need to
    // check atomic state again.
    node->notification_();
    list = node->next_;
    host->Destruct(node);
  }
}

// If the value is available or becomes available, this calls the closure
// immediately. Otherwise, the add closure to the waiter list where it will be
// called when the value becomes available.
void AsyncValue::EnqueueWaiter(llvm::unique_function<void()>&& waiter,
                               WaitersAndState old_value) {
  // Create the node for our waiter.
  auto* node = GetHostContext()->Construct<NotifierListNode>(std::move(waiter));
  auto old_state = old_value.getInt();

  // Swap the next link in. old_value.getInt() must be unavailable when
  // evaluating the loop condition. The acquire barrier on the compare_exchange
  // ensures that prior changes to waiter list are visible here as we may call
  // RunWaiter() on it. The release barrier ensures that prior changes to *node
  // appear to happen before it's added to the list.
  node->next_ = old_value.getPointer();
  auto new_value = WaitersAndState(node, old_state);
  while (!waiters_and_state_.compare_exchange_weak(old_value, new_value,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
    // While swapping in our waiter, the value could have become available.  If
    // so, just run the waiter.
    if (old_value.getInt() == State::kConcrete ||
        old_value.getInt() == State::kError) {
      assert(old_value.getPointer() == nullptr);
      node->next_ = nullptr;
      RunWaiters(node);
      return;
    }
    // Update the waiter list in new_value.
    node->next_ = old_value.getPointer();
  }

  // compare_exchange_weak succeeds. The old_value must be in either
  // kUnconstructed or kConstructed state.
  assert(old_value.getInt() == State::kUnconstructed ||
         old_value.getInt() == State::kConstructed);
}

void AsyncValue::SetError(DecodedDiagnostic diag_in) {
  auto s = state();
  assert(s == State::kUnconstructed || s == State::kConstructed);

  if (kind() == Kind::kConcrete) {
    if (s == State::kConstructed) {
      // ~AsyncValue erases type_id_ and makes a few assertion on real
      // destruction, but this AsyncValue is still alive.
      GetTypeInfo().destructor(this, /*destroys_object=*/false);
    }
    char* this_ptr = reinterpret_cast<char*>(this);
    auto& error = *reinterpret_cast<DecodedDiagnostic**>(
        this_ptr + AsyncValue::kDataOrErrorOffset);
    error = new DecodedDiagnostic(std::move(diag_in));
    NotifyAvailable(State::kError);
  } else {
    assert(kind() == Kind::kIndirect);
    auto error_av =
        MakeErrorAsyncValueRef(host_context_.get(), std::move(diag_in));
    cast<IndirectAsyncValue>(this)->ForwardTo(std::move(error_av));
  }
}

void AsyncValue::SetErrorLocationIfUnset(DecodedLocation location) {
  auto& diag = const_cast<DecodedDiagnostic&>(GetError());
  if (!diag.location) diag.location = std::move(location);
}

// Mark this IndirectAsyncValue as forwarding to the specified value.  This
// gives the IndirectAsyncValue a +1 reference.
void IndirectAsyncValue::ForwardTo(RCReference<AsyncValue> value) {
  assert(IsUnavailable());

  auto s = value->state();
  if (s == State::kConcrete || s == State::kError) {
    assert(!value_ && "IndirectAsyncValue::ForwardTo is called more than once");
    auto* concrete_value = value.release();
    if (auto* indirect_value = dyn_cast<IndirectAsyncValue>(concrete_value)) {
      concrete_value = indirect_value->value_;
      assert(concrete_value != nullptr);
      assert(concrete_value->kind() == Kind::kConcrete);
      concrete_value->AddRef();
      indirect_value->DropRef();
    }
    value_ = concrete_value;
    type_id_ = concrete_value->type_id_;
    NotifyAvailable(s);
  } else {
    // Copy value here because the evaluation order of
    // value->AndThen(std::move(value)) is not defined prior to C++17.
    AsyncValue* value2 = value.get();
    value2->AndThen(
        [this2 = FormRef(this), value2 = std::move(value)]() mutable {
          this2->ForwardTo(std::move(value2));
        });
  }
}

}  // namespace tfrt
