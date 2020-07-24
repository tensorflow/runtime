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

//===- async_value.h - Generic future type used by HostContext --*- C++ -*-===//
//
// This file declares AsyncValue, a generic "future" type that can be fulfilled
// by an asynchronously provided value or an error.  This is called AsyncValue
// instead of 'future' because there are many different types of futures
// (including std::future) which do not share the same characteristics of this
// type.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_ASYNC_VALUE_H_
#define TFRT_HOST_CONTEXT_ASYNC_VALUE_H_

#include <atomic>
#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>

#include "llvm/ADT/PointerIntPair.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_context_ptr.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/type_traits.h"

namespace tfrt {

namespace internal {
template <typename T>
class ConcreteAsyncValue;
}
class HostContext;
class NotifierListNode;

namespace internal {

template <typename T>
constexpr bool kMaybeBase = std::is_class<T>::value && !std::is_final<T>::value;

}  // namespace internal

// This is a future of the specified value type. Arbitrary C++ types may be used
// here, even non-copyable types and expensive ones like tensors.
//
// An AsyncValue is in one of two states: unavailable or available. If it is in
// the unavailable state, it may have a list of waiters which are notified when
// the value transitions to another state.
//
// The actual payload data is stored in the templated subclass
// ConcreteAsyncValue. This achieves good cache efficiency by storing the meta
// data and the payload data in consecutive memory locations.
class AsyncValue {
 public:
  ~AsyncValue();

  // TODO: Rename Is* methods to be more consistent with the states. For example
  // we can rename them to IsError(), IsConcrete(),
  // IsUnavailableIndirect().

  // Return true if state is kUnconstructed.
  bool IsUnconstructed() const;

  // Return true if state is kConstructed.
  bool IsConstructed() const;

  // Return true if state is kConcrete.
  bool IsConcrete() const;

  // Return true if state is kError.
  bool IsError() const;

  // Return true if this async value is resolved to a concrete value or error.
  bool IsAvailable() const;
  bool IsUnavailable() const { return !IsAvailable(); }

  // Return true if this is an IndirectAsyncValue that hasn't been resolved.
  // Currently an IndirectAsyncValue is available if and only if it is resolved.
  bool IsUnresolvedIndirect() const;

  // Return true if reference count is 1.
  bool IsUnique() const { return refcount_.load() == 1; }

  // Add a new reference to this object.
  //
  // Return this object for convenience. e.g. when a call-site wants to feed an
  // AsyncValue object into a function call that takes the object at +1, we can
  // write: foo(av->AddRef());
  AsyncValue* AddRef() { return AddRef(1); }
  AsyncValue* AddRef(uint32_t count);

  // Drop a reference to this object, potentially deallocating it.
  void DropRef() { DropRef(1); }
  void DropRef(uint32_t count);

  // Return the stored value as type T. Requires that the state be constructed
  // or concrete and that T be the exact type or a base type of the payload type
  // of this AsyncValue. When T is a base type of the payload type, the
  // following additional conditions are required:
  // 1) Both the payload type and T are polymorphic (have virtual function) or
  //    neither are.
  // 2) The payload type does not use multiple inheritance.
  // The above conditions are required since we store only the offset of the
  // payload type in AsyncValue as data_traits_.buf_offset. Violation of either
  // 1) or 2) requires additional pointer adjustments to get the proper pointer
  // for the base type, which we do not have sufficient information to perform
  // at runtime.
  template <typename T>
  const T& get() const;

  // Same as the const overload of get(), except for returning a non-const ref.
  template <typename T>
  T& get();

  // Returns the underlying error. IsError() must be true.
  const DecodedDiagnostic& GetError() const;

  // Returns the underlying error, or nullptr if there is none.
  const DecodedDiagnostic* GetErrorIfPresent() const;

  // Set the error location if unset. IsError() must be true.
  void SetErrorLocationIfUnset(DecodedLocation location);

  template <typename T>
  bool IsType() const {
    return GetTypeId<T>() == type_id_;
  }

  // Change state to kConcrete. Requires that this AsyncValue
  // previously have state constructed.
  void SetStateConcrete();

  // Construct the payload of the AsyncValue in place and change its state to
  // kConcrete. Requires that this AsyncValue previously have state
  // kUnconstructed or kConstructed.
  template <typename T, typename... Args>
  void emplace(Args&&... args);

  void SetError(DecodedDiagnostic diag);

  // If the value is available or becomes available, this calls the closure
  // immediately.  Otherwise, adds the waiter to the waiter list and calls it
  // when the value becomes available.
  template <typename WaiterT>
  void AndThen(WaiterT&& waiter);

  /// Return the total number of async values that are currently live in the
  /// process. This is intended for debugging/assertions only, and shouldn't be
  /// used for mainline logic in the runtime.
  static ssize_t GetNumAsyncValueInstances() {
    assert(AsyncValueAllocationTrackingEnabled() &&
           "AsyncValue instance tracking disabled!");
    return total_allocated_async_values_.load(std::memory_order_relaxed);
  }

  /// Returns true if we track the number of alive AsyncValue instances in
  /// total_allocated_async_values_.
  static bool AsyncValueAllocationTrackingEnabled() {
    // For now we track the number of alive AsyncValue instances only in debug
    // builds.
#ifdef NDEBUG
    return false;
#else
    return true;
#endif
  }

  // What sort of AsyncValue this is.
  //
  // We make this an unsigned type so that loading the enum from the bitfield
  // does not sign extend.
  enum class Kind : uint8_t {
    kConcrete = 0,  // ConcreteAsyncValue
    kIndirect = 1,  // IndirectAsyncValue
  };

  // Return the kind of this AsyncValue.
  Kind kind() const { return kind_; }

  class State {
   public:
    // The state of AsyncValue.
    enum StateEnum : int8_t {
      // The underlying value's constructor is not called and the value is not
      // available for consumption. This state can transition to kConstructed,
      // kConcrete and kError.
      kUnconstructed = 0,
      // The underlying value's constructor is called but the value is not
      // available for consumption. This state can transition to
      // kConcrete and kError.
      kConstructed = 1,
      // The underlying value is available for consumption. This state can not
      // transition to any other state.
      kConcrete = 2,
      // This AsyncValue is available and contains an error. This state can not
      // transition to any other state.
      kError = 3,
    };

    /* implicit */ State(StateEnum s) : state_(s) {}

    /* implicit */ operator StateEnum() { return state_; }

    // Return true if state is kUnconstructed.
    bool IsUnconstructed() const { return state_ == kUnconstructed; }

    // Return true if state is kConstructed.
    bool IsConstructed() const { return state_ == kConstructed; }

    // Return true if state is kConcrete.
    bool IsConcrete() const { return state_ == kConcrete; }

    // Return true if state is kError.
    bool IsError() const { return state_ == kError; }

    // Return true if this async value is resolved to a concrete value or error.
    bool IsAvailable() const { return state_ == kConcrete || state_ == kError; }
    bool IsUnavailable() const { return !IsAvailable(); }

    const char* DebugString() const {
      switch (state_) {
        case kUnconstructed:
          return "kUnconstructed";
        case kConstructed:
          return "kConstructed";
        case kConcrete:
          return "kConcrete";
        case kError:
          return "kError";
      }
    }

   private:
    StateEnum state_;
  };

  // Return which state this AsyncValue is in.
  State state() const {
    return waiters_and_state_.load(std::memory_order_acquire).getInt();
  }

 protected:
  // -----------------------------------------------------------
  // Implementation details follow.  Clients should ignore them.

  friend class HostContext;
  friend class IndirectAsyncValue;
  // Destructor returns the size of the derived AsyncValue to be deallocated.
  // The second bool argument indicates whether to destruct the AsyncValue
  // object or simply destroy the payloads.
  using Destructor = size_t (*)(AsyncValue*, bool);

  template <typename T>
  AsyncValue(HostContextPtr host, Kind kind, State state, TypeTag<T>)
      : host_context_(host),
        kind_(kind),
        has_vtable_(std::is_polymorphic<T>()),
        type_id_(GetTypeId<T>()),
        waiters_and_state_(WaitersAndState(nullptr, state)) {
    if (AsyncValueAllocationTrackingEnabled())
      total_allocated_async_values_.fetch_add(1, std::memory_order_relaxed);
  }

  AsyncValue(HostContextPtr host, Kind kind, State state)
      : host_context_(host),
        kind_(kind),
        has_vtable_(false),
        type_id_(0),
        waiters_and_state_(WaitersAndState(nullptr, state)) {
    if (AsyncValueAllocationTrackingEnabled())
      total_allocated_async_values_.fetch_add(1, std::memory_order_relaxed);
  }

  AsyncValue(const AsyncValue&) = delete;
  AsyncValue& operator=(const AsyncValue&) = delete;

  HostContext* GetHostContext() const { return host_context_.get(); }
  HostContextPtr GetHostContextPtr() const { return host_context_; }

  void NotifyAvailable(State available_state);
  void Destroy();
  void RunWaiters(NotifierListNode* list);

  // IsTypeIdCompatible returns true if the type value stored in this AsyncValue
  // instance can be safely cast to `T`. This is a conservative check. I.e.
  // IsTypeIdCompatible may return true even if the value cannot be safely cast
  // to `T`. However, if it returns false then the value definitely cannot be
  // safely cast to `T`. This means it is useful mainly as a debugging aid for
  // use in assert() etc.

  template <typename T,
            typename std::enable_if<internal::kMaybeBase<T>>::type* = nullptr>
  bool IsTypeIdCompatible() const {
    // We can't do a GetTypeId<T> in this case because `T` might be an abstract
    // class.  So we conservatively return true.
    return true;
  }

  template <typename T,
            typename std::enable_if<!internal::kMaybeBase<T>>::type* = nullptr>
  bool IsTypeIdCompatible() const {
    return GetTypeId<T>() == type_id_;
  }

  // Return the ID of the given type. Note that at most 2^16-2 (approx. 64K)
  // unique types can be used in AsyncValues, since the ID is 16 bits, and 0 and
  // 2^16-1 are not allowed to be used as type IDs.
  template <typename T>
  static uint16_t GetTypeId() {
    return internal::ConcreteAsyncValue<T>::concrete_type_id_;
  }

  // Creates a AsyncValue::TypeInfo object for `T` and store it in the global
  // TypeInfo table. Returns the "type id" for `T` which currently happens to
  // be one plus the index of this TypeInfo object in the TypeInfo table.
  //
  // This should only be called from the initializer for the static
  // ConcreteAsyncValue concreate_type_id_ field.
  template <typename T>
  static uint16_t CreateTypeInfoAndReturnTypeId() {
    return CreateTypeInfoAndReturnTypeIdImpl(
        DestructorFn<internal::ConcreteAsyncValue<T>>);
  }

  static uint16_t CreateTypeInfoAndReturnTypeIdImpl(Destructor destructor);

  std::atomic<uint32_t> refcount_{1};
  const HostContextPtr host_context_;

  Kind kind_ : 2;
  // has_vtable_ has the same value for a given payload type T. If we want to
  // use the unused bits here for other purpose in the future, we can move
  // has_vtable_ to a global vector<bool> indexed by type_id_.
  const bool has_vtable_ : 1;

  // Unused padding bits.
  unsigned unused_ : 5;

  // This is a 16-bit value that identifies the type.
  uint16_t type_id_ = 0;

  struct NotifierListNodePointerTraits {
    static inline void* getAsVoidPointer(NotifierListNode* ptr) { return ptr; }
    static inline NotifierListNode* getFromVoidPointer(void* ptr) {
      return static_cast<NotifierListNode*>(ptr);
    }
    // NotifierListNode has an alignment of at least alignof(void*).
    enum {
      NumLowBitsAvailable =
          llvm::PointerLikeTypeTraits<void**>::NumLowBitsAvailable
    };
  };

  // The waiter list and the state are compacted into one single atomic word as
  // accesses to them are tightly related. To change the state from unavailable
  // (i.e. kUnconstructed or kConstructed) to available
  // (i.e. kConcrete or kError), we also need to empty the waiter
  // list. To add a node to the waiter list, we want to make sure the state is
  // unavailable, otherwise we could run the new node immediately.
  //
  // Invariant: If the state is not available, then the waiter list must be
  // nullptr.
  using WaitersAndState =
      llvm::PointerIntPair<NotifierListNode*, 2, State::StateEnum,
                           NotifierListNodePointerTraits>;
  std::atomic<WaitersAndState> waiters_and_state_;

  /// We assume (and static_assert) that this is the offset of
  /// ConcreteAsyncValue::data_, which is the same as the offset of
  /// ConcreteAsyncValue::error_.
  static constexpr int kDataOrErrorOffset = 16;

 private:
  template <typename T>
  const T& GetConcreteValue() const;

  // The destructor function for a derived AsyncValue. The `destroys_object`
  // argument indicates whether to destruct the AsyncValue object or simply
  // destroy the payloads.
  template <typename Derived>
  static size_t DestructorFn(AsyncValue* v, bool destroys_object) {
    if (destroys_object) {
      static_cast<Derived*>(v)->~Derived();
    } else {
      static_cast<Derived*>(v)->Destroy();
    }
    return sizeof(Derived);
  }

  // Information about a ConcreteAsyncValue<T> subclass.
  struct TypeInfo {
    Destructor destructor;
  };

  static_assert(sizeof(TypeInfo) == 8 || sizeof(void*) != 8,
                "We force sizeof(TypeInfo) to be 8 bytes so that x86 complex"
                "addressing modes can address into TypeInfoTable");

  // Get the TypeInfo instance for this AsyncValue.
  const TypeInfo& GetTypeInfo() const;

  using TypeInfoTable = ConcurrentVector<TypeInfo>;

  // Returns the TypeInfoTable instance (there is one per process).
  static TypeInfoTable* GetTypeInfoTableSingleton();

  void EnqueueWaiter(llvm::unique_function<void()>&& waiter,
                     WaitersAndState old_value);

  /// This is a global counter of the number of AsyncValue instances currently
  /// live in the process.  This is intended to be used for debugging only, and
  /// is only kept in sync if AsyncValueAllocationTrackingEnabled() returns
  /// true.
  static std::atomic<ssize_t> total_allocated_async_values_;
};

// We only optimize the code for 64-bit architectures for now.
static_assert(sizeof(AsyncValue) == 16 || sizeof(void*) != 8,
              "Unexpected size for AsyncValue");

namespace internal {
// Subclass for storing the payload of the AsyncValue
template <typename T>
class ConcreteAsyncValue : public AsyncValue {
 public:
  // Tag type for making a ConcreteAsyncValue without calling underlying value's
  // constructor.
  struct UnconstructedPayload {};

  // Tag type for making a ConcreteAsyncValue with the underlying value
  // constructed but not available for consumption.
  struct ConstructedPayload {};

  // Tag type for making a ConcreteAsyncValue with the underlying value
  // constructed and available for consumption.
  struct ConcretePayload {};

  // Make a ConcreteAsyncValue with kUnconstructed state.
  explicit ConcreteAsyncValue(HostContextPtr host, UnconstructedPayload)
      : AsyncValue(host, Kind::kConcrete, State::kUnconstructed, TypeTag<T>()) {
    VerifyOffsets();
  }

  // Make a ConcreteAsyncValue with kError state.
  explicit ConcreteAsyncValue(HostContextPtr host, DecodedDiagnostic diagnostic)
      : AsyncValue(host, Kind::kConcrete, State::kError, TypeTag<T>()) {
    error_ = new DecodedDiagnostic(std::move(diagnostic));
    VerifyOffsets();
  }

  // Make a ConcreteAsyncValue with kConstructed state.
  template <typename... Args>
  explicit ConcreteAsyncValue(HostContextPtr host, ConstructedPayload,
                              Args&&... args)
      : AsyncValue(host, Kind::kConcrete, State::kConstructed, TypeTag<T>()) {
    new (&data_) T(std::forward<Args>(args)...);
  }

  // Make a ConcreteAsyncValue with kConcrete state.
  template <typename... Args>
  explicit ConcreteAsyncValue(HostContextPtr host, ConcretePayload,
                              Args&&... args)
      : AsyncValue(host, Kind::kConcrete, State::kConcrete, TypeTag<T>()) {
    new (&data_) T(std::forward<Args>(args)...);
  }

  ~ConcreteAsyncValue() { Destroy(); }

  // Return the underlying error. IsError() must return true.
  const DecodedDiagnostic& GetError() const {
    assert(IsError());
    return *error_;
  }

  void SetError(DecodedDiagnostic diag_in) {
    auto s = state();
    assert(s == State::kUnconstructed || s == State::kConstructed);
    if (s == State::kConstructed) {
      data_.~T();
    }
    error_ = new DecodedDiagnostic(std::move(diag_in));
    NotifyAvailable(State::kError);
  }

  const T& get() const {
    assert(IsConcrete());
    return data_;
  }

  T& get() {
    assert(IsConcrete());
    return data_;
  }

  // Construct the payload of the AsyncValue in place and change its state to
  // kConcrete. Requires that this AsyncValue previously have state unavailable.
  template <typename... Args>
  void emplace(Args&&... args) {
    new (&data_) T(std::forward<Args>(args)...);
    NotifyAvailable(State::kConcrete);
  }

  static bool classof(const AsyncValue* v) {
    return v->kind() == AsyncValue::Kind::kConcrete;
  }

 private:
  friend class AsyncValue;

  union {
    DecodedDiagnostic* error_;
    T data_;
  };

  void Destroy() {
    auto s = state();
    if (s == State::kError) {
      delete error_;
    } else if (s == State::kConstructed || s == State::kConcrete) {
      data_.~T();
    }
  }

  static void VerifyOffsets() {
    static_assert(offsetof(ConcreteAsyncValue<T>, data_) ==
                      AsyncValue::kDataOrErrorOffset,
                  "Offset of ConcreteAsyncValue::data_ is assumed to be "
                  "AsyncValue::kDataOrErrorOffset == 16");
    static_assert(offsetof(ConcreteAsyncValue<T>, error_) ==
                      AsyncValue::kDataOrErrorOffset,
                  "Offset of ConcreteAsyncValue::error_ is assumed to be "
                  "AsyncValue::kDataOrErrorOffset == 16");
  }

  static const uint16_t concrete_type_id_;
};

template <typename T>
const uint16_t ConcreteAsyncValue<T>::concrete_type_id_ =
    AsyncValue::CreateTypeInfoAndReturnTypeId<T>();
}  // namespace internal

struct DummyValueForErrorAsyncValue {};

class ErrorAsyncValue
    : public internal::ConcreteAsyncValue<DummyValueForErrorAsyncValue> {
 public:
  ErrorAsyncValue(HostContextPtr host, DecodedDiagnostic diagnostic)
      : internal::ConcreteAsyncValue<DummyValueForErrorAsyncValue>(
            host, std::move(diagnostic)) {}
};

// IndirectAsyncValue represents an uncomputed AsyncValue of unspecified kind
// and type. IndirectAsyncValue is used when an AsyncValue must be returned,
// but the value it holds is not ready and the producer of the value doesn't
// know what type it will ultimately be, or whether it will be an error.
class IndirectAsyncValue : public AsyncValue {
  friend class AsyncValue;

 public:
  explicit IndirectAsyncValue(HostContextPtr host)
      : AsyncValue(host, Kind::kIndirect, State::kUnconstructed) {}

  IndirectAsyncValue* AddRef() { return AddRef(1); }
  IndirectAsyncValue* AddRef(uint32_t count) {
    return static_cast<IndirectAsyncValue*>(AsyncValue::AddRef(count));
  }

  // Mark this IndirectAsyncValue as forwarding to the specified value. This
  // gives the IndirectAsyncValue a +1 reference.
  // This method must be called at most once.
  void ForwardTo(RCReference<AsyncValue> value);

  static bool classof(const AsyncValue* v) {
    return v->kind() == AsyncValue::Kind::kIndirect;
  }

 private:
  ~IndirectAsyncValue() { Destroy(); }

  void Destroy() {
    if (value_) {
      value_->DropRef();
      value_ = nullptr;
    }
  }

  // TODO(b/139835587): Use LSB to remember if ForwardTo was called and assert
  // it.
  AsyncValue* value_ = nullptr;
};

// -----------------------------------------------------------
// Implementation details follow.  Clients should ignore them.
//
inline AsyncValue::~AsyncValue() {
  assert(waiters_and_state_.load().getPointer() == nullptr &&
         "An async value with waiters should never have refcount of zero");
  if (AsyncValueAllocationTrackingEnabled())
    total_allocated_async_values_.fetch_sub(1, std::memory_order_relaxed);

  // Catch use-after-free errors more eagerly, by triggering the size assertion
  // in the 'get' accessor.
  type_id_ = ~0;
}

inline bool AsyncValue::IsAvailable() const {
  auto s = state();
  return s == State::kConcrete || s == State::kError;
}

inline bool AsyncValue::IsError() const { return state() == State::kError; }

inline bool AsyncValue::IsUnconstructed() const {
  return state() == State::kUnconstructed;
}

inline bool AsyncValue::IsConstructed() const {
  return state() == State::kConstructed;
}

inline bool AsyncValue::IsConcrete() const {
  return state() == State::kConcrete;
}

// Return true if this is an IndirectAsyncValue that hasn't been resolved.
// Currently an IndirectAsyncValue is available if and only if it is resolved.
inline bool AsyncValue::IsUnresolvedIndirect() const {
  return IsUnavailable() && (kind() == Kind::kIndirect);
}

inline AsyncValue* AsyncValue::AddRef(uint32_t count) {
  if (count > 0) {
    assert(refcount_.load() > 0);
    // Increasing the reference counter can always be done with
    // memory_order_relaxed: New references to an object can only be formed from
    // an existing reference, and passing an existing reference from one thread
    // to another must already provide any required synchronization.
    refcount_.fetch_add(count, std::memory_order_relaxed);
  }
  return this;
}

inline void AsyncValue::DropRef(uint32_t count) {
  assert(refcount_.load() > 0);
  // We expect that `count` argument will often equal the actual reference count
  // here; optimize for that.
  // If `count` == reference count, only an acquire barrier is needed
  // to prevent the effects of the deletion from leaking before this point.
  auto is_last_ref = refcount_.load(std::memory_order_acquire) == count;
  if (!is_last_ref) {
    // If `count` != reference count, a release barrier is needed in
    // addition to an acquire barrier so that prior changes by this thread
    // cannot be seen to occur after this decrement.
    is_last_ref =
        refcount_.fetch_sub(count, std::memory_order_acq_rel) == count;
  }
  // Destroy this value if the refcount drops to zero.
  if (is_last_ref) {
    Destroy();
  }
}

template <typename T>
const T& AsyncValue::GetConcreteValue() const {
  // Make sure both T (the stored type) and BaseT have vtable_ptr or
  // neither have the vtable_ptr.
  assert(std::is_polymorphic<T>::value == has_vtable_);
  assert(IsTypeIdCompatible<T>() && "Incorrect accessor");

  const char* this_ptr = reinterpret_cast<const char*>(this);
  return *reinterpret_cast<const T*>(this_ptr + AsyncValue::kDataOrErrorOffset);
}

template <typename T>
const T& AsyncValue::get() const {
  auto s = state();
  (void)s;

  switch (kind()) {
    case Kind::kConcrete:
      TFRT_DLOG_IF(FATAL, s != State::kConstructed && s != State::kConcrete)
          << "Cannot call get() when ConcreteAsyncValue isn't "
             "constructed; state: "
          << s;
      return GetConcreteValue<T>();
    case Kind::kIndirect:
      TFRT_DLOG_IF(FATAL, s != State::kConcrete)
          << "Cannot call get() when IndirectAsyncValue isn't ok; state: " << s;
      auto* iv_value = cast<IndirectAsyncValue>(this)->value_;
      assert(iv_value && "Indirect value not resolved");
      return iv_value->get<T>();
  }
}

template <typename T>
T& AsyncValue::get() {
  return const_cast<T&>(static_cast<const AsyncValue*>(this)->get<T>());
}

inline void AsyncValue::SetStateConcrete() {
  assert(IsConstructed() && kind() == Kind::kConcrete);
  NotifyAvailable(State::kConcrete);
}

template <typename T, typename... Args>
void AsyncValue::emplace(Args&&... args) {
  assert(GetTypeId<T>() == type_id_ && "Incorrect accessor");
  assert(IsUnconstructed() && kind() == Kind::kConcrete);

  static_cast<internal::ConcreteAsyncValue<T>*>(this)->emplace(
      std::forward<Args>(args)...);
}

// Returns the underlying error, or nullptr if there is none.
inline const DecodedDiagnostic* AsyncValue::GetErrorIfPresent() const {
  switch (kind()) {
    case Kind::kConcrete: {
      if (state() != State::kError) return nullptr;

      const char* this_ptr = reinterpret_cast<const char*>(this);
      return *reinterpret_cast<const DecodedDiagnostic* const*>(
          this_ptr + AsyncValue::kDataOrErrorOffset);
    }
    case Kind::kIndirect: {
      auto* iv_value = cast<IndirectAsyncValue>(this)->value_;
      // Unresolved IndirectAsyncValues are not errors.
      if (!iv_value) return nullptr;

      assert(iv_value->kind() != Kind::kIndirect);
      return iv_value->GetErrorIfPresent();
    }
  }
}

inline const DecodedDiagnostic& AsyncValue::GetError() const {
  auto* result = GetErrorIfPresent();
  assert(result && "Cannot call GetError() when error isn't available.");
  return *result;
}

template <typename WaiterT>
void AsyncValue::AndThen(WaiterT&& waiter) {
  // Clients generally want to use AndThen without them each having to check
  // to see if the value is present. Check for them, and immediately run the
  // lambda if it is already here.
  auto old_value = waiters_and_state_.load(std::memory_order_acquire);
  if (old_value.getInt() == State::kConcrete ||
      old_value.getInt() == State::kError) {
    assert(old_value.getPointer() == nullptr);
    waiter();
    return;
  }
  EnqueueWaiter(std::forward<WaiterT>(waiter), old_value);
}

inline raw_ostream& operator<<(raw_ostream& os,
                               const AsyncValue::State& state) {
  os << state.DebugString();
  return os;
}

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_ASYNC_VALUE_H_
