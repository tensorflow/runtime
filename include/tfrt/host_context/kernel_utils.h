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

//===- kernel_utils.h - Helpers for kernels implementations -----*- C++ -*-===//
//
// This file declares simple helper routines to make it easier to write kernels.
// Because this is part of host_context, this is intended to be small and simple
// things and is nearly header-only.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_KERNEL_UTILS_H_
#define TFRT_HOST_CONTEXT_KERNEL_UTILS_H_

#include <type_traits>

#include "llvm/Support/Error.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_frame.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/rc_array.h"
#include "tfrt/support/type_traits.h"

namespace tfrt {

template <typename T>
AsyncValueRef<T> ForwardValue(T& value, AsyncValueRef<Chain> chain,
                              HostContext* host) {
  auto result = host->MakeUnconstructedAsyncValueRef<T>();
  auto* chain_av = chain.GetAsyncValue();
  chain_av->AndThen([result = result.CopyRef(), value = std::move(value),
                     chain = std::move(chain)]() mutable {
    if (chain.IsError()) {
      result.SetError(chain.GetError());
    } else {
      result.emplace(std::move(value));
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Registration helpers used to make sync kernels easier to define.
//===----------------------------------------------------------------------===//

// TFRT_KERNEL is a macro that makes defining kernels more straightforward.
// For simple kernels with a few arguments and a single movable result,
// you can define the function using the native C++ types directly:
//
//   int32_t Add(int32_t a, int32_t b) { return a + b; }
//
// For non-strict kernels where one or more of the arguments are asynchronous,
// you can use the "Argument" wrapper type as your function arguments and return
// the result by value. The Argument<> internally wraps an AsyncValue:
//
//   int32_t Add(Argument<int32_t> a, Argument<int32_t> b) { return *a + *b; }
//
// All kernels defined with TFRT_KERNEL accept an implicit Chain as their last
// argument. For example, the Add kernels above could be called like this:
//
//   %c0 = tfrt.new.chain
//   %z = "Add"(%x, %y, %c0) : (i32, i32, !tfrt.chain) -> i32
//
// For kernels with multiple results, you can return std::pair/tuple as the
// result type:
//
//   // q = (n / d), r = n % d;
//   std::pair<int32_t, int32_t> DivRem(int32_t n, int32_t d) {
//     return {n / d, n % d};
//   }
//
// Or you can use the "Result" wrapper type. Results should appear after
// arguments:
//
//   // q = (n / d), r = n % d;
//   void DivRem(Argument<int32_t> n, Argument<int32_t> d,
//               Result<int32_t> q, Result<int32_t> r) {
//     q.Emplace(*n / *d);
//     r.Emplace(*n % *d);
//   }
//
// There is also an "Attribute" wrapper for kernels that need to consume
// attribute values. Note that attribute values are essentially literals, so
// these should generally only be used for things that never change for a given
// executable, as opposed to values that will be computed at runtime. Attributes
// should appear after arguments and results:
//
//   int32_t MakeInt(Attribute<int32_t> value) {
//     return *value;
//   }
//
// Similarly StringAttribute, ArrayAttribute<T> and AggregateAttr are also
// provided. They work the same way as attribute but for arrays of T or
// characters or nested arrays of heterogeneous types.
//
// For kernels that may fail at runtime, for sync kernels the preferred way is
// to return Expected<T> to report the error:
//
//   Expected<std::string> ReadFile(std::string path) {
//     auto* f = OpenFile(*path);
//     if (!f) {
//       return Error("Could not open file");
//     }
//     <Read file here>
//     return bytes;
//   }
//
// For sync kernels, KernelErrorHandler may be used:
//
//   void ReadFile(Argument<std::string> path, Result<std::string> bytes,
//                 KernelErrorHandler handler) {
//     auto* f = OpenFile(*path);
//     if (!f) {
//       handler.ReportError("Could not open file");
//       return;
//     }
//     <Read file here>
//   }
//
// WARNING: KernelErrorHandler can't be used asynchronously because it holds a
// pointer to the AsyncKernelFrame, which is destroyed when the kernel returns.
//
// Kernels can also take the AsyncKernelFrame if they need access to the
// HostContext or anything else the above wrapper types don't provide.
//
// See the definitions of the wrapper types below for more details.
//
// TODO(b/141203112): Switch to template when we can use C++17.
#define TFRT_KERNEL(...) \
  ::tfrt::TfrtKernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Invoke

// Kernels should use this so we know the kernel has an argument.
template <typename T>
class Argument {
 public:
  explicit Argument(AsyncValue* value) : value_(value) {}

  AsyncValue* value() const { return value_; }
  AsyncValueRef<T> ValueRef() const {
    return AsyncValueRef<T>(FormRef(value_));
  }
  T& get() const { return value_->template get<T>(); }
  T* operator->() const { return &get(); }
  T& operator*() const { return get(); }

 private:
  // Does not own the async value.
  AsyncValue* value_;
};

// ArgumentView is used to project the payload of an AsyncValue via a view
// class.
//
// ViewT needs to satisfy certain contracts as shown in the following
// SampleViewT class in order for ArgumentView<> to treat it as a view class.
//
// class SampleViewT {
//  public:
//    // Required type alias for ArgumentView<> to recognize the class as a view
//    // class.
//    using UnderlyingT = SampleUnderlyingType;
//
//    // Required constructor for ArgumentView<> to instantiate the view class.
//    SampleViewT(UnderlyingT* underlying);
//
//    // View functions
//
//    UnderlyingT& underlying_;
// };
template <typename ViewT>
class ArgumentView {
  using UnderlyingT = typename ViewT::UnderlyingT;

 public:
  explicit ArgumentView(AsyncValue* value)
      : value_(value), arg_(&value->template get<UnderlyingT>()) {}

  AsyncValue* value() const { return value_; }
  ViewT& get() const { return arg_; }
  ViewT* operator->() const { return &get(); }
  ViewT& operator*() const { return get(); }

 private:
  // Does not own the async value.
  AsyncValue* value_;
  mutable ViewT arg_;
};

// RemainingArguments collects all remaining arguments in an ArrayRef. There can
// be at most one RemainingArguments/RepeatedArguments, and it must appear after
// all other Arguments.
//
// This should only be used when argument types are unknown, for example
// TFRTCall.
class RemainingArguments {
 public:
  explicit RemainingArguments(ArrayRef<AsyncValue*> remaining_arguments)
      : remaining_arguments_(remaining_arguments) {}

  ArrayRef<AsyncValue*> values() const { return remaining_arguments_; }
  size_t size() const { return remaining_arguments_.size(); }
  AsyncValue* operator[](size_t i) const { return remaining_arguments_[i]; }

 private:
  // Does not own the async values.
  ArrayRef<AsyncValue*> remaining_arguments_;
};

// RepeatedArguments collects all remaining arguments of the same type in an
// ArrayRef. There can be at most one RemainingArguments/RepeatedArguments, and
// it must appear after all other Arguments.
template <typename T>
class RepeatedArguments {
 public:
  class Iterator {
    using AsyncValueIterator = ArrayRef<AsyncValue*>::iterator;

   public:
    explicit Iterator(AsyncValueIterator it) : it_{it} {}

    T& operator*() const { return (*it_)->get<T>(); }
    T* operator->() const { return &(*it_)->get<T>(); }

    bool operator==(Iterator other) const { return it_ == other.it_; }
    bool operator!=(Iterator other) const { return it_ != other.it_; }

    Iterator& operator++() {
      ++it_;
      return *this;
    }

   private:
    AsyncValueIterator it_;
  };

  explicit RepeatedArguments(ArrayRef<AsyncValue*> repeated_arguments)
      : repeated_arguments_(repeated_arguments) {}

  ArrayRef<AsyncValue*> values() const { return repeated_arguments_; }

  size_t size() const { return repeated_arguments_.size(); }
  T& operator[](size_t index) const {
    return repeated_arguments_[index]->template get<T>();
  }

  // Enable the ranged-for usage, e.g. for (auto& v : repeated_arguments) {...}
  Iterator begin() const { return Iterator{repeated_arguments_.begin()}; }
  Iterator end() const { return Iterator{repeated_arguments_.end()}; }

 private:
  // Does not own the async values.
  ArrayRef<AsyncValue*> repeated_arguments_;
};

// Kernels should use this so we know the kernel has a result.
template <typename T>
class Result {
 public:
  explicit Result(HostContext* host, AsyncValue** result)
      : host_(host), result_(result) {
    assert(*result == nullptr);
  }

  // Construct the result in place.
  template <typename... Args>
  void Emplace(Args&&... args) {
    Set(host_->MakeAvailableAsyncValueRef<T>(std::forward<Args>(args)...));
  }

  // Use this argument as a result without a deep copy.
  void Set(Argument<T> argument) {
    assert(*result_ == nullptr);
    argument.value()->AddRef();
    *result_ = argument.value();
  }

  void Set(RCReference<AsyncValue> value) {
    assert(*result_ == nullptr);
    *result_ = value.release();
  }

  void Set(AsyncValueRef<T> value) {
    assert(*result_ == nullptr);
    *result_ = value.release();
  }

  AsyncValueRef<T> Allocate() {
    assert(*result_ == nullptr);
    auto result = host_->MakeUnconstructedAsyncValueRef<T>();
    // result_ is stored in AsyncKernelFrame and needs a +1 ref count.
    *result_ = result.CopyRef().release();
    return result;
  }

  RCReference<IndirectAsyncValue> AllocateIndirect() {
    assert(*result_ == nullptr);
    auto result = host_->MakeIndirectAsyncValue();
    // result_ is stored in AsyncKernelFrame and needs a +1 ref count.
    *result_ = result.CopyRef().release();
    return result;
  }

 private:
  HostContext* host_;
  AsyncValue** result_;
};

// RemainingResults collects all remaining results in a MutableArrayRef. There
// can be at most one RemainingResults, and it must appear after all other
// Results.
//
// This should only be used when result types are unknown, for example TFRTCall.
class RemainingResults {
 public:
  RemainingResults(HostContext* host,
                   MutableArrayRef<RCReference<AsyncValue>> remaining_results)
      : host_(host), remaining_results_(remaining_results) {}

  MutableArrayRef<RCReference<AsyncValue>> values() const {
    return remaining_results_;
  }
  size_t size() const { return remaining_results_.size(); }
  RCReference<AsyncValue>& operator[](size_t i) const {
    return remaining_results_[i];
  }

  template <typename T>
  const RCReference<AsyncValue>& AllocateAt(int index) {
    assert(!remaining_results_[index]);
    auto result = host_->MakeUnconstructedAsyncValueRef<T>().ReleaseRCRef();
    remaining_results_[index] = std::move(result);
    return remaining_results_[index];
  }

  // This sets the specified result to a newly created IndirectAsyncResult and
  // returns an unowned pointer to it.
  RCReference<IndirectAsyncValue> AllocateIndirectResultAt(int index) {
    auto indirect = host_->MakeIndirectAsyncValue();
    remaining_results_[index] = indirect.CopyRef();
    return indirect;
  }

 private:
  HostContext* host_;
  MutableArrayRef<RCReference<AsyncValue>> remaining_results_;
};

// RemainingAttributes collects all remaining attributes. There can be at most
// one RemainingAttributes, and it must appear after all other Attribute.
class RemainingAttributes {
 public:
  explicit RemainingAttributes(ArrayRef<uint8_t> attribute_section,
                               ArrayRef<const void*> remaining_attributes)
      : attribute_section_(attribute_section),
        remaining_attributes_(remaining_attributes) {}

  size_t size() const { return remaining_attributes_.size(); }

  template <typename T>
  Attribute<T> Get(size_t i) const {
    return Attribute<T>(remaining_attributes_[i]);
  }

  StringAttribute GetStringAttribute(size_t i) const {
    return StringAttribute(remaining_attributes_[i]);
  }

  template <typename T>
  ArrayAttribute<T> GetArrayAttribute(size_t i) const {
    return ArrayAttribute<T>(remaining_attributes_[i]);
  }

  AggregateAttr GetAggregateAttr(size_t i) const {
    return AggregateAttr(remaining_attributes_[i]);
  }

 private:
  ArrayRef<uint8_t> attribute_section_;
  ArrayRef<const void*> remaining_attributes_;
};

// Kernels can take one of these if they need to report runtime errors.
class KernelErrorHandler {
 public:
  KernelErrorHandler(AsyncKernelFrame* frame) : frame_(frame) {}  // NOLINT

  // Example usage:
  //
  // kernel_handler.ReportError("This is an error message");
  // int i = 2;
  // TensorShape shape = ...
  // kernel_handler.ReportError("Error: i is ", i, ", shape is ", shape);
  template <typename... Args>
  void ReportError(Args&&... args) {
    frame_->ReportError(std::forward<Args>(args)...);
  }

  // Example usage is similar to ReportError().
  template <typename... Args>
  RCReference<AsyncValue> EmitError(Args&&... args) {
    return frame_->EmitError(std::forward<Args>(args)...);
  }

  Location GetLocation() const { return frame_->GetLocation(); }

 private:
  AsyncKernelFrame* frame_;
};

// This class is an implementation detail of TFRT_KERNEL.
template <typename F, F f>
struct TfrtKernelImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct TfrtKernelImpl<Return (*)(Args...), impl_fn> {
  // This is the main entry point that gets registered as a kernel.
  static void Invoke(AsyncKernelFrame* frame) {
    SyncKernelCallHelper<Args..., TypeTag<int>>::template Invoke<0, 0, 0, false,
                                                                 false>(frame);
  }

 private:
  // Check whether a type T has an internal UnderlyingT type.
  template <typename T>
  using UnderlyingT = typename T::UnderlyingT;

  template <typename T>
  using IsViewT = is_detected<UnderlyingT, T>;

  // Helper that introspects the kernel arguments to derive the signature and
  // cast parts of the AsyncKernelFrame to their appropriate type before passing
  // them to impl_fn. Works by recursively unpacking the arguments.
  template <typename... RemainingArgs>
  struct SyncKernelCallHelper;

  // Casts the return value of the kernel, if non-void. Otherwise ignores the
  // return value.
  template <bool has_kernel_error, typename T>
  struct SyncKernelReturnHelper {
    static void Invoke(AsyncKernelFrame* frame, const Args&... args) {
      static_assert(
          !has_kernel_error,
          "Do not return by value when using KernelErrorHandler. Set the "
          "kernel's return value with Result<> instead. "
          "Bad: Chain my_kernel(KernelErrorHandler handler). "
          "Good: void my_kernel(Result<Chain> out, KernelErrorHandler "
          "handler).");
      HandleReturn(frame, impl_fn(args...));
    }
  };

  template <bool has_kernel_error>
  struct SyncKernelReturnHelper<has_kernel_error, void> {
    static void Invoke(AsyncKernelFrame* frame, const Args&... args) {
      impl_fn(args...);
    }
  };

  // Store result as an AsyncValue output in AsyncKernelFrame by creating a
  // ConcreteAsyncValue.
  template <typename T>
  static void StoreResultAt(AsyncKernelFrame* frame, int index, T&& t) {
    frame->EmplaceResultAt<std::decay_t<T>>(index, std::forward<T>(t));
  }

  // Store the output Chain as an AsyncValue output in AsyncKernelFrame by
  // re-using the ready chain cached in HostContext.
  static void StoreResultAt(AsyncKernelFrame* frame, int index, Chain t) {
    frame->SetResultAt(index, frame->GetHostContext()->GetReadyChain());
  }

  // Store an already created AsyncValue as a result in the AsyncKernelFrame.
  template <typename T>
  static void StoreResultAt(AsyncKernelFrame* frame, int index,
                            AsyncValueRef<T> t) {
    frame->SetResultAt(index, std::move(t));
  }

  // Store the function result back to the output AsyncValue in the
  // AsyncKernelFrame.
  template <typename T>
  static void HandleReturn(AsyncKernelFrame* frame, T&& t) {
    assert(frame->GetNumResults() == 1 && "Extra results passed to kernel.");
    StoreResultAt(frame, 0, std::forward<T>(t));
  }

  // For kernel functions that return std::pair<>, store the result as the first
  // and second output AsyncValue in the AsyncKernelFrame.
  template <typename T1, typename T2>
  static void HandleReturn(AsyncKernelFrame* frame, std::pair<T1, T2>&& t) {
    assert(frame->GetNumResults() == 2 &&
           "Incorrect number of results passed to kernel.");
    StoreResultAt(frame, 0, std::move(t.first));
    StoreResultAt(frame, 1, std::move(t.second));
  }

  // For kernel functions that return std::tuple<>, store the results in order
  // as the output AsyncValues in the AsyncKernelFrame.
  template <typename... T>
  static void HandleReturn(AsyncKernelFrame* frame, std::tuple<T...>&& t) {
    assert(frame->GetNumResults() == sizeof...(T) &&
           "Incorrect number of results passed to kernel.");
    EmplaceTupleResult(frame, std::move(t),
                       std::make_index_sequence<sizeof...(T)>{});
  }

  // For kernel functions that return Expected<T>, if the returned Expected<T>
  // contains an error, call frame->ReportError() to report the error message
  // and set an error in the output AsyncValue. Otherwise, store the return
  // value as output AsyncValue.
  template <typename T>
  static void HandleReturn(AsyncKernelFrame* frame, llvm::Expected<T>&& t) {
    if (t) {
      HandleReturn(frame, std::move(*t));
    } else {
      frame->ReportError(StrCat(t.takeError()));
    }
  }

  // For kernel functions that return AsyncValueRef<std::tuple<>>, store the
  // results in order as the output AsyncValues in the AsyncKernelFrame.
  template <typename... T>
  static void HandleReturn(AsyncKernelFrame* frame,
                           AsyncValueRef<std::tuple<T...>> t) {
    assert(frame->GetNumResults() == sizeof...(T) &&
           "Incorrect number of results passed to kernel.");
    AllocateTupleResult(frame, t.CopyRef(),
                        std::make_index_sequence<sizeof...(T)>{});

    t.AndThen([frame, t = t.CopyRef(),
               results = RCArray<AsyncValue>(frame->GetResults())] {
      if (t.IsError()) {
        for (int i = 0; i < sizeof...(T); i++) {
          results[i]->SetError(t.GetError());
        }
        return;
      }
      EmplaceTupleResult(results.values(), t.get(),
                         std::make_index_sequence<sizeof...(T)>{});
    });
  }

  // Helper function for allocating multiple AsyncValue in AsyncKernelFrame for
  // each value type in std::tuple<>.
  template <typename... T, size_t... I>
  static void AllocateTupleResult(AsyncKernelFrame* frame,
                                  AsyncValueRef<std::tuple<T...>> t,
                                  std::index_sequence<I...>) {
    std::ignore =
        std::initializer_list<int>{(frame->AllocateResultAt<T>(I), 0)...};
  }

  // Helper function for storing multiple return values in std::tuple<> as
  // AsyncValues in the 'output'.
  template <typename... T, size_t... I>
  static void EmplaceTupleResult(ArrayRef<AsyncValue*> output,
                                 std::tuple<T...>& input,
                                 std::index_sequence<I...>) {
    // Use braced-init-list to retrieve the results in the tuple in sequence.
    std::ignore = std::initializer_list<int>{
        (output[I]->emplace<std::decay_t<T>>(std::move(std::get<I>(input))),
         0)...};
  }

  // Helper function for storing multiple return values in std::tuple<> as
  // output AsyncValue in AsyncKernelFrame.
  template <typename TupleT, size_t... I>
  static void EmplaceTupleResult(AsyncKernelFrame* frame, TupleT&& result,
                                 std::index_sequence<I...>) {
    // Use braced-init-list to retrieve the results in the tuple in sequence.
    std::ignore = std::initializer_list<int>{
        (StoreResultAt(frame, I, std::get<I>(std::forward<TupleT>(result))),
         0)...};
  }

  // Specialization to cast a single input argument (Head).
  template <typename Head, typename... Tail>
  struct SyncKernelCallHelper<Argument<Head>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1,
                    "Do not place Arguments after RemainingArguments");
      static_assert(out_idx == 0, "Arguments should appear before results.");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");
      Argument<Head> arg(frame->GetArgAt(in_idx));
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx + 1, out_idx, const_idx, has_kernel_error,
          (has_in_chain || std::is_same<Head, Chain>())>(frame, pargs..., arg);
    }
  };

  template <typename Head, typename... Tail>
  struct SyncKernelCallHelper<ArgumentView<Head>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1,
                    "Do not place ArgumentViews after RemainingArguments");
      static_assert(out_idx == 0,
                    "ArgumentViews should appear before results.");
      static_assert(const_idx == 0,
                    "ArgumentViews should appear before attributes.");
      ArgumentView<Head> arg(frame->GetArgAt(in_idx));
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx + 1, out_idx, const_idx, has_kernel_error, has_in_chain>(
          frame, pargs..., arg);
    }
  };

  // RemainingArguments provides an ArrayRef<AsyncValue*> containing all
  // remaining arguments. Useful for variadic kernels.
  template <typename... Tail>
  struct SyncKernelCallHelper<RemainingArguments, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(
          in_idx != -1,
          "Do not use more than one RemainingArguments/RepeatedArguments");
      static_assert(out_idx == 0, "Arguments should appear before results.");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");
      RemainingArguments remaining_arguments(
          frame->GetArguments().drop_front(in_idx));

      SyncKernelCallHelper<Tail...>::template Invoke<
          -1, out_idx, const_idx, has_kernel_error, has_in_chain>(
          frame, pargs..., remaining_arguments);
    }
  };

  // RepeatedArguments provides an ArrayRef<AsyncValue*> containing all
  // remaining arguments. Useful for variadic kernels.
  template <typename T, typename... Tail>
  struct SyncKernelCallHelper<RepeatedArguments<T>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(
          in_idx != -1,
          "Do not use more than one RemainingArguments/RepeatedArguments");
      static_assert(out_idx == 0, "Arguments should appear before results.");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");
      RepeatedArguments<T> repeated_arguments(
          frame->GetArguments().drop_front(in_idx));
      SyncKernelCallHelper<Tail...>::template Invoke<
          -1, out_idx, const_idx, has_kernel_error, has_in_chain>(
          frame, pargs..., repeated_arguments);
    }
  };

  // Specialization to cast a single result argument (Head).
  template <typename Head, typename... Tail>
  struct SyncKernelCallHelper<Result<Head>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(out_idx != -1,
                    "Do not place Results after RemainingResults");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");
      Result<Head> arg(frame->GetHostContext(), &frame->GetResults()[out_idx]);
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx, out_idx + 1, const_idx, has_kernel_error, has_in_chain>(
          frame, pargs..., arg);
    }
  };

  // RemainingResults provides an MutableArrayRef<AsyncValue*> containing all
  // remaining results. Useful for variadic kernels.
  template <typename... Tail>
  struct SyncKernelCallHelper<RemainingResults, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(out_idx != -1, "Do not use more than one RemainingResults");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");

      MutableArrayRef<AsyncValue*> returned_results =
          frame->GetResults().drop_front(out_idx);

      SmallVector<RCReference<AsyncValue>, 4> result_values;
      result_values.resize(returned_results.size());

      RemainingResults remaining_results(frame->GetHostContext(),
                                         result_values);
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx, -1, const_idx, has_kernel_error, has_in_chain>(
          frame, pargs..., remaining_results);

      bool overwrote_result = false;
      for (size_t i = 0, e = result_values.size(); i != e; ++i) {
        // In error cases, ReportError() may have filled in the kernel frame's
        // results already with errors.  We only want to overwrite these if
        // returned_results has valid results.
        if (auto* result = result_values[i].release()) {
          if (returned_results[i]) {
            assert(returned_results[i]->IsError() &&
                   "shouldn't overwrite a normal value result");
            returned_results[i]->DropRef();
            overwrote_result = true;
          }
          returned_results[i] = result;
        }
      }

      // If we overwrote a result an error result, make sure there is at least
      // one error result left - otherwise this will have masked the error
      // entirely.
#ifndef NDEBUG
      if (overwrote_result) {
        bool had_error = false;
        for (auto& result : frame->GetResults())
          if (result->IsError()) {
            had_error = true;
            break;
          }
        assert(had_error &&
               "RemainingResults kernel overwrote all the errors with "
               "normal results");
      }
#endif
    }
  };

  // Specialization to cast a single attribute (Head).
  template <typename Head, typename... Tail>
  struct SyncKernelCallHelper<Attribute<Head>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(const_idx != -1,
                    "Do not place Attributes after RemainingAttributes");
      Attribute<Head> arg = frame->GetAttributeAt<Head>(const_idx);
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx, out_idx, const_idx + 1, has_kernel_error, has_in_chain>(
          frame, pargs..., arg);
    }
  };

  // Like the above, but for arrays.
  template <typename Head, typename... Tail>
  struct SyncKernelCallHelper<ArrayAttribute<Head>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(const_idx != -1,
                    "Do not place ArrayAttribute after RemainingAttributes");
      ArrayAttribute<Head> arg = frame->GetArrayAttributeAt<Head>(const_idx);
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx, out_idx, const_idx + 1, has_kernel_error, has_in_chain>(
          frame, pargs..., arg);
    }
  };

  // Like the above, but for strings.
  template <typename... Tail>
  struct SyncKernelCallHelper<StringAttribute, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(const_idx != -1,
                    "Do not place StringAttribute after RemainingAttributes");
      StringAttribute arg = frame->GetStringAttribute(const_idx);
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx, out_idx, const_idx + 1, has_kernel_error, has_in_chain>(
          frame, pargs..., arg);
    }
  };

  // Like the above, but for typed attributes.
  template <typename TypedAttrT, typename... Tail>
  struct SyncKernelCallTypedAttrHelper {
    static_assert(std::is_base_of<TypedAttrBase, TypedAttrT>::value,
                  "TypedAttrT must be derived from class TypedAttrBase");
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(const_idx != -1,
                    "Do not place typed attributes after RemainingAttributes");
      TypedAttrT arg(frame->GetAttributes()[const_idx]);
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx, out_idx, const_idx + 1, has_kernel_error, has_in_chain>(
          frame, pargs..., arg);
    }
  };

  // Like the above, but for StringAttr.
  template <typename... Tail>
  struct SyncKernelCallHelper<StringAttr, Tail...>
      : SyncKernelCallTypedAttrHelper<StringAttr, Tail...> {};

  // Like the above, but for DenseAttr.
  template <typename... Tail>
  struct SyncKernelCallHelper<DenseAttr, Tail...>
      : SyncKernelCallTypedAttrHelper<DenseAttr, Tail...> {};

  // Like the above, but for ShapeAttr.
  template <typename... Tail>
  struct SyncKernelCallHelper<ShapeAttr, Tail...>
      : SyncKernelCallTypedAttrHelper<ShapeAttr, Tail...> {};

  // Like the above, but for ArrayAttr.
  template <typename... Tail>
  struct SyncKernelCallHelper<ArrayAttr, Tail...>
      : SyncKernelCallTypedAttrHelper<ArrayAttr, Tail...> {};

  // Like the above, but for AggregateAttr.
  template <typename... Tail>
  struct SyncKernelCallHelper<AggregateAttr, Tail...>
      : SyncKernelCallTypedAttrHelper<AggregateAttr, Tail...> {};

  template <typename... Tail>
  struct SyncKernelCallHelper<RemainingAttributes, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(const_idx != -1,
                    "Do not use more than one RemainingAttributes");
      RemainingAttributes remaining_attributes(
          frame->GetAttributeSection(),
          frame->GetAttributes().drop_front(const_idx));
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx, out_idx, -1, has_kernel_error, has_in_chain>(
          frame, pargs..., remaining_attributes);
    }
  };

  // If this kernel can fail, pass it an error argument.
  template <typename... Tail>
  struct SyncKernelCallHelper<KernelErrorHandler, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      SyncKernelCallHelper<Tail...>::template Invoke<in_idx, out_idx, const_idx,
                                                     true, has_in_chain>(
          frame, pargs..., KernelErrorHandler(frame));
    }
  };

  // If this kernel requires ExecutionContext, pass it as an argument.
  template <typename... Tail>
  struct SyncKernelCallHelper<const ExecutionContext&, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx, out_idx, const_idx, has_kernel_error, has_in_chain>(
          frame, pargs..., frame->GetExecutionContext());
    }
  };

  // If this kernel requires the frame for some reason, pass it as an argument.
  template <typename... Tail>
  struct SyncKernelCallHelper<AsyncKernelFrame*, Tail...> {
    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx, out_idx, const_idx, has_kernel_error, has_in_chain>(
          frame, pargs..., frame);
    }
  };

  // Treat other pointer as an Argument.
  template <typename Head, typename... Tail>
  struct SyncKernelCallHelper<Head*, Tail...> {
    static_assert(!std::is_same<Head, HostContext>::value,
                  "HostContext* is not allowed as a kernel argument. Use const "
                  "ExecutionContext& instead.");

    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1,
                    "Do not place Arguments after RemainingArguments");
      static_assert(out_idx == 0, "Arguments should appear before results.");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");
      static_assert(!std::is_same<Head, Chain>(),
                    "Do not pass Chain as pointer.");
      auto* arg = &frame->GetArgAt<Head>(in_idx);
      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx + 1, out_idx, const_idx, has_kernel_error, has_in_chain>(
          frame, pargs..., arg);
    }
  };

  // Treat any other type as an Argument.
  template <typename Head, typename... Tail>
  struct SyncKernelCallHelper<Head, Tail...> {
    using ArgT = std::decay_t<Head>;

    template <typename T>
    static T GetArg(AsyncValue* value, std::true_type) {
      return T(&value->template get<typename ArgT::UnderlyingT>());
    }

    template <typename T>
    static T& GetArg(AsyncValue* value, std::false_type) {
      return value->get<ArgT>();
    }

    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1,
                    "Do not place Arguments after RemainingArguments");
      static_assert(out_idx == 0, "Arguments should appear before results.");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");
      auto* async_value = frame->GetArgAt(in_idx);
      auto&& arg = GetArg<ArgT>(async_value, IsViewT<ArgT>());

      SyncKernelCallHelper<Tail...>::template Invoke<
          in_idx + 1, out_idx, const_idx, has_kernel_error,
          (has_in_chain || std::is_same<ArgT, Chain>())>(frame, pargs..., arg);
    }
  };

  // Base case: No arguments left.
  // TypeTag<T> is a dummy template parameter to work around the restriction
  // of GCC that fully specialized template is not allowed in a template class.
  template <typename T>
  struct SyncKernelCallHelper<TypeTag<T>> {
    // Verify the result index for non-void Return type
    template <typename ReturnT, int out_idx>
    struct AssertIndex {
      static void Verify(AsyncKernelFrame* frame) {
        static_assert(out_idx == 0,
                      "Don't mix return by value and Result<> syntax.");
      }
    };

    // Verify the result index for void Return type.
    template <int out_idx>
    struct AssertIndex<void, out_idx> {
      static void Verify(AsyncKernelFrame* frame) {
        assert((out_idx == frame->GetNumResults() || out_idx == -1) &&
               "Extra results passed to kernel.");
      }
    };

    template <int in_idx, int out_idx, int const_idx, bool has_kernel_error,
              bool has_in_chain, typename... PreviousArgs>
    static void Invoke(AsyncKernelFrame* frame, const PreviousArgs&... pargs) {
      AssertIndex<Return, out_idx>::Verify(frame);

      assert((in_idx == frame->GetNumArgs() || in_idx == -1 ||
              (!has_in_chain && in_idx == frame->GetNumArgs() - 1 &&
               frame->GetArgAt(in_idx)->IsType<Chain>())) &&
             "Extra arguments passed to kernel.");
      assert(const_idx == frame->GetNumAttributes() ||
             const_idx == -1 && "Extra attributes passed to kernel.");
      SyncKernelReturnHelper<has_kernel_error, Return>::Invoke(frame, pargs...);
    }
  };
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_KERNEL_UTILS_H_
