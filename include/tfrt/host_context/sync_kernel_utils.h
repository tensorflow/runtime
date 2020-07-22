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

//===- sync_kernel_utils.h - Helpers for defining sync kernels --*- C++ -*-===//
//
// This file declares simple helper routines to make it easier to write
// synchronous kernels.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_SYNC_KERNEL_UTILS_H_
#define TFRT_HOST_CONTEXT_SYNC_KERNEL_UTILS_H_

#include <type_traits>

#include "llvm/Support/Error.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/sync_kernel_frame.h"
#include "tfrt/host_context/value.h"
#include "tfrt/support/type_traits.h"

namespace tfrt {

//===----------------------------------------------------------------------===//
// Registration helpers used to make sync kernels easier to define.
//===----------------------------------------------------------------------===//

// TFRT_SYNC_KERNEL is a macro that makes defining kernels more straightforward.
// For simple kernels with a few arguments and a single movable result,
// you can define the function using the native C++ types directly:
//
//   int32_t Add(int32_t a, int32_t b) { return a + b; }
//
// And then wrap this function with TFRT_SYNC_KERNEL for registritaion with
// KernelRegistry:
//
//  registry->AddSyncKernel("tfrt.sync.add.i32", TFRT_SYNC_KERNEL(Add));
//
// For kernels with multiple results, you can return std::pair/tuple as the
// result type:
//
//   // q = (n / d), r = n % d;
//   std::pair<int32_t, int32_t> DivRem(int32_t n, int32_t d) {
//     return {n / d, n % d};
//   }
//
// There is an "Attribute" wrapper for kernels that need to consume
// attribute values. Note that attribute values are essentially literals, so
// these should generally only be used for things that never change for a given
// executable, as opposed to values that will be computed at runtime. Attributes
// should appear after arguments:
//
//   int32_t MakeInt(Attribute<int32_t> value) {
//     return *value;
//   }
//
// Similarly StringAttribute, ArrayAttribute<T> and AggregateAttr are also
// provided. They work the same way as attribute but for arrays of T or
// characters or nested arrays of heterogeneous types.
//
// For kernels that may fail at runtime, return Expected<T> to report the error:
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
//
// Kernels can also take the SyncKernelFrame if they need access to the anything
// else the above wrapper types don't provide.
//
// See the definitions of the wrapper types below for more details.
//
// TODO(b/141203112): Switch to template when we can use C++17.
#define TFRT_SYNC_KERNEL(...) \
  ::tfrt::TfrtSyncKernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Invoke

// RemainingSyncArguments collects all remaining arguments in an ArrayRef. There
// can be at most one RemainingSyncArguments, and it must appear after all other
// Arguments.
class RemainingSyncArguments {
 public:
  explicit RemainingSyncArguments(ArrayRef<uint32_t> reg_indices,
                                  ArrayRef<Value*> registers)
      : reg_indices_{reg_indices}, registers_{registers} {}

  size_t size() const { return reg_indices_.size(); }
  Value* operator[](size_t i) const { return registers_[reg_indices_[i]]; }

 private:
  ArrayRef<uint32_t> reg_indices_;
  ArrayRef<Value*> registers_;
};

// This class is an implementation detail of TFRT_SYNC_KERNEL.
template <typename F, F f>
struct TfrtSyncKernelImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct TfrtSyncKernelImpl<Return (*)(Args...), impl_fn> {
  // This is the main entry point that gets registered as a kernel.
  static void Invoke(SyncKernelFrame* frame) {
    SyncKernelCallHelper<Args..., TypeTag<int>>::template Invoke<0, 0>(frame);
  }

 private:
  // Check whether a type T has an internal UnderlyingT type.
  template <typename T>
  using UnderlyingT = typename T::UnderlyingT;

  template <typename T>
  using IsViewT = is_detected<UnderlyingT, T>;

  // Casts the return value of the kernel, if non-void. Otherwise ignores the
  // return value.
  template <typename T, typename Enable = void>
  struct SyncKernelReturnHelper {
    static void Invoke(SyncKernelFrame* frame, const Args&... args) {
      HandleReturn(frame, impl_fn(args...));
    }
  };

  // Specialize for the case when T is void.
  template <typename T>
  struct SyncKernelReturnHelper<
      T, std::enable_if_t<std::is_same<T, void>::value>> {
    static void Invoke(SyncKernelFrame* frame, const Args&... args) {
      impl_fn(args...);
    }
  };

  // Store result as a Value output in SyncKernelFrame.
  template <typename T>
  static void StoreResultAt(SyncKernelFrame* frame, int index, T&& t) {
    frame->EmplaceResultAt<std::decay_t<T>>(index, std::forward<T>(t));
  }

  // Store the function result back to the output Value in the
  // SyncKernelFrame.
  template <typename T>
  static void HandleReturn(SyncKernelFrame* frame, T&& t) {
    assert(frame->GetNumResults() == 1 && "Extra results passed to kernel.");
    StoreResultAt(frame, 0, std::forward<T>(t));
  }

  // For kernel functions that return std::pair<>, store the result as the first
  // and second output Value in the SyncKernelFrame.
  template <typename T1, typename T2>
  static void HandleReturn(SyncKernelFrame* frame, std::pair<T1, T2>&& t) {
    assert(frame->GetNumResults() == 2 &&
           "Incorrect number of results passed to kernel.");
    StoreResultAt(frame, 0, std::move(t.first));
    StoreResultAt(frame, 1, std::move(t.second));
  }

  // For kernel functions that return std::tuple<>, store the results in order
  // as the output Values in the SyncKernelFrame.
  template <typename... T>
  static void HandleReturn(SyncKernelFrame* frame, std::tuple<T...>&& t) {
    assert(frame->GetNumResults() == sizeof...(T) &&
           "Incorrect number of results passed to kernel.");
    EmplaceTupleResult(frame, std::move(t),
                       std::make_index_sequence<sizeof...(T)>{});
  }

  // Helper function for storing multiple return values in std::tuple<> as
  // output Value in SyncKernelFrame.
  template <typename TupleT, size_t... I>
  static void EmplaceTupleResult(SyncKernelFrame* frame, TupleT&& result,
                                 std::index_sequence<I...>) {
    // Use braced-init-list to retrieve the results in the tuple in sequence.
    std::ignore = std::initializer_list<int>{
        (StoreResultAt(frame, I, std::get<I>(std::forward<TupleT>(result))),
         0)...};
  }

  // For kernel functions that return Expected<T>, if the returned Expected<T>
  // contains an error, call frame->SetError() to report the error message.
  // Otherwise, store the return value as output Value.
  template <typename T>
  static void HandleReturn(SyncKernelFrame* frame, llvm::Expected<T>&& t) {
    if (t) {
      HandleReturn(frame, std::move(*t));
    } else {
      frame->SetError(t.takeError());
    }
  }

  // For kernel functions that return Error, call frame->SetError() to report
  // the error message.
  static void HandleReturn(SyncKernelFrame* frame, Error&& t) {
    if (t) {
      frame->SetError(std::move(t));
    }
  }

  // Helper that introspects the kernel arguments to derive the signature and
  // cast parts of the SyncKernelFrame to their appropriate type before passing
  // them to impl_fn. Works by recursively unpacking the arguments.
  template <typename... RemainingArgs>
  struct SyncKernelCallHelper;

  // Specialization to cast a single attribute (Head).
  template <typename Head, typename... Tail>
  struct SyncKernelCallHelper<Attribute<Head>, Tail...> {
    template <int in_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(SyncKernelFrame* frame, const PreviousArgs&... pargs) {
      Attribute<Head> arg = frame->GetAttributeAt<Head>(const_idx);
      SyncKernelCallHelper<Tail...>::template Invoke<in_idx, const_idx + 1>(
          frame, pargs..., arg);
    }
  };

  // Like the above, but for arrays.
  template <typename Head, typename... Tail>
  struct SyncKernelCallHelper<ArrayAttribute<Head>, Tail...> {
    template <int in_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(SyncKernelFrame* frame, const PreviousArgs&... pargs) {
      ArrayAttribute<Head> arg = frame->GetArrayAttributeAt<Head>(const_idx);
      SyncKernelCallHelper<Tail...>::template Invoke<in_idx, const_idx + 1>(
          frame, pargs..., arg);
    }
  };

  // Like the above, but for strings.
  template <typename... Tail>
  struct SyncKernelCallHelper<StringAttribute, Tail...> {
    template <int in_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(SyncKernelFrame* frame, const PreviousArgs&... pargs) {
      StringAttribute arg = frame->GetStringAttribute(const_idx);
      SyncKernelCallHelper<Tail...>::template Invoke<in_idx, const_idx + 1>(
          frame, pargs..., arg);
    }
  };

  // Like the above, but for typed attributes.
  template <typename TypedAttrT, typename... Tail>
  struct SyncKernelCallTypedAttrHelper {
    static_assert(std::is_base_of<TypedAttrBase, TypedAttrT>::value,
                  "TypedAttrT must be derived from class TypedAttrBase");
    template <int in_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(SyncKernelFrame* frame, const PreviousArgs&... pargs) {
      TypedAttrT arg(frame->GetAttributeAt(const_idx));
      SyncKernelCallHelper<Tail...>::template Invoke<in_idx, const_idx + 1>(
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

  // If this kernel requires ExecutionContext, pass it as an argument.
  template <typename... Tail>
  struct SyncKernelCallHelper<const ExecutionContext&, Tail...> {
    template <int in_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(SyncKernelFrame* frame, const PreviousArgs&... pargs) {
      SyncKernelCallHelper<Tail...>::template Invoke<in_idx, const_idx>(
          frame, pargs..., frame->GetExecutionContext());
    }
  };

  // Treat other pointer as an Argument.
  template <typename Head, typename... Tail>
  struct SyncKernelCallHelper<Head*, Tail...> {
    static_assert(!std::is_same<Head, HostContext>::value,
                  "HostContext* is not allowed as a kernel argument. Use const "
                  "ExecutionContext& instead.");

    template <int in_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(SyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");
      auto* arg = &frame->GetArgAt<Head>(in_idx);
      SyncKernelCallHelper<Tail...>::template Invoke<in_idx + 1, const_idx>(
          frame, pargs..., arg);
    }
  };

  // Treat any other type as an Argument.
  template <typename Head, typename... Tail>
  struct SyncKernelCallHelper<Head, Tail...> {
    using ArgT = std::decay_t<Head>;

    template <typename T>
    static T GetArg(Value* value, std::true_type) {
      return T(&value->template get<typename ArgT::UnderlyingT>());
    }

    template <typename T>
    static T& GetArg(Value* value, std::false_type) {
      return value->get<ArgT>();
    }

    template <int in_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(SyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1,
                    "Do not place Arguments after RemainingSyncArguments");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");
      auto* value = frame->GetArgAt(in_idx);
      auto&& arg = GetArg<ArgT>(value, IsViewT<ArgT>());

      SyncKernelCallHelper<Tail...>::template Invoke<in_idx + 1, const_idx>(
          frame, pargs..., arg);
    }
  };

  // RemainingSyncArguments provides an ArrayRef<Value*> containing all
  // remaining arguments. Useful for variadic kernels.
  template <typename... Tail>
  struct SyncKernelCallHelper<RemainingSyncArguments, Tail...> {
    template <int in_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(SyncKernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1,
                    "Do not use more than one RemainingSyncArguments");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");
      RemainingSyncArguments remaining_arguments(
          frame->GetArguments().drop_front(in_idx), frame->GetRegisters());

      SyncKernelCallHelper<Tail...>::template Invoke<-1, const_idx>(
          frame, pargs..., remaining_arguments);
    }
  };

  // Base case: No arguments left.
  // TypeTag<T> is a dummy template parameter to work around the restriction
  // of GCC that fully specialized template is not allowed in a template class.
  template <typename T>
  struct SyncKernelCallHelper<TypeTag<T>> {
    template <int in_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(SyncKernelFrame* frame, const PreviousArgs&... pargs) {
      assert((in_idx == -1 || in_idx == frame->GetNumArgs()) &&
             "Extra arguments passed to kernel.");
      assert(const_idx == frame->GetNumAttributes() &&
             "Extra attributes passed to kernel.");
      SyncKernelReturnHelper<Return>::Invoke(frame, pargs...);
    }
  };
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_SYNC_KERNEL_UTILS_H_
