/*
 * Copyright 2022 The TensorFlow Runtime Authors
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

#ifndef TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_CUSTOM_CALL_H_
#define TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_CUSTOM_CALL_H_

#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/jitrt/types.h"

namespace tfrt {
namespace jitrt {

// Forward declare template defined below.
template <typename Fn, typename... Ts>
class CustomCallHandler;

// Forward declare template defined below.
template <typename... Ts>
class CustomCallBinding;

class CustomCall {
 public:
  virtual ~CustomCall() = default;

  virtual llvm::StringRef name() const = 0;
  virtual mlir::LogicalResult call(void** args) = 0;

  static CustomCallBinding<> Bind(std::string callee);
};

class CustomCallRegistry {
 public:
  // The type for custom call registration functions.
  using RegistrationFunction = void (*)(CustomCallRegistry*);

  CustomCallRegistry();
  ~CustomCallRegistry() = default;

  CustomCallRegistry(const CustomCallRegistry&) = delete;
  CustomCallRegistry& operator=(const CustomCallRegistry&) = delete;

  void Register(std::unique_ptr<CustomCall> custom_call);

  CustomCall* Find(llvm::StringRef callee) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Use this macro to add a function that will register custom calls that are
// statically linked in the binary. FUNC should be a function pointer with the
// prototype given by the CustomCallRegistry::RegistrationFunction alias.
#define JITRT_STATIC_CUSTOM_CALL_REGISTRATION(FUNC) \
  JITRT_STATIC_CUSTOM_CALL_REGISTRATION_IMPL(FUNC, __COUNTER__)
#define JITRT_STATIC_CUSTOM_CALL_REGISTRATION_IMPL(FUNC, N)       \
  static bool jitrt_static_custom_call_##N##_registered_ = []() { \
    ::tfrt::jitrt::AddStaticCustomCallRegistration(FUNC);         \
    return true;                                                  \
  }()

// Registers all statically linked custom calls in the given registry.
void RegisterStaticCustomCalls(CustomCallRegistry* custom_call_registry);

// Adds a custom call registration function to the registry. This should not be
// used directly; use JITRT_STATIC_CUSTOM_CALL_REGISTRATION instead.
void AddStaticCustomCallRegistration(
    CustomCallRegistry::RegistrationFunction registration);

// Custom call binding describes the function signature of the expected custom
// call handler using its variadic template parameter.
//
//   Custom call binding:
//     CustomCallBinding<int32_t, MemrefDesc>
//
//   Function signature:
//     LogicalResult MyHandle(int32_t algo, MemrefDesc memref);
//
template <typename... Ts>
class CustomCallBinding {
 public:
  template <typename T>
  CustomCallBinding<Ts..., T> Arg() && {
    return {std::move(*this)};
  }

  template <typename Fn>
  std::unique_ptr<CustomCall> To(Fn fn) {
    return std::unique_ptr<CustomCall>(new CustomCallHandler<Fn, Ts...>(
        std::forward<Fn>(fn), std::move(callee_)));
  }

 private:
  template <typename...>
  friend class CustomCallBinding;
  friend class CustomCall;

  explicit CustomCallBinding(std::string callee) : callee_(std::move(callee)) {
    static_assert(sizeof...(Ts) == 0, "custom call arguments must be empty");
  }

  template <typename... TTs>
  CustomCallBinding(CustomCallBinding<TTs...>&& other)  // NOLINT
      : callee_(std::move(other.callee_)) {}

  CustomCallBinding(CustomCallBinding&) = delete;

  std::string callee_;  // custom call target
};

inline CustomCallBinding<> CustomCall::Bind(std::string callee) {
  return CustomCallBinding<>(std::move(callee));
}

// Custom call arguments decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` arguments
//
//   template<>
//   struct CustomCallArgDecoding<MyType> {
//    static mlir::FailureOr<MyType> Decode(mlir::TypeID type_id, void* value);
//   };
//
template <typename T>
struct CustomCallArgDecoding;

// -------------------------------------------------------------------------- //
// A little bit of template metaprogramming to implement type safe binding
// of custom calls to C++ functions. This is internal implementation details,
// and must not be relied on in any of the client code.

namespace internal {

// TODO(ezhulenev): C++17 https://en.cppreference.com/w/cpp/types/is_invocable.
template <typename F, typename... Args>
struct is_invocable
    : std::is_constructible<
          std::function<void(Args...)>,
          std::reference_wrapper<typename std::remove_reference<F>::type>> {};

struct DecodedArg {
  mlir::TypeID type_id;
  void* value;
};

// Decodes arguments from the encoded data.
llvm::SmallVector<DecodedArg> DecodeArgs(void** args);

// When decoding input data we need to keep track of how many arguments we
// decoded so far to index into the correct data strucuture.
struct DecodingOffsets {
  int64_t args = 0;
};

using DecodedArgs = llvm::SmallVector<DecodedArg>;  // NOLINT

template <typename T, std::size_t index>
struct Decode {
  static mlir::FailureOr<T> call(DecodingOffsets& offsets, DecodedArgs& args) {
    internal::DecodedArg arg = args[offsets.args++];
    return CustomCallArgDecoding<T>::Decode(arg.type_id, arg.value);
  }
};

// Decodes type id from the opaque argument pointer.
mlir::TypeID DecodeTypeid(void* type_id);

// Converts mlir TypeID to tfrt DType. Returns error if type is not supported.
mlir::FailureOr<DType> TypeIdToDType(mlir::TypeID type_id);

}  // namespace internal

// Custom call handler binds concrete custom call implementation of type `Fn` to
// the custom call function signature. `Fn` can be a function pointer, or a
// lambda.
//
// Custom call handler uses the variadic template parameter `Ts` to decode the
// opaque pointers passed to the `call` function into the C++ types that are
// forwarded to the custom call implementation.
template <typename Fn, typename... Ts>
class CustomCallHandler : public CustomCall {
  static constexpr int64_t kSize = sizeof...(Ts);

  static_assert(internal::is_invocable<Fn, Ts...>::value,
                "incompatible custom call handler types");

 public:
  llvm::StringRef name() const override { return callee_; }

  mlir::LogicalResult call(void** args) override {
    // Decode arguments from the opaque pointers.
    auto decoded_args = internal::DecodeArgs(args);
    if (decoded_args.size() != kSize) return mlir::failure();

    return call(std::move(decoded_args), std::make_index_sequence<kSize>{});
  }

  template <std::size_t... Is>
  mlir::LogicalResult call(internal::DecodedArgs args,
                           std::index_sequence<Is...>) {
    // A helper structure to allow each decoder find the correct offset in the
    // arguments.
    internal::DecodingOffsets offsets;

    // Decode all arguments into mlir::FailureOr containers. It is guaranteed
    // that initializer list will be evaluated left-to-right, and we can rely
    // on correct offsets computation.
    std::tuple<mlir::FailureOr<Ts>...> fn_args = {
        internal::Decode<Ts, Is>::call(offsets, args)...};

    // Check that all of them were successfully decoded.
    std::array<bool, kSize> decoded = {
        mlir::succeeded(std::get<Is>(fn_args))...};
    if (llvm::any_of(decoded, [](bool succeeded) { return !succeeded; }))
      return mlir::failure();

    // Forward unpacked arguments to the callback.
    return fn_(std::move(*std::get<Is>(fn_args))...);
  }

 private:
  template <typename...>
  friend class CustomCallBinding;

  CustomCallHandler(Fn fn, std::string callee)  // NOLINT
      : fn_(std::move(fn)), callee_(std::move(callee)) {}

  Fn fn_;
  std::string callee_;
};

// -------------------------------------------------------------------------- //
// Custom call arguments decoding.

template <>
struct CustomCallArgDecoding<MemrefDesc> {
  // Struct corresponding to the `rt-to-llvm` pass LLVM struct encoding the
  // memref information.
  struct EncodedMemref {
    int64_t element_type_id;
    int64_t rank;
    void* descriptor;
  };

  template <typename T, int rank>
  static ArrayRef<int64_t> Sizes(StridedMemRefType<T, rank>* memref) {
    return llvm::makeArrayRef(memref->sizes);
  }

  template <typename T>
  static ArrayRef<int64_t> Sizes(StridedMemRefType<T, 0>* memref) {
    return {};
  }

  template <typename T, int rank>
  static ArrayRef<int64_t> Strides(StridedMemRefType<T, rank>* memref) {
    return llvm::makeArrayRef(memref->strides);
  }

  template <typename T>
  static ArrayRef<int64_t> Strides(StridedMemRefType<T, 0>* memref) {
    return {};
  }

  static mlir::FailureOr<MemrefDesc> Decode(mlir::TypeID type_id, void* value) {
    MemrefDesc memref;

    // Check that encoded value holds the correct type id.
    if (type_id != mlir::TypeID::get<MemrefDesc>()) return mlir::failure();

    // Get the encoded memref from the opaque pointer.
    auto* encoded = reinterpret_cast<EncodedMemref*>(value);

    // Get the memref element data type.
    void* opaque = reinterpret_cast<void*>(encoded->element_type_id);
    mlir::TypeID element_type_id = mlir::TypeID::getFromOpaquePointer(opaque);

    auto dtype = internal::TypeIdToDType(element_type_id);
    if (mlir::failed(dtype)) return mlir::failure();
    memref.dtype = *dtype;

    // Unpack the StridedMemRefType into the MemrefDesc.
    auto unpack_strided_memref = [&](auto rank_tag) {
      constexpr int rank = decltype(rank_tag)::value;

      using Descriptor = StridedMemRefType<float, rank>;
      auto* descriptor = reinterpret_cast<Descriptor*>(encoded->descriptor);

      memref.data = descriptor->data;
      memref.offset = descriptor->offset;

      auto sizes = Sizes(descriptor);
      memref.sizes.assign(sizes.begin(), sizes.end());

      auto strides = Strides(descriptor);
      memref.strides.assign(strides.begin(), strides.end());
    };

    // Dispatch based on the memref rank.
    switch (encoded->rank) {
      case 0:
        unpack_strided_memref(std::integral_constant<int, 0>{});
        break;
      case 1:
        unpack_strided_memref(std::integral_constant<int, 1>{});
        break;
      case 2:
        unpack_strided_memref(std::integral_constant<int, 2>{});
        break;
      case 3:
        unpack_strided_memref(std::integral_constant<int, 3>{});
        break;
      case 4:
        unpack_strided_memref(std::integral_constant<int, 4>{});
        break;
      case 5:
        unpack_strided_memref(std::integral_constant<int, 5>{});
        break;
      default:
        assert(false && "unsupported memref rank");
        return mlir::failure();
    }

    return memref;
  }
};

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_CUSTOM_CALL_H_
