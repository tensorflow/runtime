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
#include <tuple>
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
  virtual mlir::LogicalResult call(void** args, void** attrs) = 0;

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

namespace internal {

// A type tag to distinguish arguments from the attributes in the
// `CustomCallBinding` variadic template argument.
template <typename T>
struct Attr {};

}  // namespace internal

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

  template <typename T>
  CustomCallBinding<Ts..., internal::Attr<T>> Attr(std::string attr) && {
    attrs_.push_back(std::move(attr));
    return {std::move(*this)};
  }

  template <typename Fn>
  std::unique_ptr<CustomCall> To(Fn fn) {
    return std::unique_ptr<CustomCall>(new CustomCallHandler<Fn, Ts...>(
        std::forward<Fn>(fn), std::move(callee_), std::move(attrs_)));
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
      : callee_(std::move(other.callee_)), attrs_(std::move(other.attrs_)) {}

  CustomCallBinding(CustomCallBinding&) = delete;

  std::string callee_;              // custom call target
  std::vector<std::string> attrs_;  // names of bound attributes
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

// Custom call attribute decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` attributes
//
//   template<>
//   struct CustomCallAttrDecoding<MyType> {
//    static mlir::FailureOr<MyType> Decode(llvm::StringRef name,
//                                          mlir::TypeID type_id, void* value);
//   }
//
template <typename T>
struct CustomCallAttrDecoding;

// -------------------------------------------------------------------------- //
// A little bit of template metaprogramming to implement type safe binding
// of custom calls to C++ functions. This is internal implementation details,
// and must not be relied on in any of the client code.

namespace internal {

// TODO(ezhulenev): C++17 https://en.cppreference.com/w/cpp/types/is_invocable.
template <class R, typename F, typename... Args>
struct is_invocable
    : std::is_constructible<
          std::function<R(Args...)>,
          std::reference_wrapper<typename std::remove_reference<F>::type>> {};

// A helper struct to extract the type of the handler argument.
template <typename T>
struct FnArgType {
  using Type = T;
};

// Extracts the underlying type from the attribute type tag.
template <typename T>
struct FnArgType<internal::Attr<T>> {
  using Type = T;
};

struct DecodedArg {
  mlir::TypeID type_id;
  void* value;
};

struct DecodedAttr {
  llvm::StringRef name;
  mlir::TypeID type_id;
  void* value;
};

// Decodes arguments from the encoded data.
llvm::SmallVector<DecodedArg> DecodeArgs(void** args);

// Decodes attributes from the encoded data.
llvm::StringMap<DecodedAttr> DecodeAttrs(void** attrs);

// When decoding input data we need to keep track of how many arguments and
// attributes we decoded so far to index into the correct data strucuture.
struct DecodingOffsets {
  int64_t args = 0;
  int64_t attrs = 0;
};

using DecodedArgs = llvm::SmallVector<DecodedArg>;  // NOLINT
using DecodedAttrs = llvm::StringMap<DecodedAttr>;  // NOLINT

template <typename T, std::size_t index>
struct Decode {
  static mlir::FailureOr<T> call(DecodingOffsets& offsets, DecodedArgs& args,
                                 ArrayRef<std::string> attr_names,
                                 DecodedAttrs& attrs) {
    internal::DecodedArg arg = args[offsets.args++];
    return CustomCallArgDecoding<T>::Decode(arg.type_id, arg.value);
  }
};

template <typename T, std::size_t index>
struct Decode<internal::Attr<T>, index> {
  static mlir::FailureOr<T> call(DecodingOffsets& offsets, DecodedArgs& args,
                                 ArrayRef<std::string> attr_names,
                                 DecodedAttrs& attrs) {
    internal::DecodedAttr attr = attrs[attr_names[offsets.attrs++]];
    return CustomCallAttrDecoding<T>::Decode(attr.name, attr.type_id,
                                             attr.value);
  }
};

// Decodes type id from the opaque argument/attribute pointer.
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

  template <typename T>
  using FnArgType = typename internal::FnArgType<T>::Type;

  static_assert(
      internal::is_invocable<mlir::LogicalResult, Fn, FnArgType<Ts>...>::value,
      "incompatible custom call handler types");

 public:
  llvm::StringRef name() const override { return callee_; }

  mlir::LogicalResult call(void** args, void** attrs) override {
    // Decode arguments and attributes from the opaque pointers.
    auto decoded_args = internal::DecodeArgs(args);
    auto decoded_attrs = internal::DecodeAttrs(attrs);

    // Check that all required attributes were passed to the custom call.
    bool all_attrs = llvm::all_of(attrs_, [&](auto& attr) {
      return decoded_attrs.find(attr) != decoded_attrs.end();
    });
    if (!all_attrs) return mlir::failure();

    // Check that the number of passed arguments matches the signature. Each
    // individual argument decoding will check the actual type.
    if (decoded_args.size() != (kSize - attrs_.size())) return mlir::failure();

    return call(std::move(decoded_args), std::move(decoded_attrs),
                std::make_index_sequence<kSize>{});
  }

  template <std::size_t... Is>
  mlir::LogicalResult call(internal::DecodedArgs args,
                           internal::DecodedAttrs attrs,
                           std::index_sequence<Is...>) {
    // A helper structure to allow each decoder find the correct offset in the
    // arguments or attributes.
    internal::DecodingOffsets offsets;

    // Decode all arguments into mlir::FailureOr containers. It is guaranteed
    // that initializer list will be evaluated left-to-right, and we can rely
    // on correct offsets computation.
    std::tuple<mlir::FailureOr<FnArgType<Ts>>...> fn_args = {
        internal::Decode<Ts, Is>::call(offsets, args, attrs_, attrs)...};

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

  CustomCallHandler(Fn fn, std::string callee,
                    std::vector<std::string> attrs)  // NOLINT
      : fn_(std::move(fn)),
        callee_(std::move(callee)),
        attrs_(std::move(attrs)) {}

  Fn fn_;
  std::string callee_;
  std::vector<std::string> attrs_;
};

// -------------------------------------------------------------------------- //
// Custom arguments attributes decoding.

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
    // Check that encoded value holds the correct type id.
    if (type_id != mlir::TypeID::get<MemrefDesc>()) return mlir::failure();

    // Get the encoded memref from the opaque pointer.
    auto* encoded = reinterpret_cast<EncodedMemref*>(value);

    // Get the memref element data type.
    void* opaque = reinterpret_cast<void*>(encoded->element_type_id);
    mlir::TypeID element_type_id = mlir::TypeID::getFromOpaquePointer(opaque);

    auto dtype = internal::TypeIdToDType(element_type_id);
    if (mlir::failed(dtype)) return mlir::failure();

    // Unpack the StridedMemRefType into the MemrefDesc.
    auto unpack_strided_memref = [&](auto rank_tag) -> MemrefDesc {
      constexpr int rank = decltype(rank_tag)::value;

      using Descriptor = StridedMemRefType<float, rank>;
      auto* descriptor = reinterpret_cast<Descriptor*>(encoded->descriptor);

      return MemrefDesc(*dtype, descriptor->data, descriptor->offset,
                        Sizes(descriptor), Strides(descriptor));
    };

    // Dispatch based on the memref rank.
    switch (encoded->rank) {
      case 0:
        return unpack_strided_memref(std::integral_constant<int, 0>{});
      case 1:
        return unpack_strided_memref(std::integral_constant<int, 1>{});
      case 2:
        return unpack_strided_memref(std::integral_constant<int, 2>{});
      case 3:
        return unpack_strided_memref(std::integral_constant<int, 3>{});
      case 4:
        return unpack_strided_memref(std::integral_constant<int, 4>{});
      case 5:
        return unpack_strided_memref(std::integral_constant<int, 5>{});
      default:
        assert(false && "unsupported memref rank");
        return mlir::failure();
    }
  }
};

// -------------------------------------------------------------------------- //
// Custom call attributes decoding.

template <>
struct CustomCallAttrDecoding<llvm::StringRef> {
  static mlir::FailureOr<llvm::StringRef> Decode(llvm::StringRef name,
                                                 mlir::TypeID type_id,
                                                 void* value) {
    if (type_id != mlir::TypeID::get<llvm::StringRef>()) return mlir::failure();
    return llvm::StringRef(reinterpret_cast<const char*>(value));
  }
};

#define JITRT_REGISTER_SCALAR_ATTR_DECODING(T)                            \
  template <>                                                             \
  struct CustomCallAttrDecoding<T> {                                      \
    static mlir::FailureOr<T> Decode(llvm::StringRef name,                \
                                     mlir::TypeID type_id, void* value) { \
      if (type_id != mlir::TypeID::get<T>()) return mlir::failure();      \
      return *reinterpret_cast<T*>(value);                                \
    }                                                                     \
  }

JITRT_REGISTER_SCALAR_ATTR_DECODING(int32_t);
JITRT_REGISTER_SCALAR_ATTR_DECODING(int64_t);
JITRT_REGISTER_SCALAR_ATTR_DECODING(float);
JITRT_REGISTER_SCALAR_ATTR_DECODING(double);

#undef JITRT_REGISTER_SCALAR_ATTR_DECODING

#define JITRT_REGISTER_ARRAY_ATTR_DECODING(T)                                  \
  template <>                                                                  \
  struct CustomCallAttrDecoding<ArrayRef<T>> {                                 \
    struct EncodedMemref {                                                     \
      int64_t size;                                                            \
      T data;                                                                  \
    };                                                                         \
                                                                               \
    static mlir::FailureOr<ArrayRef<T>> Decode(llvm::StringRef name,           \
                                               mlir::TypeID type_id,           \
                                               void* value) {                  \
      if (type_id != mlir::TypeID::get<ArrayRef<T>>()) return mlir::failure(); \
      auto* encoded = reinterpret_cast<EncodedMemref*>(value);                 \
      return ArrayRef<T>(&encoded->data, encoded->size);                       \
    }                                                                          \
  }

JITRT_REGISTER_ARRAY_ATTR_DECODING(int32_t);
JITRT_REGISTER_ARRAY_ATTR_DECODING(int64_t);
JITRT_REGISTER_ARRAY_ATTR_DECODING(float);
JITRT_REGISTER_ARRAY_ATTR_DECODING(double);

#undef JITRT_REGISTER_ARRAY_ATTR_DECODING

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_CUSTOM_CALL_H_
