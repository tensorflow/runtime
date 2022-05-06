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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/jitrt/types.h"
#include "tfrt/support/map_by_type.h"

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
  // Container for passing data between JitRt user and the custom call handler.
  using UserData = MapByType<CustomCall>;

  // A type for matching all remaining custom call arguments.
  class RemainingArgs;

  virtual ~CustomCall() = default;

  virtual llvm::StringRef name() const = 0;
  virtual mlir::LogicalResult call(void** args, void** attrs,
                                   const UserData* user_data) = 0;

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

// A type tag to distinguish arguments tied to the attributes in the
// `CustomCallBinding` variadic template argument.
template <typename T>
struct Attr {};

// A type tag to distinguish arguments tied to the user data in the
// `CustomCallBinding` variadic template argument.
template <typename T>
struct UserData {};

// A template for checking if type is a wrapped attribute or user data.
template <typename>
struct IsTagged : std::false_type {};

template <typename T>
struct IsTagged<internal::Attr<T>> : std::true_type {};

template <typename T>
struct IsTagged<internal::UserData<T>> : std::true_type {};

// Checks if remaining arguments are in the parameter pack.
template <typename... Ts>
struct HasRemainingArgs;

template <typename T, typename... Ts>
struct HasRemainingArgs<T, Ts...> {
  static constexpr bool value =
      std::is_same<CustomCall::RemainingArgs, T>::value ||
      HasRemainingArgs<Ts...>::value;
};

template <>
struct HasRemainingArgs<> : std::false_type {};

// Decoded pair of an argument type and opaque value.
struct DecodedArg {
  mlir::TypeID type_id;
  void* value;
};

// Decoded triple of an attribute name, type and opaque value.
struct DecodedAttr {
  llvm::StringRef name;
  mlir::TypeID type_id;
  void* value;
};

// Decodes arguments from the encoded data.
llvm::SmallVector<DecodedArg> DecodeArgs(void** args);

// Decodes attributes from the encoded data.
llvm::SmallVector<DecodedAttr> DecodeAttrs(void** attrs);

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

  CustomCallBinding<Ts..., CustomCall::RemainingArgs> RemainingArgs() && {
    static_assert(!internal::HasRemainingArgs<Ts...>::value,
                  "remaining arguments can be passed just once");
    return {std::move(*this)};
  }

  template <typename T>
  CustomCallBinding<Ts..., internal::Attr<T>> Attr(std::string attr) && {
    attrs_.push_back(std::move(attr));
    return {std::move(*this)};
  }

  template <typename T>
  CustomCallBinding<Ts..., internal::UserData<T>> UserData() && {
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

// CustomCall remaining arguments wraps the type-erased `DecodedArg` container,
// and provides a type-safe API for accessing individual arguments.
class CustomCall::RemainingArgs {
 public:
  explicit RemainingArgs(ArrayRef<internal::DecodedArg> args)
      : args_(args.begin(), args.end()) {}

  size_t size() const { return args_.size(); }

  template <typename T>
  bool isa(size_t index) const {
    return args_[index].type_id == mlir::TypeID::get<T>();
  }

  template <typename T>
  mlir::FailureOr<T> get(size_t index) const {
    return CustomCallArgDecoding<T>::Decode(args_[index].type_id,
                                            args_[index].value);
  }

 private:
  llvm::SmallVector<internal::DecodedArg> args_;
};

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

// Extracts the underlying type from the user data type tag.
template <typename T>
struct FnArgType<internal::UserData<T>> {
  using Type = T;
};

// A template for counting regular arguments in the Ts pack.
template <typename T, typename... Ts>
struct NumArgs {
  static constexpr int64_t value = !IsTagged<T>::value + NumArgs<Ts...>::value;
};

template <typename T>
struct NumArgs<T> {
  static constexpr int64_t value = !IsTagged<T>::value;
};

// When decoding input data we need to keep track of how many arguments and
// attributes we decoded so far to index into the correct data strucuture.
struct DecodingOffsets {
  int64_t args = 0;
  int64_t attrs = 0;
};

template <typename T, size_t index>
struct Decode {
  static mlir::FailureOr<T> call(DecodingOffsets& offsets,
                                 llvm::SmallVector<DecodedArg>& args,
                                 ArrayRef<std::string> attrs_names,
                                 ArrayRef<size_t> attrs_idx,
                                 llvm::SmallVector<DecodedAttr>& attrs,
                                 const CustomCall::UserData* user_data) {
    internal::DecodedArg arg = args[offsets.args++];
    return CustomCallArgDecoding<T>::Decode(arg.type_id, arg.value);
  }
};

template <typename T, size_t index>
struct Decode<internal::Attr<T>, index> {
  static mlir::FailureOr<T> call(DecodingOffsets& offsets,
                                 llvm::SmallVector<DecodedArg>& args,
                                 ArrayRef<std::string> attrs_names,
                                 ArrayRef<size_t> attrs_idx,
                                 llvm::SmallVector<DecodedAttr>& attrs,
                                 const CustomCall::UserData* user_data) {
    // Find decoded attribute corresponding for the given attribute index.
    int64_t idx = offsets.attrs++;
    llvm::StringRef attr = attrs_names[idx];

    // Given that attributes are passed to the custom call handler
    // lexicographically sorted by name, we can find the attribute we are
    // looking for only between the `attrs_idx` offset and the end of the
    // attributes array.
    for (size_t i = attrs_idx[idx]; i < attrs.size(); ++i) {
      if (attrs[i].name == attr)
        return CustomCallAttrDecoding<T>::Decode(
            attrs[i].name, attrs[i].type_id, attrs[i].value);
    }

    // Attribute we were looking for was not passed as an argument.
    return mlir::failure();
  }
};

template <typename T, size_t index>
struct Decode<internal::UserData<T>, index> {
  static mlir::FailureOr<T> call(DecodingOffsets& offsets,
                                 llvm::SmallVector<DecodedArg>& args,
                                 ArrayRef<std::string> attrs_names,
                                 ArrayRef<size_t> attrs_idx,
                                 llvm::SmallVector<DecodedAttr>& attrs,
                                 const CustomCall::UserData* user_data) {
    if (!user_data || !user_data->contains<T>()) return mlir::failure();
    return user_data->get<T>();
  }
};

template <size_t index>
struct Decode<CustomCall::RemainingArgs, index> {
  static mlir::FailureOr<CustomCall::RemainingArgs> call(
      DecodingOffsets& offsets, llvm::SmallVector<DecodedArg>& args,
      ArrayRef<std::string> attr_names, ArrayRef<size_t> attrs_idx,
      llvm::SmallVector<DecodedAttr>& attrs,
      const CustomCall::UserData* user_data) {
    auto remaining = llvm::ArrayRef<DecodedArg>(args).drop_front(offsets.args);
    return CustomCall::RemainingArgs(remaining);
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
  static constexpr int64_t kNumArgs = internal::NumArgs<Ts...>::value;

  template <typename T>
  using FnArgType = typename internal::FnArgType<T>::Type;

  static_assert(
      internal::is_invocable<mlir::LogicalResult, Fn, FnArgType<Ts>...>::value,
      "incompatible custom call handler types");

 public:
  llvm::StringRef name() const override { return callee_; }

  mlir::LogicalResult call(void** args, void** attrs,
                           const UserData* user_data) override {
    // Decode arguments and attributes from the opaque pointers.
    auto decoded_args = internal::DecodeArgs(args);
    auto decoded_attrs = internal::DecodeAttrs(attrs);

    // Check that the number of passed arguments matches the signature. Each
    // individual argument decoding will check the actual type.
    if (internal::HasRemainingArgs<Ts...>::value) {
      if (decoded_args.size() < kNumArgs - 1) return mlir::failure();
    } else {
      if (decoded_args.size() != kNumArgs) return mlir::failure();
    }

    return call(std::move(decoded_args), std::move(decoded_attrs), user_data,
                std::make_index_sequence<kSize>{});
  }

  template <size_t... Is>
  mlir::LogicalResult call(llvm::SmallVector<internal::DecodedArg> args,
                           llvm::SmallVector<internal::DecodedAttr> attrs,
                           const UserData* user_data,
                           std::index_sequence<Is...>) {
    // A helper structure to allow each decoder find the correct offset in the
    // arguments or attributes.
    internal::DecodingOffsets offsets;

    // Decode all arguments into mlir::FailureOr containers. It is guaranteed
    // that initializer list will be evaluated left-to-right, and we can rely
    // on correct offsets computation.
    std::tuple<mlir::FailureOr<FnArgType<Ts>>...> fn_args = {
        internal::Decode<Ts, Is>::call(offsets, args, attrs_, attrs_idx_, attrs,
                                       user_data)...};

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
        attrs_(std::move(attrs)),
        attrs_idx_(attrs_.size()) {
    // Sort attributes names.
    std::vector<std::string> sorted = attrs_;
    llvm::sort(sorted);

    // Find index or every attribute in the sorted attributes vector.
    for (size_t i = 0; i < attrs_.size(); ++i) {
      const std::string& attr = attrs_[i];
      attrs_idx_[i] = std::distance(sorted.begin(), llvm::find(sorted, attr));
    }
  }

  Fn fn_;
  std::string callee_;
  std::vector<std::string> attrs_;
  // A mapping from the attribute index to its index in the lexicographically
  // sorter vector of attribute names. Attributes passed in the custom call
  // handler sorted by the name, we use this index to efficiently find the
  // decoded attribute entry.
  std::vector<size_t> attrs_idx_;
};

// -------------------------------------------------------------------------- //
// C structures corresponding to the `rt-to-llvm` pass LLVM structs encoding
// various types of arguments/attributes.

namespace internal {

struct EncodedString {
  int64_t size;
  const char* data;
};

struct EncodedMemref {
  int64_t element_type_id;
  int64_t rank;
  void* descriptor;
};

template <typename T>
struct EncodedArray {
  int64_t size;
  T data;
};

}  // namespace internal

// -------------------------------------------------------------------------- //
// Custom arguments attributes decoding.

// A flat view into the memref. If the memref shapes is not required for the
// custom call, it's much cheaper to pass the flat view struct instead of
// building a MemrefDesc.
struct FlatMemrefView {
  tfrt::DType dtype;
  void* data;
  int64_t size_in_bytes;
};

raw_ostream& operator<<(raw_ostream& os, const FlatMemrefView& view);

template <>
struct CustomCallArgDecoding<MemrefDesc> {
  static mlir::FailureOr<MemrefDesc> Decode(mlir::TypeID type_id, void* value);
};

template <>
struct CustomCallArgDecoding<FlatMemrefView> {
  static mlir::FailureOr<FlatMemrefView> Decode(mlir::TypeID type_id,
                                                void* value);
};

#define JITRT_REGISTER_SCALAR_ARG_DECODING(T)                             \
  template <>                                                             \
  struct CustomCallArgDecoding<T> {                                       \
    static mlir::FailureOr<T> Decode(mlir::TypeID type_id, void* value) { \
      if (type_id != mlir::TypeID::get<T>()) return mlir::failure();      \
      return *reinterpret_cast<T*>(value);                                \
    }                                                                     \
  }

JITRT_REGISTER_SCALAR_ARG_DECODING(int32_t);
JITRT_REGISTER_SCALAR_ARG_DECODING(int64_t);
JITRT_REGISTER_SCALAR_ARG_DECODING(float);
JITRT_REGISTER_SCALAR_ARG_DECODING(double);

#undef JITRT_REGISTER_SCALAR_ARG_DECODING

// -------------------------------------------------------------------------- //
// Custom call attributes decoding.

template <>
struct CustomCallAttrDecoding<llvm::StringRef> {
  static mlir::FailureOr<llvm::StringRef> Decode(llvm::StringRef name,
                                                 mlir::TypeID type_id,
                                                 void* value) {
    if (type_id != mlir::TypeID::get<llvm::StringRef>()) return mlir::failure();
    auto* encoded = reinterpret_cast<internal::EncodedString*>(value);
    return llvm::StringRef(encoded->data, encoded->size);
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
    static mlir::FailureOr<ArrayRef<T>> Decode(llvm::StringRef name,           \
                                               mlir::TypeID type_id,           \
                                               void* value) {                  \
      if (type_id != mlir::TypeID::get<ArrayRef<T>>()) return mlir::failure(); \
      auto* encoded = reinterpret_cast<internal::EncodedArray<T>*>(value);     \
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
