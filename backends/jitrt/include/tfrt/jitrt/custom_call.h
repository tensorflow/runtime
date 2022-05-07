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
#include "llvm/Support/Compiler.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/jitrt/types.h"
#include "tfrt/support/map_by_type.h"

namespace tfrt {
namespace jitrt {

// Forward declare template defined below.
template <typename... Ts>
class CustomCallBinding;

class CustomCall {
 public:
  // Container for passing data between JitRt user and the custom call handler.
  using UserData = MapByType<CustomCall>;

  // A type for matching all remaining custom call arguments.
  class RemainingArgs;

  // Custom call handler can check arguments and attributes types and names
  // at runtime, however this comes at extra cost and can be optionally
  // disabled. If the version of the compiler that generated the JitRt program
  // doesn't match the custom call handler, it can lead to undefined behavior.
  enum class RuntimeChecks : uint8_t {
    // Check arguments and attributes types, also check attribute names. It is
    // safe to pass extra arguments to the custom call handler when name
    // checking is enabled, because it will safely skip irrelevant attributes.
    kDefault = 0,

    // Check only the types of the arguments and attributes. If an attribute
    // with the same type but different name is passed to the custom call
    // handler,
    // it will happily proceed ignoring the name mismatch.
    kTypes = 1,

    // Do not check arguments and attributes types, and do not check that the
    // user data was passed to the custom call. This is the most dangerous
    // option, because it blindly reinterprets opaque memory passed to the
    // handler, and can easily lead to segfaults if the data doesn't match the
    // expected custom call signature.
    kNone = 2
  };

  static constexpr bool CheckNames(RuntimeChecks checks) {
    return checks == RuntimeChecks::kDefault;
  }

  static constexpr bool CheckTypes(RuntimeChecks checks) {
    return checks != RuntimeChecks::kNone;
  }

  static constexpr bool CheckUserData(RuntimeChecks checks) {
    return checks != RuntimeChecks::kNone;
  }

  template <typename T>
  static bool CheckType(RuntimeChecks checks, mlir::TypeID type_id) {
    return !CheckTypes(checks) || type_id == mlir::TypeID::get<T>();
  }

  virtual ~CustomCall() = default;

  virtual llvm::StringRef name() const = 0;
  virtual mlir::LogicalResult call(void** args, void** attrs,
                                   const UserData* user_data) = 0;

  static CustomCallBinding<> Bind(std::string callee);
};

// Forward declare template defined below.
template <CustomCall::RuntimeChecks checks, typename Fn, typename... Ts>
class CustomCallHandler;

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
struct IsWrapped : std::false_type {};

template <typename T>
struct IsWrapped<internal::Attr<T>> : std::true_type {};

template <typename T>
struct IsWrapped<internal::UserData<T>> : std::true_type {};

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

}  // namespace internal

// Custom call binding describes the function signature of the expected custom
// call handler using its variadic template parameter.
//
//   Custom call binding:
//     CustomCallBinding<int32_t, MemrefView>
//
//   Function signature:
//     LogicalResult MyHandle(int32_t algo, MemrefView memref);
//
template <typename... Ts>
class CustomCallBinding {
 public:
  using RuntimeChecks = CustomCall::RuntimeChecks;

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

  template <RuntimeChecks checks = RuntimeChecks::kDefault, typename Fn>
  std::unique_ptr<CustomCall> To(Fn fn) {
    return std::unique_ptr<CustomCall>(new CustomCallHandler<checks, Fn, Ts...>(
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
//   template <CustomCall::RuntimeChecks checks>
//   struct CustomCallArgDecoding<MyType, checks> {
//    static mlir::FailureOr<MyType> Decode(mlir::TypeID type_id, void* value);
//   };
//
template <typename T, CustomCall::RuntimeChecks>
struct CustomCallArgDecoding;

// Custom call attribute decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` attributes
//
//   template <CustomCall::RuntimeChecks checks>
//   struct CustomCallAttrDecoding<MyType, checks> {
//    static mlir::FailureOr<MyType> Decode(llvm::StringRef name,
//                                          mlir::TypeID type_id, void* value);
//   }
//
template <typename T, CustomCall::RuntimeChecks>
struct CustomCallAttrDecoding;

// A type tag to declare MLIR TypeID specializations for types passed to the
// custom calls. We don't want to declare specializations for scalar types
// directly in this translation unit, so we rely on a tag to wrap them.
//
// See explicit TypeID declarations at the end of this file.
template <typename T>
struct Tagged {};

// -------------------------------------------------------------------------- //
// C structures corresponding to the `rt-to-llvm` pass LLVM structs encoding
// various types of arguments/attributes.

namespace internal {

struct EncodedString {
  int64_t size;
  const char* data;
};

struct EncodedMemref {
  uint8_t dtype;
  uint8_t rank;
  void* descriptor;
};

template <typename T>
struct EncodedArray {
  int64_t size;
  T data;
};

}  // namespace internal

// -------------------------------------------------------------------------- //
// Helpers for decoding opaque arguments and attributes memory.

namespace internal {

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

// A convenience wrapper around opaque arguments memory.
class DecodedArgs {
 public:
  explicit DecodedArgs(void** args)
      : args_(args), num_args_(*reinterpret_cast<int64_t*>(args_[0])) {}

  LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t size() const { return num_args_; }

  LLVM_ATTRIBUTE_ALWAYS_INLINE DecodedArg operator[](size_t i) const {
    void** arg_base = args_ + 1 + i * 2;

    DecodedArg arg;
    arg.type_id = mlir::TypeID::getFromOpaquePointer(arg_base[0]);
    arg.value = arg_base[1];

    return arg;
  }

 private:
  void** args_;
  int64_t num_args_;
};

// A convenience wrapper around opaque attributes memory.
class DecodedAttrs {
 public:
  explicit DecodedAttrs(void** attrs)
      : attrs_(attrs), num_attrs_(*reinterpret_cast<int64_t*>(attrs_[0])) {}

  LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t size() const { return num_attrs_; }

  LLVM_ATTRIBUTE_ALWAYS_INLINE DecodedAttr operator[](size_t i) const {
    void** attr_base = attrs_ + 1 + i * 3;

    DecodedAttr attr;
    auto* name = reinterpret_cast<internal::EncodedString*>(attr_base[0]);
    attr.name = llvm::StringRef(name->data, name->size);
    attr.type_id = mlir::TypeID::getFromOpaquePointer(attr_base[1]);
    attr.value = attr_base[2];

    return attr;
  }

 private:
  void** attrs_;
  int64_t num_attrs_;
};

}  // namespace internal

// -------------------------------------------------------------------------- //
// CustomCall remaining arguments wraps the type-erased `DecodedArg` container,
// and provides a type-safe API for accessing individual arguments.

class CustomCall::RemainingArgs {
 public:
  using RuntimeChecks = CustomCall::RuntimeChecks;

  RemainingArgs(internal::DecodedArgs args, size_t offset)
      : args_(args), offset_(offset) {
    assert(offset <= args_.size() && "illegal remaining args offset");
  }

  size_t size() const { return args_.size() - offset_; }

  template <typename T>
  bool isa(size_t index) const {
    return args_[index + offset_].type_id == mlir::TypeID::get<Tagged<T>>();
  }

  template <typename T, RuntimeChecks checks = RuntimeChecks::kDefault>
  mlir::FailureOr<T> get(size_t index) const {
    internal::DecodedArg arg = args_[index + offset_];
    return CustomCallArgDecoding<T, checks>::Decode(arg.type_id, arg.value);
  }

 private:
  internal::DecodedArgs args_;
  size_t offset_;
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
  static constexpr int64_t value = !IsWrapped<T>::value + NumArgs<Ts...>::value;
};

template <typename T>
struct NumArgs<T> {
  static constexpr int64_t value = !IsWrapped<T>::value;
};

// When decoding input data we need to keep track of how many arguments and
// attributes we decoded so far to index into the correct data strucuture.
struct DecodingOffsets {
  int64_t args = 0;
  int64_t attrs = 0;
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static mlir::FailureOr<T> call(
      DecodingOffsets& offsets, internal::DecodedArgs args,
      ArrayRef<std::string> attrs_names, ArrayRef<size_t> attrs_idx,
      internal::DecodedAttrs attrs, const CustomCall::UserData* user_data) {
    internal::DecodedArg arg = args[offsets.args++];
    return CustomCallArgDecoding<T, checks>::Decode(arg.type_id, arg.value);
  }
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode<internal::Attr<T>, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static mlir::FailureOr<T> call(
      DecodingOffsets& offsets, internal::DecodedArgs args,
      ArrayRef<std::string> attrs_names, ArrayRef<size_t> attrs_idx,
      internal::DecodedAttrs attrs, const CustomCall::UserData* user_data) {
    // Find decoded attribute corresponding for the given attribute index.
    int64_t idx = offsets.attrs++;

    // Do not check the attribute name, and decode attribute at the given index.
    if (!CustomCall::CheckNames(checks)) {
      size_t i = attrs_idx[idx];
      return CustomCallAttrDecoding<T, checks>::Decode(
          attrs[i].name, attrs[i].type_id, attrs[i].value);
    }

    llvm::StringRef attr = attrs_names[idx];

    // Given that attributes are passed to the custom call handler
    // lexicographically sorted by name, we can find the attribute we are
    // looking for only between the `attrs_idx` offset and the end of the
    // attributes array.
    for (size_t i = attrs_idx[idx]; i < attrs.size(); ++i) {
      if (LLVM_LIKELY(attrs[i].name == attr))
        return CustomCallAttrDecoding<T, checks>::Decode(
            attrs[i].name, attrs[i].type_id, attrs[i].value);
    }

    // Attribute we were looking for was not passed as an argument.
    return mlir::failure();
  }
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode<internal::UserData<T>, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static mlir::FailureOr<T> call(
      DecodingOffsets& offsets, internal::DecodedArgs args,
      ArrayRef<std::string> attrs_names, ArrayRef<size_t> attrs_idx,
      internal::DecodedAttrs attrs, const CustomCall::UserData* user_data) {
    if (!CustomCall::CheckUserData(checks)) return user_data->get<T>();

    // Check that the user data was passed to the custom call handler, and that
    // it containts the type we are looking for.
    const T* data = user_data ? user_data->getIfExists<T>() : nullptr;
    if (LLVM_UNLIKELY(!data)) return mlir::failure();
    return *data;
  }
};

template <CustomCall::RuntimeChecks checks>
struct Decode<CustomCall::RemainingArgs, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static mlir::FailureOr<CustomCall::RemainingArgs>
  call(DecodingOffsets& offsets, internal::DecodedArgs args,
       ArrayRef<std::string> attr_names, ArrayRef<size_t> attrs_idx,
       internal::DecodedAttrs attrs, const CustomCall::UserData* user_data) {
    return CustomCall::RemainingArgs(args, offsets.args);
  }
};

}  // namespace internal

// Custom call handler binds concrete custom call implementation of type `Fn` to
// the custom call function signature. `Fn` can be a function pointer, or a
// lambda.
//
// Custom call handler uses the variadic template parameter `Ts` to decode the
// opaque pointers passed to the `call` function into the C++ types that are
// forwarded to the custom call implementation.
template <CustomCall::RuntimeChecks checks, typename Fn, typename... Ts>
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
    internal::DecodedArgs decoded_args(args);
    internal::DecodedAttrs decoded_attrs(attrs);

    // Check that the number of passed arguments matches the signature. Each
    // individual argument decoding will check the actual type.
    if (internal::HasRemainingArgs<Ts...>::value) {
      if (LLVM_UNLIKELY(decoded_args.size() < kNumArgs - 1))
        return mlir::failure();
    } else {
      if (LLVM_UNLIKELY(decoded_args.size() != kNumArgs))
        return mlir::failure();
    }

    // Check that we have enough attributes passed to the custom call. Each
    // individual attribute decoding will check the name and the type.
    if (LLVM_UNLIKELY(decoded_attrs.size() < attrs_.size()))
      return mlir::failure();

    return call(decoded_args, decoded_attrs, user_data,
                std::make_index_sequence<kSize>{});
  }

  template <size_t... Is>
  LLVM_ATTRIBUTE_ALWAYS_INLINE mlir::LogicalResult call(
      internal::DecodedArgs args, internal::DecodedAttrs attrs,
      const UserData* user_data, std::index_sequence<Is...>) {
    // A helper structure to allow each decoder find the correct offset in the
    // arguments or attributes.
    internal::DecodingOffsets offsets;

    // Decode all arguments into mlir::FailureOr containers. It is guaranteed
    // that initializer list will be evaluated left-to-right, and we can rely
    // on correct offsets computation.
    std::tuple<mlir::FailureOr<FnArgType<Ts>>...> fn_args = {
        internal::Decode<Ts, checks>::call(offsets, args, attrs_, attrs_idx_,
                                           attrs, user_data)...};

    // Check that all of them were successfully decoded.
    std::array<bool, kSize> decoded = {
        mlir::succeeded(std::get<Is>(fn_args))...};
    if (LLVM_UNLIKELY(llvm::any_of(decoded, [](bool ok) { return !ok; })))
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
// Custom arguments attributes decoding.

namespace internal {

template <typename T, int rank>
int64_t NumElements(StridedMemRefType<T, rank>* memref) {
  int64_t num_elements = 1;
  for (int d = 0; d < rank; ++d) num_elements *= memref->sizes[d];
  return num_elements;
}

template <typename T>
int64_t NumElements(StridedMemRefType<T, 0>* memref) {
  return 0;
}

template <typename T, int rank>
ArrayRef<int64_t> Sizes(StridedMemRefType<T, rank>* memref) {
  return llvm::makeArrayRef(memref->sizes);
}

template <typename T>
ArrayRef<int64_t> Sizes(StridedMemRefType<T, 0>* memref) {
  return {};
}

template <typename T, template <int64_t> class Decoding>
LLVM_ATTRIBUTE_ALWAYS_INLINE mlir::FailureOr<T> DecodeMemref(
    EncodedMemref* encoded) {
  switch (encoded->rank) {
    case 0:
      return Decoding<0>::decode(encoded);
    case 1:
      return Decoding<1>::decode(encoded);
    case 2:
      return Decoding<2>::decode(encoded);
    case 3:
      return Decoding<3>::decode(encoded);
    case 4:
      return Decoding<4>::decode(encoded);
    case 5:
      return Decoding<5>::decode(encoded);
    default:
      assert(false && "unsupported memref rank");
      return mlir::failure();
  }
}
}  // namespace internal

// A view into the memref argument. Corresponds to the MemrefView, however it
// doesn't own the sizes/strides vectors, and cheap to pass around.
struct MemrefView {
  tfrt::DType dtype;
  void* data;
  int64_t offset;
  ArrayRef<int64_t> sizes;
};

// A flat view into the memref argument. If the memref shapes is not required
// for the custom call, it's cheaper to pass the flat view.
struct FlatMemrefView {
  tfrt::DType dtype;
  void* data;
  int64_t size_in_bytes;
};

raw_ostream& operator<<(raw_ostream& os, const MemrefView& view);
raw_ostream& operator<<(raw_ostream& os, const FlatMemrefView& view);

template <CustomCall::RuntimeChecks checks>
struct CustomCallArgDecoding<MemrefView, checks> {
  using EncodedMemref = internal::EncodedMemref;

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  static mlir::FailureOr<MemrefView> Decode(mlir::TypeID type_id, void* value) {
    if (!CustomCall::CheckType<Tagged<MemrefView>>(checks, type_id))
      return mlir::failure();

    auto* encoded = reinterpret_cast<EncodedMemref*>(value);
    return internal::DecodeMemref<MemrefView, Impl>(encoded);
  }

  template <int64_t rank>
  struct Impl {
    LLVM_ATTRIBUTE_ALWAYS_INLINE
    static mlir::FailureOr<MemrefView> decode(EncodedMemref* encoded) {
      using Descriptor = ::StridedMemRefType<float, rank>;
      DType dtype = static_cast<DType>(encoded->dtype);
      auto* descriptor = reinterpret_cast<Descriptor*>(encoded->descriptor);
      return MemrefView{dtype, descriptor->data, descriptor->offset,
                        internal::Sizes(descriptor)};
    }
  };
};

template <CustomCall::RuntimeChecks checks>
struct CustomCallArgDecoding<FlatMemrefView, checks> {
  using EncodedMemref = internal::EncodedMemref;

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  static mlir::FailureOr<FlatMemrefView> Decode(mlir::TypeID type_id,
                                                void* value) {
    if (!CustomCall::CheckType<Tagged<MemrefView>>(checks, type_id))
      return mlir::failure();

    auto* encoded = reinterpret_cast<EncodedMemref*>(value);
    return internal::DecodeMemref<FlatMemrefView, Impl>(encoded);
  }

  template <int64_t rank>
  struct Impl {
    LLVM_ATTRIBUTE_ALWAYS_INLINE
    static mlir::FailureOr<FlatMemrefView> decode(EncodedMemref* encoded) {
      using Descriptor = ::StridedMemRefType<float, rank>;
      DType dtype = static_cast<DType>(encoded->dtype);
      auto* descriptor = reinterpret_cast<Descriptor*>(encoded->descriptor);
      int64_t size = GetHostSize(dtype) * internal::NumElements(descriptor);
      return FlatMemrefView{dtype, descriptor->data, size};
    }
  };
};

#define JITRT_REGISTER_SCALAR_ARG_DECODING(T)                                \
  template <CustomCall::RuntimeChecks checks>                                \
  struct CustomCallArgDecoding<T, checks> {                                  \
    LLVM_ATTRIBUTE_ALWAYS_INLINE static mlir::FailureOr<T> Decode(           \
        mlir::TypeID type_id, void* value) {                                 \
      if (LLVM_UNLIKELY(!CustomCall::CheckType<Tagged<T>>(checks, type_id))) \
        return mlir::failure();                                              \
                                                                             \
      return *reinterpret_cast<T*>(value);                                   \
    }                                                                        \
  }

JITRT_REGISTER_SCALAR_ARG_DECODING(int32_t);
JITRT_REGISTER_SCALAR_ARG_DECODING(int64_t);
JITRT_REGISTER_SCALAR_ARG_DECODING(float);
JITRT_REGISTER_SCALAR_ARG_DECODING(double);

#undef JITRT_REGISTER_SCALAR_ARG_DECODING

// -------------------------------------------------------------------------- //
// Custom call attributes decoding.

template <CustomCall::RuntimeChecks checks>
struct CustomCallAttrDecoding<llvm::StringRef, checks> {
  using StringRef = llvm::StringRef;

  LLVM_ATTRIBUTE_ALWAYS_INLINE static mlir::FailureOr<StringRef> Decode(
      llvm::StringRef name, mlir::TypeID type_id, void* value) {
    if (!CustomCall::CheckType<Tagged<StringRef>>(checks, type_id))
      return mlir::failure();

    auto* encoded = reinterpret_cast<internal::EncodedString*>(value);
    return StringRef(encoded->data, encoded->size);
  }
};

#define JITRT_REGISTER_SCALAR_ATTR_DECODING(T)                     \
  template <CustomCall::RuntimeChecks checks>                      \
  struct CustomCallAttrDecoding<T, checks> {                       \
    LLVM_ATTRIBUTE_ALWAYS_INLINE static mlir::FailureOr<T> Decode( \
        llvm::StringRef name, mlir::TypeID type_id, void* value) { \
      if (!CustomCall::CheckType<Tagged<T>>(checks, type_id))      \
        return mlir::failure();                                    \
                                                                   \
      return *reinterpret_cast<T*>(value);                         \
    }                                                              \
  }

JITRT_REGISTER_SCALAR_ATTR_DECODING(int32_t);
JITRT_REGISTER_SCALAR_ATTR_DECODING(int64_t);
JITRT_REGISTER_SCALAR_ATTR_DECODING(float);
JITRT_REGISTER_SCALAR_ATTR_DECODING(double);

#undef JITRT_REGISTER_SCALAR_ATTR_DECODING

#define JITRT_REGISTER_ARRAY_ATTR_DECODING(T)                                \
  template <CustomCall::RuntimeChecks checks>                                \
  struct CustomCallAttrDecoding<ArrayRef<T>, checks> {                       \
    LLVM_ATTRIBUTE_ALWAYS_INLINE static mlir::FailureOr<ArrayRef<T>> Decode( \
        llvm::StringRef name, mlir::TypeID type_id, void* value) {           \
      if (!CustomCall::CheckType<Tagged<ArrayRef<T>>>(checks, type_id))      \
        return mlir::failure();                                              \
                                                                             \
      auto* encoded = reinterpret_cast<internal::EncodedArray<T>*>(value);   \
      return ArrayRef<T>(&encoded->data, encoded->size);                     \
    }                                                                        \
  }

JITRT_REGISTER_ARRAY_ATTR_DECODING(int32_t);
JITRT_REGISTER_ARRAY_ATTR_DECODING(int64_t);
JITRT_REGISTER_ARRAY_ATTR_DECODING(float);
JITRT_REGISTER_ARRAY_ATTR_DECODING(double);

#undef JITRT_REGISTER_ARRAY_ATTR_DECODING

// Declare/define an explicit specialialization for mlir::TypeID for types used
// by the custom calls. This forces the compiler to emit a strong definition for
// a class and controls which translation unit and shared object will actually
// have it.
//
// See mlir::TypeID for more documentation.
//
// Because custom calls do not "own" the types passed across the function
// boundary, we declare/define specializations for tagged types to avoid
// potential conflicts with other libraries.
#define JITRT_DECLARE_EXPLICIT_TYPE_ID(T) \
  MLIR_DECLARE_EXPLICIT_TYPE_ID(::tfrt::jitrt::Tagged<T>)

#define JITRT_DEFINE_EXPLICIT_TYPE_ID(T) \
  MLIR_DEFINE_EXPLICIT_TYPE_ID(::tfrt::jitrt::Tagged<T>)

}  // namespace jitrt
}  // namespace tfrt

JITRT_DECLARE_EXPLICIT_TYPE_ID(llvm::StringRef);
JITRT_DECLARE_EXPLICIT_TYPE_ID(tfrt::jitrt::FlatMemrefView);
JITRT_DECLARE_EXPLICIT_TYPE_ID(tfrt::jitrt::MemrefView);
JITRT_DECLARE_EXPLICIT_TYPE_ID(int32_t);
JITRT_DECLARE_EXPLICIT_TYPE_ID(int64_t);
JITRT_DECLARE_EXPLICIT_TYPE_ID(float);
JITRT_DECLARE_EXPLICIT_TYPE_ID(double);
JITRT_DECLARE_EXPLICIT_TYPE_ID(ArrayRef<int32_t>);
JITRT_DECLARE_EXPLICIT_TYPE_ID(ArrayRef<int64_t>);
JITRT_DECLARE_EXPLICIT_TYPE_ID(ArrayRef<float>);
JITRT_DECLARE_EXPLICIT_TYPE_ID(ArrayRef<double>);

#endif  // TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_CUSTOM_CALL_H_
