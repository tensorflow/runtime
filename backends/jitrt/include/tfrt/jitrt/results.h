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

#ifndef TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_RESULTS_H_
#define TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_RESULTS_H_

#include <functional>
#include <type_traits>
#include <utility>

#include "llvm/Support/Compiler.h"
#include "mlir/Support/LogicalResult.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/msan.h"
#include "third_party/tensorflow/compiler/xla/mlir/runtime/utils/async_runtime_api.h"
#include "third_party/tensorflow/compiler/xla/runtime/results.h"
#include "third_party/tensorflow/compiler/xla/runtime/types.h"

namespace tfrt {
namespace jitrt {

//===----------------------------------------------------------------------===//
// Result converters for integrating with TFRT/BEF kernels (RemainingResults).
//===----------------------------------------------------------------------===//

// Result converter that returns converted values through the RemainingResults,
// and allows adding user-provided conversion functions dynamically.
template <typename ConversionContext>
class RemainingResultsConverter : public xla::runtime::ResultConverter {
  static_assert(!std::is_void<ConversionContext>::value,
                "Conversion context can't be void");

 public:
  // A user provided function to augment all emitted errors, e.g. it can be used
  // to attach diagnostics collected at runtime to the error message.
  using AugmentError = std::function<Error(const Error&)>;

  // It is the caller's responsibility to guarantee that conversion context
  // will outlive all pending conversions (in case of returning async values).
  RemainingResultsConverter(RemainingResults results,
                            ConversionContext& context,
                            AugmentError augment_error = {})
      : results_(results), context_(context), augment_error_(augment_error) {
    AddConversion(UnsupportedReturnType);
  }

  ~RemainingResultsConverter() override = default;

  mlir::LogicalResult ReturnValue(unsigned result_index,
                                  const xla::runtime::Type* type,
                                  const xla::runtime::Type* runtime_type,
                                  void* ret) const final {
    for (auto& convert : llvm::reverse(conversion_callbacks_)) {
      auto converted =
          convert(context_, results_, result_index, type, runtime_type, ret);
      if (mlir::succeeded(converted)) return mlir::success();
    }
    return mlir::failure();
  }

  void ReturnError(const absl::Status& error) const final {
    assert(!error.ok());
    if (results_.empty()) return;
    results_[0] = MakeErrorAsyncValueRef(
        augment_error_ ? absl::InternalError(toString(
                             augment_error_(MakeStringError(error.message()))))
                       : error);
    for (size_t i = 1; i < results_.size(); ++i) results_[i] = results_[0];
  }

  // Adds a conversion function to this converter. Conversion callback must be
  // convertible to the `ConversionCallbackFn` function type:
  //
  //   mlir::LogicalResult(ConversionContext&, RemainingResults, unsigned,
  //                       const Type* type, const Type* runtime_type, void*)
  //
  // Conversion function must return `success` if it successfully handled the
  // return type and set the result async value. If conversion function returns
  // `failure` converter will try the next conversion function.
  //
  // When attempting to convert a retuned value via 'ReturnValue', the most
  // recently added conversions will be invoked first.
  template <typename FnT>
  void AddConversion(FnT&& callback) {
    conversion_callbacks_.emplace_back(std::forward<FnT>(callback));
  }

  RemainingResultsConverter(RemainingResultsConverter&&) = default;
  RemainingResultsConverter& operator=(RemainingResultsConverter&&) = default;

 private:
  using ConversionCallbackFn = llvm::function_ref<mlir::LogicalResult(
      ConversionContext&, RemainingResults, unsigned, const xla::runtime::Type*,
      const xla::runtime::Type*, void*)>;

  // If result type was not matched by any of the user defined conversion
  // functions we return an error to the caller.
  static mlir::LogicalResult UnsupportedReturnType(
      ConversionContext& ctx, RemainingResults results, unsigned result_index,
      const xla::runtime::Type* t, const xla::runtime::Type* rt, const void*) {
    results.MakeErrorAt(result_index, StrCat("unsupported return type: ", *rt,
                                             " (derived from: ", *t, ")"));
    return mlir::failure();
  }

  RemainingResults results_;
  ConversionContext& context_;  // must outlive all pending result conversions
  AugmentError augment_error_;
  llvm::SmallVector<ConversionCallbackFn, 4> conversion_callbacks_;
};

// A template that converts function pointer passed as non-type template
// parameter into the struct compatible with the `StaticReturnValueConverter`.
template <typename ConversionContext,
          mlir::LogicalResult (*convert)(ConversionContext&, RemainingResults,
                                         unsigned, const xla::runtime::Type*,
                                         const xla::runtime::Type*, void*)>
struct ReturnValueConversion {
  mlir::LogicalResult operator()(ConversionContext& ctx,
                                 RemainingResults results,
                                 unsigned result_index,
                                 const xla::runtime::Type* t,
                                 const xla::runtime::Type* rt,
                                 void* ret) const {
    return convert(ctx, results, result_index, t, rt, ret);
  }
};

// Remaining results converter with statically registered conversion functions.
//
// Conversion function type must define `operator()` with a signature:
//
//   mlir::LogicalResult operator()(ConversionContext&, RemainingResults,
//                                  unsigned result_index, const Type* type,
//                                  const Type* runtime_type, void*) const;
//
// Conversion function must return `success` if it successfully handled the
// return type and set the result async value. If conversion function returns
// `failure` converter will try the next conversion function.
template <typename ConversionContext, typename... ConversionFns>
class StaticRemainingResultsConverter : public xla::runtime::ResultConverter {
 public:
  StaticRemainingResultsConverter(RemainingResults results,
                                  ConversionContext& context)
      : results_(results), context_(context) {}

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  mlir::LogicalResult ReturnValue(unsigned result_index,
                                  const xla::runtime::Type* type,
                                  const xla::runtime::Type* runtime_type,
                                  void* ret) const final {
    return convert_(context_, results_, result_index, type, runtime_type, ret);
  }

  void ReturnError(const absl::Status& error) const final {
    assert(!error.ok());
    if (results_.empty()) return;
    results_[0] = MakeErrorAsyncValueRef(error);
    for (size_t i = 1; i < results_.size(); ++i) results_[i] = results_[0];
  }

 private:
  template <typename... Fns>
  struct Impl;

  template <typename Fn, typename... Fns>
  struct Impl<Fn, Fns...> {
    LLVM_ATTRIBUTE_ALWAYS_INLINE
    mlir::LogicalResult operator()(ConversionContext& ctx,
                                   RemainingResults results,
                                   unsigned result_index,
                                   const xla::runtime::Type* t,
                                   const xla::runtime::Type* rt,
                                   void* ret) const {
      auto converted = convert(ctx, results, result_index, t, rt, ret);
      if (LLVM_LIKELY(mlir::succeeded(converted))) return mlir::success();
      return try_next(ctx, results, result_index, t, rt, ret);
    }

    Fn convert;
    Impl<Fns...> try_next;
  };

  template <typename Fn>
  struct Impl<Fn> {
    LLVM_ATTRIBUTE_ALWAYS_INLINE
    mlir::LogicalResult operator()(ConversionContext& ctx,
                                   RemainingResults results,
                                   unsigned result_index,
                                   const xla::runtime::Type* t,
                                   const xla::runtime::Type* rt,
                                   void* ret) const {
      auto converted = convert(ctx, results, result_index, t, rt, ret);
      if (LLVM_LIKELY(mlir::succeeded(converted))) return mlir::success();
      results.MakeErrorAt(result_index, StrCat("unsupported return type: ", *rt,
                                               " (derived from: ", *t, ")"));
      return mlir::failure();
    }

    Fn convert;
  };

  RemainingResults results_;
  ConversionContext& context_;  // must outlive all pending result conversions
  Impl<ConversionFns...> convert_;
};

//===----------------------------------------------------------------------===//
// Default conversion functions that do not require conversion context.
//===----------------------------------------------------------------------===//

namespace internal {

// Converts returned values of `async::TokenType` type to the async chains.
mlir::LogicalResult ReturnAsyncToken(RemainingResults results,
                                     unsigned result_index,
                                     const xla::runtime::Type* type,
                                     const xla::runtime::Type* runtime_type,
                                     void* result_ptr);

// Following functions always construct a new tensor for the returned memref.
// This is not correct in general, because returned memref can be one of the
// original operands or global constant memref. These function must be used only
// when it is guaranteed that the compiled region will always allocate new
// memrefs for the results.

// Converts returned values of `async<memref<...>>` type to the async values
// of newly constructed DenseHostTensors.
mlir::LogicalResult ReturnAsyncMemrefAsDenseHostTensor(
    RemainingResults results, unsigned result_index,
    const xla::runtime::Type* type, const xla::runtime::Type* runtime_type,
    void* result_ptr);

// Converts returned values of `memref<...>` type to the async values of newly
// constructed DenseHostTensors.
mlir::LogicalResult ReturnMemrefAsDenseHostTensor(
    RemainingResults results, unsigned result_index,
    const xla::runtime::Type* type, const xla::runtime::Type* runtime_type,
    void* result_ptr);

}  // namespace internal

#define DECLARE_CONTEXT_ADAPTOR(NAME)                                         \
  template <typename ConversionContext>                                       \
  static mlir::LogicalResult NAME(                                            \
      ConversionContext&, RemainingResults results, unsigned result_index,    \
      const xla::runtime::Type* type, const xla::runtime::Type* runtime_type, \
      void* result_ptr) {                                                     \
    return internal::NAME(results, result_index, type, runtime_type,          \
                          result_ptr);                                        \
  }

DECLARE_CONTEXT_ADAPTOR(ReturnAsyncToken)
DECLARE_CONTEXT_ADAPTOR(ReturnAsyncMemrefAsDenseHostTensor)
DECLARE_CONTEXT_ADAPTOR(ReturnMemrefAsDenseHostTensor)

#undef DECLARE_CONTEXT_ADAPTOR

// -------------------------------------------------------------------------- //

// Converts returned memref values to Tensors using a user-provided Converter
// that must implement this concept:
//
// struct ConvertMemrefToTensor {
//   using ResultType        = MyTensorType;           // must be movable
//   using ConversionContext = ConversionContextType;  // must be movable
//
//   template <typename T, int rank>
//   static MyTensorType Convert(ConversionContext&, void* memref_ptr) {
//     auto* memref = static_cast<StridedMemRefType<T, rank>*>(memref_ptr);
//     return MyTensorType>(memref.basePtr, memref.data, ...);
//   }
// };
//
template <typename Converter,
          typename ResultType = typename Converter::ResultType,
          typename ConversionContext = typename Converter::ConversionContext>
mlir::LogicalResult ReturnStridedMemref(ConversionContext& ctx,
                                        RemainingResults results,
                                        unsigned result_index,
                                        const xla::runtime::Type* type,
                                        const xla::runtime::Type* runtime_type,
                                        void* result_ptr) {
  static_assert(std::is_move_constructible<ResultType>::value,
                "Conversion result type must be move constructible");
  static_assert(std::is_move_constructible<ConversionContext>::value,
                "Conversion context type must be move constructible");

  // Check if the runtime type is a valid memref.
  auto* memref = dyn_cast<xla::runtime::MemrefType>(runtime_type);
  if (!memref) return mlir::failure();

  // Dispatch to the correct extract function based on rank.
  auto rank_dispatch = [&](auto type_tag) {
    using T = decltype(type_tag);
    int64_t rank = memref->rank();

    auto convert_and_emplace = [&](auto rank_tag) {
      constexpr int rank = decltype(rank_tag)::value;
      results.EmplaceAt<ResultType>(
          result_index, Converter::template Convert<T, rank>(ctx, result_ptr));
    };

    switch (rank) {
      case 0:
        convert_and_emplace(std::integral_constant<int, 0>{});
        break;
      case 1:
        convert_and_emplace(std::integral_constant<int, 1>{});
        break;
      case 2:
        convert_and_emplace(std::integral_constant<int, 2>{});
        break;
      case 3:
        convert_and_emplace(std::integral_constant<int, 3>{});
        break;
      case 4:
        convert_and_emplace(std::integral_constant<int, 4>{});
        break;
      case 5:
        convert_and_emplace(std::integral_constant<int, 5>{});
        break;
      default:
        // TODO(ezhulenev): To simplify conversion from a void* pointer to
        // memref descriptor we rely on the StridedMemrefType<T, rank> and
        // dispatch only up to a fixed rank.
        results.MakeErrorAt(result_index,
                            StrCat("unsupported returned memref rank: ", rank));
    }
  };

  // Dispatch based on the element type.
  xla::PrimitiveType element_type = memref->element_type();

  // If the runtime memref type was derived from the Tensor type, take the
  // element type of the original tensor, because during lowering from the high
  // level dialects we can change the data type to another data type with
  // compatible memory layout (e.g. unsigned type converted to signless type).
  if (auto* tensor = dyn_cast<xla::runtime::RankedTensorType>(type))
    element_type = tensor->element_type();

  switch (element_type) {
    case xla::PrimitiveType::F32:
      rank_dispatch(float{});
      break;
    case xla::PrimitiveType::F64:
      rank_dispatch(double{});
      break;
    case xla::PrimitiveType::U8:
      rank_dispatch(uint8_t{});
      break;
    case xla::PrimitiveType::U16:
      rank_dispatch(uint16_t{});
      break;
    case xla::PrimitiveType::U32:
      rank_dispatch(uint32_t{});
      break;
    case xla::PrimitiveType::U64:
      rank_dispatch(uint64_t{});
      break;
    case xla::PrimitiveType::PRED:
      rank_dispatch(bool{});
      break;
    case xla::PrimitiveType::S8:
      rank_dispatch(int8_t{});
      break;
    case xla::PrimitiveType::S16:
      rank_dispatch(int16_t{});
      break;
    case xla::PrimitiveType::S32:
      rank_dispatch(int32_t{});
      break;
    case xla::PrimitiveType::S64:
      rank_dispatch(int64_t{});
      break;
    case xla::PrimitiveType::C64:
      rank_dispatch(std::complex<float>{});
      break;
    case xla::PrimitiveType::C128:
      rank_dispatch(std::complex<double>{});
      break;
    default:
      results.MakeErrorAt(
          result_index,
          StrCat("unsupported returned memref element type: ", element_type));
  }

  return mlir::success();
}

namespace internal {

// Adaptor that creates a function compatible with `ExtractAsyncValue` from
// the `Converter` concept compatible with `ReturnStridedMemref`.
template <typename Converter, typename T, int rank>
void Emplace(void* memref_ptr, AsyncValue* dst, void* context) {
  using ResultType = typename Converter::ResultType;
  using ConversionContext = typename Converter::ConversionContext;

  dst->emplace<ResultType>(Converter::template Convert<T, rank>(
      *reinterpret_cast<ConversionContext*>(context), memref_ptr));
}

}  // namespace internal

// Converts returned async memref values to Tensors using user provided
// Converter that must compatible with `ReturnStridedMemref` define above.
template <typename Converter,
          typename ResultType = typename Converter::ResultType,
          typename ConversionContext = typename Converter::ConversionContext>
mlir::LogicalResult ReturnAsyncStridedMemref(
    ConversionContext& ctx, RemainingResults results, unsigned result_index,
    const xla::runtime::Type* type, const xla::runtime::Type* runtime_type,
    void* result_ptr) {
  static_assert(std::is_move_constructible<ResultType>::value,
                "Conversion result type must be move constructible");
  static_assert(std::is_move_constructible<ConversionContext>::value,
                "Conversion context type must be move constructible");

  auto* value_type = dyn_cast<xla::runtime::AsyncValueType>(type);
  if (!value_type) return mlir::failure();

  // Load the pointer to the async value from a pointer to result storage.
  TFRT_MSAN_MEMORY_IS_INITIALIZED(result_ptr, sizeof(void*));
  void* ret = *reinterpret_cast<void**>(result_ptr);
  auto* value = static_cast<mlir::runtime::AsyncValue*>(ret);

  // We already verified that return value is an async value of memref.
  auto* memref = dyn_cast<xla::runtime::MemrefType>(&value_type->value_type());
  assert(memref && "we only support async values of memrefs");

  // Allocate constructed async value to be returned to the caller.
  auto dst = [&]() -> AsyncValue* {
    return results.AllocateAt<ResultType>(result_index).get();
  };

  // Dispatch to the correct extract function based on rank.
  auto rank_dispatch = [&](auto type_tag) {
    using T = decltype(type_tag);
    int64_t rank = memref->rank();

    // Pass an opaque pointer to the operands context to the emplace function.
    void* ptr = const_cast<void*>(reinterpret_cast<const void*>(&ctx));

    switch (rank) {
      case 0:
        xla::runtime::ExtractAsyncValue(value, dst(), ptr,
                                        internal::Emplace<Converter, T, 0>);
        break;
      case 1:
        xla::runtime::ExtractAsyncValue(value, dst(), ptr,
                                        internal::Emplace<Converter, T, 1>);
        break;
      case 2:
        xla::runtime::ExtractAsyncValue(value, dst(), ptr,
                                        internal::Emplace<Converter, T, 2>);
        break;
      case 3:
        xla::runtime::ExtractAsyncValue(value, dst(), ptr,
                                        internal::Emplace<Converter, T, 3>);
        break;
      case 4:
        xla::runtime::ExtractAsyncValue(value, dst(), ptr,
                                        internal::Emplace<Converter, T, 4>);
        break;
      case 5:
        xla::runtime::ExtractAsyncValue(value, dst(), ptr,
                                        internal::Emplace<Converter, T, 5>);
        break;
      default:
        // TODO(ezhulenev): Because ExtractAsyncValue takes a llvm::function_ref
        // we can't pass a runtime argument to emplace functions via lambda
        // capture, because the value might become available asynchronously and
        // this will lead to use after free. Consider adding an std::function
        // alternative for ranks higher then 5? Lambdas with small captures
        // should be stack allocated anyway, however it is implementation
        // defined.
        //
        // TODO(ezhulenev): Another alternative is to pass the desired result
        // type after conversion via the conversion context. Emplace function
        // can query all the information it needs from the conversion context,
        // e.g. expected result type rank and data type.
        results.MakeErrorAt(result_index,
                            StrCat("unsupported returned memref rank: ", rank));
    }
  };

  // Dispatch based on the memref element type.
  xla::PrimitiveType element_type = memref->element_type();

  switch (element_type) {
    case xla::PrimitiveType::F32:
      rank_dispatch(float{});
      break;
    case xla::PrimitiveType::F64:
      rank_dispatch(double{});
      break;
    case xla::PrimitiveType::PRED:
      rank_dispatch(bool{});
      break;
    case xla::PrimitiveType::S8:
      rank_dispatch(int8_t{});
      break;
    case xla::PrimitiveType::S16:
      rank_dispatch(int16_t{});
      break;
    case xla::PrimitiveType::S32:
      rank_dispatch(int32_t{});
      break;
    case xla::PrimitiveType::S64:
      rank_dispatch(int64_t{});
      break;
    case xla::PrimitiveType::U8:
      rank_dispatch(uint8_t{});
      break;
    case xla::PrimitiveType::U16:
      rank_dispatch(uint16_t{});
      break;
    case xla::PrimitiveType::U32:
      rank_dispatch(uint32_t{});
      break;
    case xla::PrimitiveType::U64:
      rank_dispatch(uint64_t{});
      break;
    case xla::PrimitiveType::C64:
      rank_dispatch(std::complex<float>{});
      break;
    case xla::PrimitiveType::C128:
      rank_dispatch(std::complex<double>{});
      break;
    default:
      results.MakeErrorAt(
          result_index,
          StrCat("unsupported returned memref element type: ", element_type));
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Helper functions for forwarding errors to remaining results.
//===----------------------------------------------------------------------===//

// Constructs error async value from the `error` and returns it for all results.
void ReturnErrors(RemainingResults results, Error error);
void ReturnErrors(RemainingResults results, DecodedDiagnostic error);

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_RESULTS_H_
