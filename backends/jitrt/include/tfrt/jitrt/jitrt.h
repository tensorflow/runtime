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

// Support library for implementing TFRT kernels that do JIT compilation using
// MLIR framework (generating kernels at runtime from hight level MLIR
// dialects, e.g. generating dense linear algebra kernels from Linalg dialect).

#ifndef TFRT_BACKENDS_JITRT_JITRT_H_
#define TFRT_BACKENDS_JITRT_JITRT_H_

#include <sys/types.h>

#if __cplusplus >= 201703L
#include <any>
#endif

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/jitrt/async_runtime.h"
#include "tfrt/jitrt/async_runtime_api.h"
#include "tfrt/jitrt/async_values_cache.h"
#include "tfrt/jitrt/constraints.h"
#include "tfrt/jitrt/memory_mapper.h"
#include "tfrt/jitrt/specialization.h"
#include "tfrt/jitrt/symbolic_shape.h"
#include "tfrt/jitrt/types.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/msan.h"

// Forward declare Eigen types.
namespace Eigen {
class ThreadPoolInterface;
}  // namespace Eigen

namespace tfrt {

class Tensor;

namespace jitrt {

// Compiled module example:
//
//   module @kernel attributes { tfrt.compiled } {
//     func @main(
//       %input0: memref<*xf32>   { jitrt.constraint = "rank"  },
//       %input1: memref<?x?xf32> { jitrt.constraint = "shape" },
//       %perm: memref<4xi32>     { jitrt.constraint = "value" }
//     ) -> !async.value<memref<?x?xf32>> {
//       ...
//       return %result : !async.value<memref<?x?xf32>>
//     }
//   }
//
// Compiled function can define constraints on its inputs, that must be
// resolved before the function can be compiled. If constraints can't be
// resolved statically from the function signature (e.g. rank is unknown), then
// the runtime will specialize generic function to concrete operands at runtime
// (concrete operands rank, shape or value).
//
// If function inputs do not have unresolved constraints, compiler will
// instantiate the default executable, that can take all compatible inputs
// without recompilation.
//
//
// (a) Rank constraint:
//
//     %arg : tensor<*xf32> { jitrt.constraint = "rank" }
//
//     Before compiling the function, unranked input type will be updated to the
//     corresponding ranked input type (e.g. unranked tensor -> ranked tensor).
//
// (b) Shape constraint:
//
//     %arg : tensor<?x?xf32> { jitrt.constraint = "shape" }
//
//     Shape of the runtime argument will be used to specialize the compiled
//     function, if this shape seen the first time, it will trigger function
//     recompilation.
//
// (c) Value constraint:
//
//     %reduction_dimension : tensor<i32> { jitrt.constraint = "value" }
//
//     Runtime value will be sunk into the body of a function as a constant,
//     and the function will be recompiled. For example this can be used to sink
//     reduction dimensions to generate more efficient code.
//
//     Value constraint is only supported for the integer data type, in practice
//     it should be reduction dimension, dimension permutation, or any similar
//     value that does not change often, and is required for generating
//     efficient code.
//
//  Shape and value specialization example:
//
//    // Computes `%arg0` mean value over the axis specified by the `%arg1`.
//    // See: https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
//    func @mean(%arg0: tensor<?x?xf32>, %arg1: tensor<i32>) -> tensor<?xf32> {
//      %0 = "tf.Mean(%arg0, %arg1)
//             : (tensor<?x?xf32>, tensor<i32>) -> tensor<?xf32>
//      return %0: tensor<?xf32>
//    }
//
//  Shape specialization to input shapes: [tensor<4x8xf32>, tensor<f32>]
//
//    func @mean(%arg0: tensor<4x8xf32>, %arg1: tensor<i32>) -> tensor<?xf32> {
//      %0 = "tf.Mean(%arg0, %arg1)
//             : (tensor<4x8xf32>, tensor<i32>) -> tensor<?xf32>
//      return %0: tensor<?xf32>
//    }
//
//    Shape specialization in this particular case doesn't bring much
//    improvement, because without knowing the reduction axis we can't infer
//    any new information from the input shape alone.
//
//  Value specialization to input values: [ <do-not-specialize>, dense<1 : i32>]
//
//    func @mean(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
//      %0 = "tf.Constant" { value = dense<1 : i32>} -> tensor<i32>
//      %1 = "tf.Mean(%arg0, %0)
//             : (tensor<4x8xf32>, tensor<i32>) -> tensor<4xf32>
//      return %1 : tensor<4xf32>
//    }
//
//    By specializing function to the concrete value of the second argument, by
//    sinking it into the function body we can infer the output shape. Also this
//    information allows to statically choose reduction implementation optimized
//    for reducing along the inner most dimension.
//
//    Furthermore static information about reduction axis allows to lower mean
//    operation to Linalg generic operation. Dynamic reduction axis is not
//    representable in Linalg, and would require multi-versioning and dynamic
//    dispatch at runtime.

// Forward declare the JitExecutable class that itself is not an executable, but
// owns one (or many) executables compiled for different shapes or values of the
// arguments. It is responsible for lazy compilation of executables for the
// concrete shapes or values if needed.
class JitExecutable;

// Forward declare the Executable class that represents a fully compiled module,
// which in practice means that it has a function pointer to the compiled
// function, and knows how to execute it, and return results to the caller.
class Executable;

struct CompilationOptions {
  // Calling convention defines an ABI for JitRt to call a compiled kernel. See
  // documentation and example below.
  using CallingConvention =
      std::function<mlir::FunctionType(mlir::FunctionType)>;

  // Returns a calling convention that only adds the kernel context argument.
  static CallingConvention DefaultCallingConvention();

  // Returns a calling convention that uses user-provided type converter to
  // convert all inputs and results types, and adds the kernel context argument.
  static CallingConvention DefaultCallingConvention(mlir::TypeConverter);

  // Compiled kernel can be specialized and recompiled at runtime to the
  // concrete input shapes and sometimes values (e.g. reduciton dimension).
  enum class Specialization {
    // Recompile specialized kernels when needed.
    kEnabled,
    // Completely disable specialized kernels (always call default executable).
    kDisabled,
    // Always use specialized kernels, and never call default executable (only
    // required for getting reproducible results in benchmarks).
    kAlways,
  };

  // LLVM optimization level when JIT compiling a kernel.
  Optional<llvm::CodeGenOpt::Level> jit_code_opt_level;

  // What level of specialization is enabled at runtime.
  Specialization specialization = Specialization::kAlways;

  // Register dialects that are allowed in the serialized module.
  std::function<void(mlir::DialectRegistry&)> register_dialects;

  // Create a pass pipeline that is called whenever the compiled module
  // gets specialized. This pipeline can use refined shape information and
  // symbolic shape attributes to do the shape inference and canonicalization.
  //
  // Original input module might have an undefined calling convention (e.g.
  // JitRt does not support unranked tensors), and specialization can be
  // required as a precondition for compilation.
  std::function<void(mlir::PassManager&)> create_specialization_pipeline;

  // Create a pass pipeline that lowers compiled module from high level
  // dialects to the LLVM dialect. JitRt will use the LLVM ORC compiler API
  // to compile the LLVM module at run time (https://llvm.org/docs/ORCv2.html).
  //
  // This compilation pipeline must create the entrypoint function with an ABI
  // compatible with the calling convention advertised to the JitRt through the
  // `calling_convention` type conversion, and for that it usually must include
  // `rt-to-kernel-function` pass to convert regular functions to "kernels".
  std::function<void(mlir::PassManager&)> create_compilation_pipeline;

  // Calling convention converts the compiled module entrypoint function type to
  // the function type with a well defined ABI (e.g. tensors do not have an ABI,
  // and must be passed across the function boundary as memrefs). In a nutshell
  // it tells the JitRt how to call the compiled kernel at run time, and how to
  // return results back to the JitRt.
  //
  // If conversion is not possible, calling convention must return a null value.
  //
  // Example: abstract kernel defined in high level dialect, e.g. MHLO
  //
  //   ```mlir
  //     func @kernel(%arg0: tensor<?xf32>,
  //                  %arg1: tensor<?xf32>) -> tensor<?x?xf32> { ... }
  //   ```
  //
  //   after calling convention conversion becomes:
  //
  //   ```mlir
  //     func @kernel(%ctx: !rt.kernel_context,
  //                  %arg0: memref<?xf32>,
  //                  %arg1: memref<?xf32>) -> memref<?x?xf32> { ... }
  //   ```
  //
  // Calling convention function type is not the same as the entrypoint function
  // type produced by the compilation pipeline for several reasons:
  //
  // 1) Compilation pipeline produces LLVM functions with LLVM types, and high
  //    level information is lost, e.g. all memrefs are deconstructed into
  //    primitive fields when passed as inputs.
  //
  // 2) Compiled kernel function always returns void, and uses runtime API to
  //    return results back to the caller (see `rt-to-kernel-function` pass).
  //
  // Calling convention function type is a JitRt-compatible description of the
  // compiled kernel ABI, so that JitRt can correctly initialize CallFrame
  // arguments, allocate memory for returned results, and then correctly decode
  // results memory into the high level types (e.g. convert returned memref
  // descriptor to a Tensorfow tensor).
  CallingConvention calling_convention = DefaultCallingConvention();
};

//----------------------------------------------------------------------------//
// Conversions from compiled kernel operands to JitRt runtime types.
//----------------------------------------------------------------------------//

// Converts tfrt Tensor to the Memref descriptor if concrete Tensor type is
// supported (currently only DenseHostTensor can be converted). Returns error
// otherwise.
Expected<MemrefDesc> ConvertTensorToMemrefDesc(const Tensor& tensor);

//----------------------------------------------------------------------------//
// Conversions from compiled kernel results to the TFRT AsyncValues.
//----------------------------------------------------------------------------//

// Return value converter is responsible for taking the value returned from the
// compiled function, and converting it to the AsyncValue. Implementation (see
// below) is relying on user defined set of conversion functions.
class ReturnValueConverterBase {
 public:
  explicit ReturnValueConverterBase(RemainingResults results)
      : results_(results) {}
  virtual ~ReturnValueConverterBase() = default;

  // Converts value `ret` of type `runtime_type` (runtime type derived from the
  // original `type`) returned from the compiled function at `result_index`
  // return position using registered conversion functions, and emplaces the
  // result async value. If the conversion failed returns a failure and sets the
  // result async value to error.
  virtual mlir::LogicalResult ReturnValue(unsigned result_index,
                                          const Type* type,
                                          const Type* runtime_type,
                                          void* ret) const = 0;

  // Returns error for all remaining results (copy of the `error` argument).
  virtual void ReturnErrors(RCReference<ErrorAsyncValue> error) const;

 protected:
  RemainingResults results() const { return results_; }

 private:
  RemainingResults results_;
};

// Return value converter class allows to register custom functions for
// converting compiled kernel execution results to returned async values.
template <typename ConversionContext>
class ReturnValueConverter : public ReturnValueConverterBase {
  static_assert(!std::is_void<ConversionContext>::value,
                "Conversion context can't be void");

 public:
  // It is the caller's responsibility to guarantee that conversion context
  // will outlive all pending conversions (in case of returning async values).
  ReturnValueConverter(RemainingResults results, ConversionContext& context)
      : ReturnValueConverterBase(results), context_(context) {
    AddConversion(UnsupportedReturnType);
  }

  ~ReturnValueConverter() override = default;

  mlir::LogicalResult ReturnValue(unsigned result_index, const Type* type,
                                  const Type* runtime_type,
                                  void* ret) const final {
    for (auto& convert : llvm::reverse(conversion_callbacks_)) {
      auto converted =
          convert(context_, results(), result_index, type, runtime_type, ret);
      if (mlir::succeeded(converted)) return mlir::success();
    }
    return mlir::failure();
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

  ReturnValueConverter(ReturnValueConverter&&) = default;
  ReturnValueConverter& operator=(ReturnValueConverter&&) = default;

 private:
  using ConversionCallbackFn = llvm::function_ref<mlir::LogicalResult(
      ConversionContext&, RemainingResults, unsigned, const Type*, const Type*,
      void*)>;

  // If result type was not matched by any of the user defined conversion
  // functions we return an error to the caller.
  static mlir::LogicalResult UnsupportedReturnType(
      ConversionContext& ctx, RemainingResults results, unsigned result_index,
      const Type* t, const Type* rt, const void*) {
    results.MakeErrorAt(result_index, StrCat("unsupported return type: ", *rt,
                                             " (derived from: ", *t, ")"));
    return mlir::failure();
  }

  ConversionContext& context_;  // must outlive all pending result conversions
  llvm::SmallVector<ConversionCallbackFn, 4> conversion_callbacks_;
};

// A template that converts function pointer passed as non-type template
// parameter into the struct compatible with the `StaticReturnValueConverter`.
template <typename ConversionContext,
          mlir::LogicalResult (*convert)(ConversionContext&, RemainingResults,
                                         unsigned, const Type*, const Type*,
                                         void*)>
struct ReturnValueConversion {
  mlir::LogicalResult operator()(ConversionContext& ctx,
                                 RemainingResults results,
                                 unsigned result_index, const Type* t,
                                 const Type* rt, void* ret) const {
    return convert(ctx, results, result_index, t, rt, ret);
  }
};

// Return value converter class with statically registered conversion functions.
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
//
// TODO(ezhulenev): Currently we only support default constructed conversion
// functions, however we can potentially pass conversion functions with
// non-empty captures.
template <typename ConversionContext, typename... ConversionFns>
class StaticReturnValueConverter : public ReturnValueConverterBase {
 public:
  StaticReturnValueConverter(RemainingResults results,
                             ConversionContext& context)
      : ReturnValueConverterBase(results), context_(context) {}

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  mlir::LogicalResult ReturnValue(unsigned result_index, const Type* type,
                                  const Type* runtime_type,
                                  void* ret) const final {
    return convert_(context_, results(), result_index, type, runtime_type, ret);
  }

 private:
  template <typename... Fns>
  struct Impl;

  template <typename Fn, typename... Fns>
  struct Impl<Fn, Fns...> {
    LLVM_ATTRIBUTE_ALWAYS_INLINE
    mlir::LogicalResult operator()(ConversionContext& ctx,
                                   RemainingResults results,
                                   unsigned result_index, const Type* t,
                                   const Type* rt, void* ret) const {
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
                                   unsigned result_index, const Type* t,
                                   const Type* rt, void* ret) const {
      auto converted = convert(ctx, results, result_index, t, rt, ret);
      if (LLVM_LIKELY(mlir::succeeded(converted))) return mlir::success();
      results.MakeErrorAt(result_index, StrCat("unsupported return type: ", *rt,
                                               " (derived from: ", *t, ")"));
      return mlir::failure();
    }

    Fn convert;
  };

  ConversionContext& context_;  // must outlive all pending result conversions
  Impl<ConversionFns...> convert_;
};

// -------------------------------------------------------------------------- //
// Default conversion functions that do not require conversion context.
// -------------------------------------------------------------------------- //

namespace internal {

// Converts returned values of `async::TokenType` type to the async chains.
mlir::LogicalResult ReturnAsyncToken(RemainingResults results,
                                     unsigned result_index, const Type* type,
                                     const Type* runtime_type,
                                     void* result_ptr);

// Following functions always construct a new tensor for the returned memref.
// This is not correct in general, because returned memref can be one of the
// original operands or global constant memref. These function must be used only
// when it is guaranteed that the compiled region will always allocate new
// memrefs for the results.

// Converts returned values of `async<memref<...>>` type to the async values
// of newly constructed DenseHostTensors.
mlir::LogicalResult ReturnAsyncMemrefAsDenseHostTensor(RemainingResults results,
                                                       unsigned result_index,
                                                       const Type* type,
                                                       const Type* runtime_type,
                                                       void* result_ptr);

// Converts returned values of `memref<...>` type to the async values of newly
// constructed DenseHostTensors.
mlir::LogicalResult ReturnMemrefAsDenseHostTensor(RemainingResults results,
                                                  unsigned result_index,
                                                  const Type* type,
                                                  const Type* runtime_type,
                                                  void* result_ptr);

}  // namespace internal

#define DECLARE_CONTEXT_ADAPTOR(NAME)                                      \
  template <typename ConversionContext>                                    \
  static mlir::LogicalResult NAME(                                         \
      ConversionContext&, RemainingResults results, unsigned result_index, \
      const Type* type, const Type* runtime_type, void* result_ptr) {      \
    return internal::NAME(results, result_index, type, runtime_type,       \
                          result_ptr);                                     \
  }

DECLARE_CONTEXT_ADAPTOR(ReturnAsyncToken)
DECLARE_CONTEXT_ADAPTOR(ReturnAsyncMemrefAsDenseHostTensor)
DECLARE_CONTEXT_ADAPTOR(ReturnMemrefAsDenseHostTensor)

#undef DECLARE_CONTEXT_ADAPTOR

// -------------------------------------------------------------------------- //

// Converts returned memref values to Tensors using user provided Converter
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
                                        unsigned result_index, const Type* type,
                                        const Type* runtime_type,
                                        void* result_ptr) {
  static_assert(std::is_move_constructible<ResultType>::value,
                "Conversion result type must be move constructible");
  static_assert(std::is_move_constructible<ConversionContext>::value,
                "Conversion context type must be move constructible");

  // Check if the runtime type is a valid memref.
  auto* memref = dyn_cast<MemrefType>(runtime_type);
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
  DType element_type = memref->element_type();

  // If the runtime memref type was derived from the Tensor type, take the
  // element type of the original tensor, because during lowering from the high
  // level dialects we can change the data type to another data type with
  // compatible memory layout (e.g. unsigned type converted to signless type).
  if (auto* tensor = dyn_cast<RankedTensorType>(type))
    element_type = tensor->element_type();

  switch (element_type) {
    case DType::F32:
      rank_dispatch(float{});
      break;
    case DType::UI8:
      rank_dispatch(uint8_t{});
      break;
    case DType::UI32:
      rank_dispatch(uint32_t{});
      break;
    case DType::UI64:
      rank_dispatch(uint64_t{});
      break;
    case DType::I1:
      rank_dispatch(bool{});
      break;
    case DType::I8:
      rank_dispatch(int8_t{});
      break;
    case DType::I32:
      rank_dispatch(int32_t{});
      break;
    case DType::I64:
      rank_dispatch(int64_t{});
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
    const Type* type, const Type* runtime_type, void* result_ptr) {
  static_assert(std::is_move_constructible<ResultType>::value,
                "Conversion result type must be move constructible");
  static_assert(std::is_move_constructible<ConversionContext>::value,
                "Conversion context type must be move constructible");

  auto* value_type = dyn_cast<AsyncValueType>(type);
  if (!value_type) return mlir::failure();

  // Load the pointer to the async value from a pointer to result storage.
  TFRT_MSAN_MEMORY_IS_INITIALIZED(result_ptr, sizeof(void*));
  void* ret = *reinterpret_cast<void**>(result_ptr);
  auto* value = static_cast<mlir::runtime::AsyncValue*>(ret);

  // We already verified that return value is an async value of memref.
  auto* memref = dyn_cast<MemrefType>(&value_type->value_type());
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
        ExtractAsyncValue(value, dst(), ptr,
                          internal::Emplace<Converter, T, 0>);
        break;
      case 1:
        ExtractAsyncValue(value, dst(), ptr,
                          internal::Emplace<Converter, T, 1>);
        break;
      case 2:
        ExtractAsyncValue(value, dst(), ptr,
                          internal::Emplace<Converter, T, 2>);
        break;
      case 3:
        ExtractAsyncValue(value, dst(), ptr,
                          internal::Emplace<Converter, T, 3>);
        break;
      case 4:
        ExtractAsyncValue(value, dst(), ptr,
                          internal::Emplace<Converter, T, 4>);
        break;
      case 5:
        ExtractAsyncValue(value, dst(), ptr,
                          internal::Emplace<Converter, T, 5>);
        break;
      default:
        // TODO(ezhulenev): Because ExtractAsyncValue takes a llvm::function_ref
        // we can't pass a runtime arguments to emplace functions via lambda
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
  DType element_type = memref->element_type();
  switch (element_type) {
    case DType::F32:
      rank_dispatch(float{});
      break;
    case DType::I1:
      rank_dispatch(bool{});
      break;
    case DType::I32:
      rank_dispatch(int32_t{});
      break;
    case DType::I64:
      rank_dispatch(int64_t{});
      break;
    default:
      results.MakeErrorAt(
          result_index,
          StrCat("unsupported returned memref element type: ", element_type));
  }

  return mlir::success();
}

//----------------------------------------------------------------------------//
// Helper functions for handling errors at runtime.
//----------------------------------------------------------------------------//

// Constructs error async value from the `error` and returns it for all results.
void ReturnErrors(RemainingResults results, Error error);
void ReturnErrors(RemainingResults results, DecodedDiagnostic error);

// Constructs error async value from the `error` and returns it for all results.
// Returns the original error to the caller.
Error ReturnErrors(const ReturnValueConverterBase& results, Error error);

//----------------------------------------------------------------------------//
// Result of compiling MLIR module to executable kernel function.
//----------------------------------------------------------------------------//

class Executable {
 public:
  // Pointer to a compiled kernel function.
  using KernelFunctionPtr = void (*)(void**);

  // Forward declare types defined below.
  struct ResultsMemoryLayout;
  struct CallFrame;
  struct ExecuteOpts;
  struct KernelContext;

  Executable(llvm::StringRef name,
             std::unique_ptr<JitRtMemoryMapper> memory_mapper,
             std::unique_ptr<mlir::ExecutionEngine> engine,
             KernelFunctionPtr fptr, FunctionType signature,
             FunctionType runtime_signature,
             ResultsMemoryLayout results_memory_layout,
             Optional<size_t> specialization,
             std::chrono::milliseconds time_to_compile)
      : name_(name.str()),
        memory_mapper_(std::move(memory_mapper)),
        engine_(std::move(engine)),
        fptr_(fptr),
        signature_(std::move(signature)),
        runtime_signature_(std::move(runtime_signature)),
        results_memory_layout_(std::move(results_memory_layout)),
        specialization_(specialization),
        time_to_compile_(time_to_compile) {
    assert(fptr_ != nullptr && "kernel function must be not null");
  }

  // Initializes call frame by adding all operands as pointers to the arguments
  // vector. Also allocates storage for the returned values. Return values
  // storage requirements inferred from the kernel function signature.
  //
  // This function leaves the kernel context argument (the first argument of a
  // kernel function) uninitialized. It will be initialized in the `Execute`
  // function right before the actual execution.
  //
  // See mlir::ExecutionEngine `packFunctionArguments` for the details.
  Error InitializeCallFrame(ArrayRef<MemrefDesc> operands,
                            CallFrame* call_frame) const;

  // Converts returned values owned by the call frame using provided value
  // converter. If result conversion fails (e.g. result type is not supported)
  // emits error async value for that result.
  //
  // If compiled function execution finished with an error (error flag is `true`
  // in the call frame) emits error async value for all results.
  Error ReturnResults(const ReturnValueConverterBase& results,
                      CallFrame* call_frame) const;

  // Executes compiled function with given operands. If operands passed at
  // runtime are not compatible with the compiled function signature, allocates
  // error async values for all results.
  //
  // Returns compiled function results via the user-provided results converter.
  // If compiled function execution completed in the error state, emits error
  // async value for all results.
  Error Execute(ArrayRef<MemrefDesc> operands,
                const ReturnValueConverterBase& results,
                const ExecuteOpts& opts) const;

  // Executes compiled function using user provided call frame.
  //
  // It is the caller responsibility to handle the compiled function results
  // stored in the call frame.
  void Execute(CallFrame& call_frame, const ExecuteOpts& opts) const;

  bool IsAsync() const { return results_memory_layout_.has_async_results; }

  llvm::StringRef name() const { return name_; }

  Optional<size_t> specialization() const { return specialization_; }

  unsigned num_results() const;

  std::chrono::milliseconds time_to_compile() const;

  // CallFrame provides a pointer-stable storage for packed function arguments
  // and storage for returned values.
  struct CallFrame {
    // Pointers to compiled kernel arguments.
    llvm::SmallVector<void*, 32> args;

    // We use single block of memory to store compiled kernel results. We need
    // to be able to store pointers to async values and tokens, and strided
    // memrefs which at runtime are represented as StridedMemrefType<T, rank>.
    //
    // Currently we only need to provide result storage for pointers and memref
    // sizes and strides (int64_t type). If we'll need to support more complex
    // return types we'll have to be more careful about alignment requirements.
    static_assert(sizeof(uintptr_t) == sizeof(int64_t),
                  "uintptr_t size must be the same as int64_t");

    // Memory where the compiled kernel will write its results.
    llvm::SmallVector<uint8_t, 128> results;

    // Indicates whether the kernel function execution finished with an error.
    bool is_error = false;

    // The error message which is available only if `is_error` is true. The
    // assumption is that the error message string is owned by the compiled
    // binary and the call frame can safely keep a non-owning pointer.
    llvm::StringRef error;
  };

  // Requirements for the contiguous block of memory to store compiled function
  // results. When we invoke a compiled fuction we allocate a block of memory,
  // and pass pointers to pre-computed offsets as output arguments to the
  // function.
  struct ResultsMemoryLayout {
    bool has_async_results;             // true iff returns async results
    size_t size;                        // number of bytes required
    llvm::SmallVector<size_t> offsets;  // offsets in the block of memory
  };

  // Extension point to enable framework-specific integration between the
  // compiled code and the framework on top of the TFRT runtime (e.g. Tensorflow
  // running on top of TFRT). For example only at Tensorflow level it is
  // possible to decide if an input tensor can be forwarded to one of the
  // outputs.
  //
  // Note that runtime integration is generic with respect to the framework
  // used. That is, `runtime::KernelContext` is unaware of the actual framework
  // (e.g. Tensorflow, JAX) that uses code generation at run-time.
  //
  // See go/mlir-rt for details.
  struct KernelContext {
    virtual ~KernelContext() = default;

    // If the allocation of the given size and alignment can be satisfied by one
    // of the inputs, then forward function should return a pointer to the
    // forwarded input memref buffer.
    virtual void* forward(size_t size, size_t alignment,
                          ArrayRef<unsigned> candidates) = 0;
  };

  // Options for configuring compiled kernel execution.
  struct ExecuteOpts {
    ExecuteOpts() : async_task_runner(nullptr), kernel_context(nullptr) {}

    // Async task runner for executing async runtime tasks. Typically it
    // schedules async tasks into the underlying thread pool. It's the caller's
    // responsibility to guarantee that it will outlive the execution of all
    // async tasks started by the executable.
    AsyncTaskRunner* async_task_runner;

    // User-provided kernel context corresponding to the JIT executable.
    // Must outlive all async tasks launched by this executable.
    KernelContext* kernel_context;
  };

  // Verifies that all types in the entrypoint function signature are supported
  // at runtime and we know how to pass arguments and fetch results. Returns
  // a pre-computed layout for the function results. If some of the operands
  // or results are not supported returns an error.
  static Expected<ResultsMemoryLayout> GetResultsMemoryLayout(
      const FunctionType& signature);

 private:
  std::string name_;  // name of the compiled kernel module

  // Called by `engine_`'s destructor; must appear before it.
  std::unique_ptr<JitRtMemoryMapper> memory_mapper_;

  std::unique_ptr<mlir::ExecutionEngine> engine_;
  KernelFunctionPtr fptr_;  // compiled function owned by the `engine_`

  // Signature of the compiled module entrypoint function before lowering to
  // the runtime dialects (see JitExecutable `signature_` for more details).
  FunctionType signature_;

  // Signature of the compiled module entrypoint function after lowering it from
  // high level dialects to the dialects supported by the jitrt runtime.
  //
  // - Operands and results types converted to the types with well-defined ABI
  //   (e.g. tensors converted to memrefs).
  //
  // - First argument is always a kernel context added to the function by the
  //   lowering pipeline.
  //
  // From this signature executable infers how to pack runtime operands
  // according to the expected memory layout, and how to convert results
  // returned from the JIT-compiled function into high level types (e.g. how to
  // convert StridedMemrefType into Tensorflow Tensor).
  //
  // To infer the type of the returned value, executable looks at the type
  // defined by the `runtime_signature_` to get the memory layout of the
  // returned value, and at the type defined by the `signature_` to get the type
  // expected by the runtime.
  FunctionType runtime_signature_;

  ResultsMemoryLayout results_memory_layout_;

  // Specialization id if this executable is a specialization, or an empty
  // optional if this executable is a default one.
  Optional<size_t> specialization_;
  // The time it took to compile this binary.
  std::chrono::milliseconds time_to_compile_;
};

//----------------------------------------------------------------------------//
// JitExecutable to manage multiple compiled executables.
//----------------------------------------------------------------------------//

// JitExecutable owns a default executable compiled from the MLIR module (if
// operands constraints allow that), and orchestrates on-demand re-compilation
// for specific argument ranks, shapes or values depending on the operands
// constraints.
class JitExecutable {
 public:
// TODO(ezhulenev): Use std::any once TFRT switches to C++17.
#if __cplusplus >= 201703L
  using UserData = std::any;
#else
  using UserData = llvm::Any;
#endif

  // Compilation task runner called at runtime when specialization compilation
  // is required with the `TaskFunction` that does the compilation, and updates
  // the internal state of the `JitExecutable`. This runner can be used by the
  // caller to offload compilation task to the specialized thread pool and
  // add tracing events (e.g. add Tensorflow profiler tracing). Task runner must
  // call the `TaskFunction`, otherwise it will lead to the deadlock.
  //
  // Caller can pass arbitrary user data to the `GetExecutable` method, and it
  // will be passed to the runner if recompilation is required. It is guaranteed
  // that the runner will be called in the same thread as `GetExecutable`.
  //
  using CompilationTaskRunner =
      llvm::unique_function<void(size_t, ArrayRef<OperandConstraint>,
                                 ArrayRef<MemrefDesc>, TaskFunction, UserData)>;

  // Inline compilation task runner runs compilation task in the caller thread.
  static void InlineCompilationTaskRunner(
      size_t num_specializations, ArrayRef<OperandConstraint> constraints,
      ArrayRef<MemrefDesc> operands, TaskFunction task, UserData user_data);

  static Expected<JitExecutable> Instantiate(
      string_view mlir_module, string_view entrypoint,
      CompilationOptions compilation_opts, string_view memory_region_name = "",
      CompilationTaskRunner runner = InlineCompilationTaskRunner);

  // Returns entrypoint operands constraints after resolving them using the
  // statically known information in the entrypoint function signature.
  ArrayRef<OperandConstraint> constraints() const;

  // Returns default executable that accepts all compatible operands
  // (operands rank and all static dimensions should match the operands).
  AsyncValuePtr<Executable> DefaultExecutable() const;

  // Returns an executable that may be specialized for the operands shape or
  // values. Can return default executable if no specialization is required, or
  // if the specialized executable is not yet available.
  //
  // Caller can pass arbitrary data via the `user_data` argument, and it will be
  // available to the compilation task runner. This can be used for tracing,
  // e.g. to track what user-level requests triggered recompilation.
  //
  // Returns an error if the operands do not match the expected function
  // signature and specialization is not possible (without trying to compile).
  // If specialization is disabled, returns the default executable without
  // checking the operands (the default executable itself will check operands
  // when called).
  //
  // Async values holding compilation results (executables) cached in the
  // JitExecutable, and successive calls with operands of the same shape
  // (symbolic shape) are cheap. If compilation fails, then the returned async
  // value will hold a compilation error message. Compilation errors are never
  // retried.
  //
  // Note: This function never falls back on the default executable if
  // specialization compilation fails.
  Expected<AsyncValuePtr<Executable>> GetExecutable(
      ArrayRef<MemrefDesc> operands, UserData user_data = {},
      const SpecializationListener* listener = nullptr);

  // Returns an async value that becomes ready when all executables owned by
  // this JitExecutable are compiled (no pending compilation tasks).
  AsyncValueRef<Chain> AllExecutablesCompiled() const;

  // JitExecutable is move-only type.
  JitExecutable(const JitExecutable&) = delete;
  JitExecutable(JitExecutable&&) = default;

 private:
  JitExecutable(string_view mlir_module, string_view entrypoint,
                string_view memory_region_name,
                CompilationOptions compilation_opts,
                ArrayRef<OperandConstraint> constraints, FunctionType signature,
                Optional<Executable> default_executable,
                CompilationTaskRunner runner);

  std::string mlir_module_;
  std::string entrypoint_;

  // Name of the memory region where JIT'ed code is compiled to.
  // This allows profilers to correctly label JIT-executed code.
  // Note: this feature might only be available on some platforms, e.g. Linux.
  std::string memory_region_name_;

  CompilationOptions compilation_opts_;

  // Entrypoint operands constraints after resolving them using the statically
  // known information in the entrypoint function signature. If constraint
  // specified by the argument attribute known to be statically satisfied by the
  // operand type (e.g. rank constraint with an operand of statically known
  // rank), then the constraint value for that operand will be updated to
  // `kResolved`.
  llvm::SmallVector<OperandConstraint> constraints_;

  // True if any of the operands has `OperandConstraint::kValue` constraint.
  bool has_value_constraints_;

  // Signature of the compiled module entrypoint function.
  //
  // This function signature is allowed to have operands and results types
  // without a well-defined ABI (e.g. it can have tensors when compiled module
  // defined in Tensorflow dialect), and it corresponds to the kernel definition
  // in one of the high level dialects (e.g. Tensorflow or mHLO).
  //
  // When compiled module prepared for execution, function operands and results
  // are mapped to the types with well-defined ABI (e.g. tensors mapped to
  // memrefs). See `signature_` documentation in the `Executable` type.
  FunctionType signature_;

  // Symbolic shape resolver assigns symbolic dimensions to runtime operands
  // based on the entrypoint function signature.
  SymbolicShapesResolver symbolic_shapes_resolver_;

  // Default executable that was not specialized to any of the arguments.
  AsyncValueRef<Executable> default_executable_;
  bool has_default_executable_;

  // A custom runner for compiling specializations.
  CompilationTaskRunner runner_;

  // Executables specialized for the arguments shapes or/and values.
  using Specializations = AsyncValuesCache<llvm::hash_code, Executable>;
  std::unique_ptr<Specializations> specializations_;
};

// Resource context caches all JitExecutables in the async value cache.
//
// We use compilation unit id as a cache key. Because this id is unique only
// within a single Bef file, it is the user's responsibility to guarantee that
// the JitExecutableCache is not reused between multiple Bef files.
using JitExecutableCache = AsyncValuesCache<size_t, JitExecutable>;

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_JITRT_H_
