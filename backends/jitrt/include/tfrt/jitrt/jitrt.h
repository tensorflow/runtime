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

#include <any>
#include <chrono>  // NOLINT(build/c++11)
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/jitrt/results.h"
#include "tfrt/support/forward_decls.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/specialization.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/type_converter.h"
#include "third_party/tensorflow/compiler/xla/runtime/arguments.h"
#include "third_party/tensorflow/compiler/xla/runtime/async_runtime.h"
#include "third_party/tensorflow/compiler/xla/runtime/async_values_cache.h"
#include "third_party/tensorflow/compiler/xla/runtime/constraints.h"
#include "third_party/tensorflow/compiler/xla/runtime/custom_call.h"
#include "third_party/tensorflow/compiler/xla/runtime/diagnostics.h"
#include "third_party/tensorflow/compiler/xla/runtime/execution_engine.h"
#include "third_party/tensorflow/compiler/xla/runtime/memory_mapper.h"
#include "third_party/tensorflow/compiler/xla/runtime/symbolic_shape.h"
#include "third_party/tensorflow/compiler/xla/runtime/types.h"

// Forward declare Eigen types.
namespace Eigen {
class ThreadPoolInterface;
}  // namespace Eigen

namespace mlir {
class PassManager;
}  // namespace mlir

// Forward declare types enabling compiled kernel <-> runtime integration.
namespace xla {
namespace runtime {
struct KernelContext;
}  // namespace runtime
}  // namespace xla

namespace tfrt {

class Tensor;

namespace jitrt {

// Compiled module example:
//
//   module @kernel attributes { tfrt.compiled } {
//     func @main(
//       %input0: memref<*xf32>   { rt.constraint = "rank"  },
//       %input1: memref<?x?xf32> { rt.constraint = "shape" },
//       %perm: memref<4xi32>     { rt.constraint = "value" }
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
//     %arg : tensor<*xf32> { rt.constraint = "rank" }
//
//     Before compiling the function, unranked input type will be updated to the
//     corresponding ranked input type (e.g. unranked tensor -> ranked tensor).
//
// (b) Shape constraint:
//
//     %arg : tensor<?x?xf32> { rt.constraint = "shape" }
//
//     Shape of the runtime argument will be used to specialize the compiled
//     function, if this shape seen the first time, it will trigger function
//     recompilation.
//
// (c) Value constraint:
//
//     %reduction_dimension : tensor<i32> { rt.constraint = "value" }
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

// Converts a custom call library into the execution engine symbols binding.
ExecutionEngine::SymbolsBinding GetSymbolsBinding(DirectCustomCallLibrary lib);

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

  // Returns a calling convention that (1) prepends the kernel context argument,
  // (2) uses the user-provided type converter to convert all inputs and results
  // types, and (3) converts result types into out-params by appending them
  // to the arguments.
  static CallingConvention ResultsToOutsCallingConvention(mlir::TypeConverter);

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
  llvm::CodeGenOpt::Level jit_code_opt_level = llvm::CodeGenOpt::Level::Default;

  // Runtime symbol map allows to pass user-defined bindings for symbols at JIT
  // compilation time (e.g. to implement custom C APIs).
  ExecutionEngine::SymbolsBinding runtime_symbol_map;

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
  // All types in the converted function signature should have a registered
  // type conversion (see `type_converter` below) to a type with defined
  // argument or result ABI (see Type::ArgumentAbi and Type::ResultAbi).
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
  //    return results back to the caller (see `rt-convert-to-entrypoint` pass).
  //
  // Calling convention function type is a JitRt-compatible description of the
  // compiled kernel ABI, so that JitRt can correctly initialize CallFrame
  // arguments, allocate memory for returned results, and then correctly decode
  // results memory into the high level types (e.g. convert returned memref
  // descriptor to a Tensorfow tensor).
  CallingConvention calling_convention = DefaultCallingConvention();

  // Type converter converts MLIR types to the corresponding run time types.
  // Executable uses its own type hierarchy, parallel to MLIR's, so that it
  // doesn't depend on any parts of the MLIR after compilation produces an
  // executable artifact, because keeping MLIR context alive can be expensive in
  // terms of memory usage.
  //
  // As a side effect, it allows loading AOT compiled executables from the obj
  // files without any dependencies on MLIR.
  //
  // Default type converter knows how to convert canonical MLIR types (memrefs,
  // tensors, etc...). All user-defined types used at the compiled function
  // boundary (arguments or results) should register a custom type conversion.
  //
  // When we compile the input IR, we first apply the `calling_convention` to
  // get the MLIR function type for the entrypoint, and then we convert it to
  // the corresponding run time function type.
  TypeConverter type_converter;
};

//----------------------------------------------------------------------------//
// Conversions from compiled kernel operands to JitRt runtime types.
//----------------------------------------------------------------------------//

// Converts tfrt Tensor to the Memref descriptor if concrete Tensor type is
// supported (currently only DenseHostTensor can be converted). Returns error
// otherwise.
Expected<MemrefDesc> ConvertTensorToMemrefDesc(const Tensor& tensor);

//----------------------------------------------------------------------------//
// Result of compiling MLIR module to executable kernel function.
//----------------------------------------------------------------------------//

namespace internal {
class JitCompilationContext;
}  // namespace internal

class Executable {
 public:
  // Forward declare types defined below.
  struct ArgumentsMemoryLayout;
  struct ResultsMemoryLayout;
  struct CallFrame;
  struct ExecuteOpts;
  struct KernelContext;

  // Initializes call frame by adding all arguments according to the compiled
  // kernel ABI. Also allocates storage for the returned values according to the
  // results memory layout.
  //
  // If `verify_arguments` is true (in debug mode it's always on, independent of
  // the argument value) this function also verifies that operands passed at run
  // time matches the executable entrypoint signature (e.g. all statically known
  // dimensions of the memrefs matches the operands). Returns an error if finds
  // a mismatch.
  //
  // This function leaves the kernel context argument (the first argument of a
  // kernel function) uninitialized. It will be initialized in the `Execute`
  // function right before the actual execution.
  Error InitializeCallFrame(ArgumentsRef arguments, CallFrame* call_frame,
                            bool verify_arguments = true) const;

  // Converts returned values owned by the call frame using provided result
  // converter. If compiled function execution finished with an error (error
  // flag is `true` in the call frame) returns error for all results.
  Error ReturnResults(const ResultConverter& results,
                      CallFrame* call_frame) const;

  // Executes compiled function with given arguments.
  //
  // If `verify_arguments` is true (in debug mode it's always on, independent of
  // the argument value) this function also verifies that arguments passed at
  // run time matches the executable entrypoint signature. If some of the
  // arguments do not match the expected type, this function allocates error
  // async values for all results and returns an error.
  //
  // Returns compiled function results via the user-provided results converter.
  // If execution completed in the error state, returns error for all results.
  Error Execute(ArgumentsRef arguments, const ResultConverter& results,
                const ExecuteOpts& opts, bool verify_arguments = true) const;

  // Executes compiled function using user provided call frame.
  //
  // It is the caller responsibility to handle the compiled function results
  // stored in the call frame.
  void Execute(CallFrame& call_frame, const ExecuteOpts& opts) const;

  bool IsAsync() const { return results_memory_layout_.has_async_results; }

  llvm::StringRef name() const { return name_; }

  Optional<size_t> specialization() const { return specialization_; }

  // Returns the number of results in the runtime signature.
  unsigned num_results() const;

  // Signature of the compiled module entrypoint function before lowering to
  // the runtime dialects. See JitExecutable's `signature_` for more details.
  const FunctionType& signature() const;

  // Signature of the compiled module entrypoint function after lowering it from
  // high level dialects to the dialects supported by the jitrt runtime.
  // See JitExecutable's `signature_` for more details.
  const FunctionType& runtime_signature() const;

  std::chrono::milliseconds time_to_compile() const;

  // Get the object file behind this executable (on linux for example, it will
  // be https://en.wikipedia.org/wiki/Executable_and_Linkable_Format
  // executable). Can be null.
  std::unique_ptr<llvm::MemoryBuffer> obj_file() const;

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

    // Tracks whether any of the outputs were set.
    bool has_set_outputs = false;

    // Indicates whether the kernel function execution finished with an error.
    bool is_error = false;

    // The error message which is available only if `is_error` is true. The
    // assumption is that the error message string is owned by the compiled
    // binary and the call frame can safely keep a non-owning pointer.
    llvm::StringRef error;
  };

  // Requirements for passing arguments to the compiled function.
  struct ArgumentsMemoryLayout {
    // Currently we always pass arguments as an array of pointers.
    size_t num_args_ptrs = 0;
  };

  // Requirements for the contiguous block of memory to store compiled function
  // results. When we invoke a compiled fuction we allocate a block of memory,
  // and pass pointers to pre-computed offsets as output arguments to the
  // function.
  struct ResultsMemoryLayout {
    bool has_async_results = false;     // true iff returns async results
    size_t size = 0;                    // number of bytes required
    llvm::SmallVector<size_t> offsets;  // offsets in the block of memory
  };

  // Options for configuring compiled kernel execution.
  struct ExecuteOpts {
    // Async task runner for executing async runtime tasks. Typically it
    // schedules async tasks into the underlying thread pool. It's the caller's
    // responsibility to guarantee that it will outlive the execution of all
    // async tasks started by the executable.
    AsyncTaskRunner* async_task_runner = nullptr;

    // A container for passing arbitrary user-provided data to the custom call
    // handlers. Must outlive all async tasks launched by this executable.
    CustomCall::UserData* custom_call_data = nullptr;

    // Diagnostic engine is responsible for passing runtime diagnostics back
    // to the caller through the diagnostic handler.
    DiagnosticEngine* diagnostic_engine = nullptr;
  };

  // Loads executable from an object file. It is the caller responsibility to
  // guarantee that signatures do match the compiled function in the object
  // file, otherwise it will surely lead to crash.
  static Expected<Executable> LoadFromObjFile(
      llvm::StringRef name, std::unique_ptr<llvm::MemoryBuffer> obj_file,
      llvm::StringRef entrypoint, FunctionType signature,
      FunctionType runtime_signature,
      ExecutionEngine::SymbolsBinding runtime_symbol_map = {},
      llvm::StringRef memory_region_name = "");

  // Verifies that all operands types in the entrypoint function signature are
  // supported at run time . Returns a pre-computed layout for the function
  // arguments. If some arguments are not supported returns an error.
  static Expected<ArgumentsMemoryLayout> GetArgumentsMemoryLayout(
      const FunctionType& signature);

  // Verifies that all results types in the entrypoint function signature are
  // supported at run time . Returns a pre-computed layout for the function
  // results. If some results are not supported returns an error.
  static Expected<ResultsMemoryLayout> GetResultsMemoryLayout(
      const FunctionType& signature);

  // TODO(ezhulenev): The following three functions should be decoupled from
  // the jitrt header file (maybe move them to runtime.h?) so that custom call
  // implementations do not have to depend on the `jitrt` target.

  // Returns the user data passed via the ExecuteOpts to the compiled kernel.
  static CustomCall::UserData* GetUserData(xla::runtime::KernelContext* ctx);

  // Returns the diagnostic engine passed via the ExecuteOpts to the compiled
  // kernel.
  static DiagnosticEngine* GetDiagnosticEngine(
      xla::runtime::KernelContext* ctx);

  // Calls the custom call handler with the given runtime context, arguments and
  // attributes.
  static mlir::LogicalResult Call(xla::runtime::KernelContext* ctx,
                                  CustomCall& call, void** args, void** attrs);

 private:
  friend class internal::JitCompilationContext;

  Executable(llvm::StringRef name,
             std::unique_ptr<XlaRuntimeMemoryMapper> memory_mapper,
             std::unique_ptr<ExecutionEngine> engine, FunctionType signature,
             FunctionType runtime_signature,
             ArgumentsMemoryLayout arguments_memory_layout,
             ResultsMemoryLayout results_memory_layout,
             Optional<size_t> specialization,
             std::chrono::milliseconds time_to_compile)
      : name_(name.str()),
        memory_mapper_(std::move(memory_mapper)),
        engine_(std::move(engine)),
        fptr_(engine_->entrypoint()),
        signature_(std::move(signature)),
        runtime_signature_(std::move(runtime_signature)),
        arguments_memory_layout_(std::move(arguments_memory_layout)),
        results_memory_layout_(std::move(results_memory_layout)),
        specialization_(specialization),
        time_to_compile_(time_to_compile) {
    assert(fptr_ != nullptr && "kernel function must be not null");
  }

  std::string name_;  // name of the compiled kernel module

  // Called by `engine_`'s destructor; must appear before it.
  std::unique_ptr<XlaRuntimeMemoryMapper> memory_mapper_;  // optional

  // JitRt execution engine owns the LLVM ORC jit compilation stack.
  std::unique_ptr<ExecutionEngine> engine_;

  // Compiled function owned by the `engine_`.
  ExecutionEngine::EntrypointFunctionPtr fptr_;

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

  ArgumentsMemoryLayout arguments_memory_layout_;
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
  using UserData = std::any;

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
      llvm::unique_function<void(size_t, ArrayRef<ArgumentConstraint>,
                                 ArgumentsRef, TaskFunction, UserData)>;

  // Inline compilation task runner runs compilation task in the caller thread.
  static void InlineCompilationTaskRunner(
      size_t num_specializations, ArrayRef<ArgumentConstraint> constraints,
      ArgumentsRef arguments, TaskFunction task, UserData user_data);

  static Expected<JitExecutable> Instantiate(
      string_view mlir_module, string_view entrypoint,
      CompilationOptions compilation_opts, string_view memory_region_name = "",
      CompilationTaskRunner runner = InlineCompilationTaskRunner);

  // Returns entrypoint operands constraints after resolving them using the
  // statically known information in the entrypoint function signature.
  ArrayRef<ArgumentConstraint> constraints() const;

  // Returns default executable that accepts all compatible operands
  // (operands rank and all static dimensions should match the operands).
  AsyncValuePtr<Executable> DefaultExecutable() const;

  // Returns an executable that may be specialized for the arguments. Can return
  // default executable if no specialization is required, or if the specialized
  // executable is not yet available.
  //
  // Caller can pass arbitrary data via the `user_data` argument, and it will be
  // available to the compilation task runner. This can be used for tracing,
  // e.g. to track what user-level requests triggered recompilation.
  //
  // Returns an error if the arguments do not match the expected function
  // signature and specialization is not possible (without trying to compile).
  // If specialization is disabled, returns the default executable without
  // checking the arguments (the default executable itself will check arguments
  // when called).
  //
  // Async values holding compilation results (executables) cached in the
  // JitExecutable, and successive calls with the same arguments are cheap (the
  // definition of "same" depend on the argument type specialization and chosen
  // hash function, e.g. shaped arguments compared using their symbolic shape).
  // If compilation fails, then the returned async value will hold a compilation
  // error message. Compilation errors are never retried.
  //
  // Note: This function never falls back on the default executable if
  // specialization compilation fails.
  Expected<AsyncValuePtr<Executable>> GetExecutable(
      ArgumentsRef arguments, UserData user_data = {},
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
                ArrayRef<ArgumentConstraint> constraints,
                FunctionType signature, Optional<Executable> default_executable,
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
  llvm::SmallVector<ArgumentConstraint> constraints_;

  // True if any of the operands has `ArgumentConstraint::kValue` constraint.
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
