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
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/calling_convention.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/compiler.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/specialization.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/type_converter.h"
#include "third_party/tensorflow/compiler/xla/runtime/arguments.h"
#include "third_party/tensorflow/compiler/xla/runtime/async_runtime.h"
#include "third_party/tensorflow/compiler/xla/runtime/async_values_cache.h"
#include "third_party/tensorflow/compiler/xla/runtime/constraints.h"
#include "third_party/tensorflow/compiler/xla/runtime/custom_call.h"
#include "third_party/tensorflow/compiler/xla/runtime/diagnostics.h"
#include "third_party/tensorflow/compiler/xla/runtime/executable.h"
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

//----------------------------------------------------------------------------//
// Conversions from compiled kernel operands to JitRt runtime types.
//----------------------------------------------------------------------------//

// Converts tfrt Tensor to the Memref descriptor if concrete Tensor type is
// supported (currently only DenseHostTensor can be converted). Returns error
// otherwise.
Expected<MemrefDesc> ConvertTensorToMemrefDesc(const Tensor& tensor);

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

  // XLA program can be specialized and recompiled at runtime to the concrete
  // input shapes and sometimes values (e.g. reduciton dimension).
  enum class Specialization {
    // Recompile specialized kernels when needed.
    kEnabled,
    // Completely disable specialized kernels (always call default executable).
    kDisabled,
    // Always use specialized kernels, and never call default executable (only
    // required for getting reproducible results in benchmarks).
    kAlways,
  };

  struct Options {
    // What level of specialization is enabled at runtime.
    Specialization specialization = Specialization::kAlways;

    // Options for the XLA runtime JitCompiler.
    xla::runtime::JitCompiler::Options compiler;
  };

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
      string_view mlir_module, string_view entrypoint, Options opts,
      string_view memory_region_name = "",
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
                string_view memory_region_name, Options opts,
                ArrayRef<ArgumentConstraint> constraints,
                FunctionType signature, Optional<Executable> default_executable,
                CompilationTaskRunner runner);

  std::string mlir_module_;
  std::string entrypoint_;

  // Name of the memory region where JIT'ed code is compiled to.
  // This allows profilers to correctly label JIT-executed code.
  // Note: this feature might only be available on some platforms, e.g. Linux.
  std::string memory_region_name_;

  Options opts_;

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
