/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

//===- cpurt.h --------------------------------------------------*- C++ -*-===//
//
// Support library for implementing TFRT kernels that do JIT compilation using
// MLIR framework (generating kernels at runtime from hight level MLIR
// dialects, e.g. generating dense linear algebra kernels from Linalg dialect).
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_JIT_CPURT_H_
#define TFRT_BACKENDS_CPU_JIT_CPURT_H_

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class ExecutionContext;
class Tensor;

namespace cpu {
namespace jit {

// Forward declare the result of compiling MLIR module to the executable.
class CompilationResult;

struct CompilationOptions {
  // The number of worker threads (host context concurrent work queue size) that
  // can be used for parallelizing compute intensive parts of the kernel.
  int num_worker_threads;

  // LLVM optimization level when JIT compiling a kernel.
  Optional<llvm::CodeGenOpt::Level> jit_code_opt_level;
};

// Compiles a kernel defined by the serialized MLIR module to the executable
// compilation result.
Expected<CompilationResult> CompileKernelMlirModule(
    string_view mlir_module, string_view entrypoint,
    const CompilationOptions& opts);

//----------------------------------------------------------------------------//
// Result of compiling MLIR module to executable kernel function.
//----------------------------------------------------------------------------//

class CompilationResult {
 public:
  CompilationResult(string_view entrypoint,
                    std::unique_ptr<mlir::ExecutionEngine> engine)
      : engine_(std::move(engine)), fptr_(*engine_->lookup(entrypoint)) {
    assert(fptr_ != nullptr && "entrypoint was not found");
  }

  // Execute compiled function with given Tensor operands. Returns error async
  // value if operands are not compatible with compiled function signature.
  AsyncValueRef<Chain> Execute(RepeatedArguments<Tensor> operands,
                               const ExecutionContext& exec_ctx) const;

  // Tensor operands converted to compiled function memref operands.
  struct MemrefArg {
    void* data;
    ssize_t offset;
    SmallVector<ssize_t, 4> sizes;
    SmallVector<ssize_t, 4> strides;
  };

  // CallFrame provides a pointer-stable storage for packed function arguments.
  struct CallFrame {
    // For now we only support functions that return async tokens, which at
    // runtime is represented as a void pointer.
    using ReturnType = void*;

    llvm::SmallVector<MemrefArg, 4> memrefs;
    ReturnType return_value;
  };

 private:
  // Pointer to a compiled kernel function.
  using KernelFunctionPtr = void (*)(void**);

  std::unique_ptr<mlir::ExecutionEngine> engine_;
  KernelFunctionPtr fptr_;
};

//----------------------------------------------------------------------------//
// Cache all compilation results in the resource context owned by the host.
//----------------------------------------------------------------------------//

class CompilationResultCache {
 public:
  explicit CompilationResultCache(HostContext* host) : host_(host) {}
  AsyncValueRef<CompilationResult> Find(intptr_t key) const;
  AsyncValueRef<CompilationResult> Insert(intptr_t key,
                                          CompilationResult compilation_result);

 private:
  HostContext* host_;
  mutable tfrt::mutex mu_;
  llvm::DenseMap<intptr_t, AsyncValueRef<CompilationResult>> cache_
      TFRT_GUARDED_BY(mu_);
};

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_JIT_CPURT_H_
