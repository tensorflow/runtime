// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- contraction_output_kernel.cc -----------------------------*- C++ -*-===//
//
// Jit compiled contraction output kernel implementation.
//
//===----------------------------------------------------------------------===//

#include "tfrt/cpu/jit/contraction_output_kernel.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "tfrt/host_context/shared_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"

namespace tfrt {
namespace cpu {
namespace jit {

//----------------------------------------------------------------------------//
// CompiledContractionOutputKernel owns the mlir ExecutionEngine that holds jit
// code generation result.
//----------------------------------------------------------------------------//

class CompiledContractionOutputKernel {
 public:
  explicit CompiledContractionOutputKernel(
      std::unique_ptr<mlir::ExecutionEngine> engine)
      : engine_(std::move(engine)) {}

  void Call(void* data, int64_t stride, int64_t row_offset, int64_t col_offset,
            int64_t rows, int64_t cols) {
    auto fptr = engine_->lookup("compute");
    assert(fptr);

    // NOTE: Jitted output kernel expects a memref in row-major layout, so we
    // swap rows with columns when we pass output block to the output kernel.

    // Pack arguments for the `StridedMemRefType` of rank `2`.
    void* memref_data = data;
    int64_t memref_offset = 0;
    int64_t memref_size_0 = cols;
    int64_t memref_size_1 = rows;
    int64_t memref_stride_0 = stride;
    int64_t memref_stride_1 = 1;

    llvm::SmallVector<void*, 1> args;
    args.push_back(&memref_data);      // memref.basePtr
    args.push_back(&memref_data);      // memref.data
    args.push_back(&memref_offset);    // memref.offset
    args.push_back(&memref_size_0);    // memref.sizes[0]
    args.push_back(&memref_size_1);    // memref.sizes[1]
    args.push_back(&memref_stride_0);  // memref.strides[0]
    args.push_back(&memref_stride_1);  // memref.strides[1]
    // Offsets are also swapped.
    args.push_back(&col_offset /*row_offset*/);
    args.push_back(&row_offset /*col_offset*/);

    (*fptr)(args.data());
  }

 private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;
};

namespace {

//----------------------------------------------------------------------------//
// Setup MLIR pass pipeline to lower to LLVM dialect, and use ORC JIT to codegen
// functions at runtime.
//----------------------------------------------------------------------------//

void InitializeCompiler() {
  static const bool initialized = ([]() -> bool {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::registerAllDialects();
    mlir::registerAllPasses();

    return true;
  })();
  (void)initialized;
}

struct CompilationOptions {
  Optional<llvm::CodeGenOpt::Level> jit_code_opt_level;
};

Expected<CompiledContractionOutputKernel> Compile(string_view source,
                                                  CompilationOptions opts) {
  auto str = source.str();
  auto src = llvm::MemoryBuffer::getMemBuffer(str, "<unknown>");

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(src), llvm::SMLoc());

  mlir::MLIRContext context;
  mlir::OwningModuleRef module(mlir::parseSourceFile(source_mgr, &context));
  if (!module) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Failed to parse kernel source");
  }

  // TODO(ezhulenev): Validate that function signature is a valid contraction
  // kernel output function.

  mlir::LowerToLLVMOptions lower_to_llvm_opts;
  mlir::PassManager pm(&context);
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createLowerToLLVMPass(lower_to_llvm_opts));

  if (failed(pm.run(*module))) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Failed to lower module to LLVM");
  }

  // Prepare JIT target machine for code generation.
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder) return builder.takeError();

  auto target_machine = builder->createTargetMachine();
  if (!target_machine) return target_machine.takeError();

  // Link with shared libraries for symbol resolution.
  llvm::SmallVector<llvm::StringRef, 4> libs;

  // Additional LLVM passes to run.
  llvm::SmallVector<const llvm::PassInfo*, 4> passes;
  auto transformer = mlir::makeLLVMPassesTransformer(passes, /*mbOptLevel=*/0,
                                                     target_machine->get());

  // Build MLIR exection engine.
  auto engine = mlir::ExecutionEngine::create(*module, transformer,
                                              opts.jit_code_opt_level, libs);
  if (!engine) return engine.takeError();

  return CompiledContractionOutputKernel(std::move(*engine));
}

//----------------------------------------------------------------------------//
// CompiledContractionOutputKernelsContext owns all compiled contraction output
// kernels in a SharedContext that is owned by a HostContext.
//----------------------------------------------------------------------------//

class CompiledKernelsContext : public SharedContext {
 public:
  explicit CompiledKernelsContext(HostContext* host_context) {}

  CompiledKernelsContext(const CompiledKernelsContext&) = delete;
  void operator=(const CompiledKernelsContext&) = delete;

  Expected<CompiledContractionOutputKernel*> GetCompiledKernel(
      string_view key, string_view mlir_module) {
    {  // Fast path. Check if the kernel is already compiled.
      mutex_lock lock(mu_);
      auto it = kernels_.find(key);
      if (it != kernels_.end()) return {&it->getValue()};
    }

    // Compile the kernel without holding a lock.
    auto compiled = Compile(mlir_module, opts_);
    if (!compiled) return compiled.takeError();

    // Double check that concurrent execution did not already compile the kernel
    // before the current thread of execution.
    mutex_lock lock(mu_);
    auto it = kernels_.find(key);
    if (it != kernels_.end()) return {&it->getValue()};

    auto inserted = kernels_.try_emplace(key, std::move(*compiled));
    return &inserted.first->getValue();
  }

 private:
  mutex mu_;
  CompilationOptions opts_;
  llvm::StringMap<CompiledContractionOutputKernel> kernels_
      TFRT_GUARDED_BY(mu_);
};

}  // namespace

// Returns contraction output kernel compiled from the MLIR module.
Expected<CompiledContractionOutputKernel*> GetCompiledContractionOutputKernel(
    HostContext* host, string_view function_name, string_view mlir_module) {
  InitializeCompiler();

  auto& ctx = host->GetOrCreateSharedContext<CompiledKernelsContext>();
  // TODO(ezhulenev): Create a shorter fingerprint from the mlir module to use
  // as a lookup key for the cache.
  const string_view key = mlir_module;
  return ctx.GetCompiledKernel(key, mlir_module);
}

// Calls compiled output kernel for each column of the contraction output block.
void CallCompiledContractionOutputKernel(
    CompiledContractionOutputKernel* kernel, DType dtype, void* data,
    int64_t stride, int64_t row_offset, int64_t col_offset, int64_t rows,
    int64_t cols) {
  kernel->Call(data, stride, row_offset, col_offset, rows, cols);
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
