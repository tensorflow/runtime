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

//===- cpurt.cc - ---------------------------------------------------------===//
//
// Support library for implementing TFRT kernels that do JIT compilation using
// MLIR framework.
//
//===----------------------------------------------------------------------===//

#include "tfrt/cpu/jit/cpurt.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "tfrt/cpu/jit/async_runtime.h"
#include "tfrt/cpu/jit/async_runtime_api.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor.h"

namespace tfrt {
namespace cpu {
namespace jit {

// Enable IR printing during the kernel compilation pipeline execution.
static bool DebugCpurtCompile() {
#if defined(DEBUG_CPURT)
  return true;
#else
  return false;
#endif
}

using MemrefArg = CompilationResult::MemrefArg;
using CallFrame = CompilationResult::CallFrame;

//----------------------------------------------------------------------------//
// CompilationResult implementation.
//----------------------------------------------------------------------------//

// Converts Tensor to Memref argument for supported tensor types. Returns
// error otherwise.
static Expected<MemrefArg> ConvertTensorToMemrefArg(const Tensor& tensor) {
  if (auto* dht = dyn_cast<DenseHostTensor>(&tensor)) {
    MemrefArg memref;
    memref.data = const_cast<void*>(dht->data());
    memref.offset = 0;
    dht->shape().GetDimensions(&memref.sizes);
    dht->shape().GetStrides(&memref.strides);
    return memref;
  }

  return MakeStringError("unsupported tensor type: ", tensor.tensor_type());
}

// Unpack `memref` argument into pointers to the data to be compatible with
// compiled MLIR function ABI.
static void AddMemrefArgument(const MemrefArg& memref,
                              llvm::SmallVectorImpl<void*>* args) {
  assert(memref.sizes.size() == memref.strides.size());
  auto add_arg = [&](const void* p) { args->push_back(const_cast<void*>(p)); };
  add_arg(&memref.data);  // memref.basePtr
  add_arg(&memref.data);  // memref.data
  add_arg(&memref.offset);
  size_t rank = memref.sizes.size();
  for (int i = 0; i < rank; ++i) add_arg(&memref.sizes[i]);
  for (int i = 0; i < rank; ++i) add_arg(&memref.strides[i]);
}

// Initializes call frame by adding all arguments to it to ensure that all the
// arguments are alive when we call compiled kernel.
//
// Returns compiled function arguments as `void*` type erased pointers. These
// pointers are pointing to the arguments that are stored in the CallFrame
// instance. See mlir::ExecutionEngine `packFunctionArguments` for the details.
static Expected<llvm::SmallVector<void*, 32>> InitializeCallFrame(
    RepeatedArguments<Tensor> operands, CallFrame* call_frame) {
  llvm::SmallVector<void*, 32> args;

  // Pack all Tensor operands as memref arguments.
  for (Tensor& tensor : operands) {
    Expected<MemrefArg> memref = ConvertTensorToMemrefArg(tensor);
    if (auto err = memref.takeError()) return std::move(err);
    call_frame->memrefs.push_back(*memref);
    AddMemrefArgument(call_frame->memrefs.back(), &args);
  }

  // Address to write the returned async.token value.
  args.push_back(&call_frame->return_value);
  return args;
}

AsyncValueRef<Chain> CompilationResult::Execute(
    RepeatedArguments<Tensor> operands,
    const ExecutionContext& exec_ctx) const {
  // CallFrame can be allocated on the stack because compiled function will
  // unpack all the arguments it needs, and async regions will not access
  // the data after the initial function will return the result.
  CallFrame call_frame;

  // Compiled function takes arguments as `void**` type erased pointer. See
  // mlir::ExecutionEngine `packFunctionArguments` for the details.
  auto args = InitializeCallFrame(operands, &call_frame);
  if (auto err = args.takeError())
    return EmitErrorAsync(exec_ctx, std::move(err));

  // Set the AsyncRuntime context to be used by all async tasks.
  ResourceContext* res_ctx = exec_ctx.resource_context();
  AsyncRuntime* runtime = res_ctx->GetOrCreateResource<AsyncRuntime>(
      "cpurt.runtime", exec_ctx.host());
  SetAsyncRuntimeContext(runtime);

  // Call the compiled function.
  (*fptr_)(args->data());

  return ConverAsyncTokenToChain(
      static_cast<mlir::runtime::AsyncToken*>(call_frame.return_value));
}

//----------------------------------------------------------------------------//
// CompilationResultCache implementation.
//----------------------------------------------------------------------------//

AsyncValueRef<CompilationResult> CompilationResultCache::Find(
    intptr_t key) const {
  tfrt::mutex_lock lock(mu_);
  auto it = cache_.find(key);
  if (it != cache_.end()) return it->second.CopyRef();
  return AsyncValueRef<CompilationResult>();
}

AsyncValueRef<CompilationResult> CompilationResultCache::Insert(
    intptr_t key, CompilationResult compilation_result) {
  tfrt::mutex_lock lock(mu_);
  auto it = cache_.find(key);
  if (it != cache_.end()) return it->second.CopyRef();

  auto emplaced =
      cache_.try_emplace(key, MakeAvailableAsyncValueRef<CompilationResult>(
                                  host_, std::move(compilation_result)));
  return emplaced.first->getSecond().CopyRef();
}

//----------------------------------------------------------------------------//
// Setup MLIR pass pipeline to lower to LLVM dialect, and use ORC JIT to codegen
// functions at runtime.
//----------------------------------------------------------------------------//

static void InitializeCompiler() {
  static const bool initialized = ([] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  })();
  (void)initialized;
}

// Runs the pipeline to lower kernel IR to LLVM dialect.
static mlir::LogicalResult LowerToLlvm(mlir::MLIRContext* context,
                                       mlir::ModuleOp module,
                                       const CompilationOptions& opts) {
  mlir::PassManager pm(context);
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // TODO(ezhulenev): Move this to a pipeline exposed upstream when it will
  // stabilize, e.g. `LinalgToAsyncRuntime`.

  // Convert all linalg operations to parallel loops, and then add async
  // operations to actually execute them in parallel using the async runtime.
  mlir::OpPassManager& fpm = pm.nest<mlir::FuncOp>();
  fpm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
  fpm.addPass(mlir::createAsyncParallelForPass(opts.num_worker_threads));
  fpm.addPass(mlir::createAsyncRefCountingPass());
  fpm.addPass(mlir::createAsyncRefCountingOptimizationPass());
  fpm.addPass(mlir::createStdExpandOpsPass());

  // Lower from high level async operations to async runtime.
  pm.addPass(mlir::createAsyncToAsyncRuntimePass());

  // Lower everything down to LLVM dialect.
  mlir::LowerToLLVMOptions lower_to_llvm_opts;
  pm.addPass(mlir::createConvertAsyncToLLVMPass());
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createLowerToLLVMPass(lower_to_llvm_opts));

  // Print IR after all passes.
  if (DebugCpurtCompile()) {
    context->disableMultithreading();
    pm.enableIRPrinting([](mlir::Pass*, mlir::Operation*) { return false; },
                        [](mlir::Pass*, mlir::Operation*) { return true; },
                        /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false, llvm::errs());
  }

  return pm.run(module);
}

Expected<CompilationResult> CompileKernelMlirModule(
    string_view mlir_module, string_view entrypoint,
    const CompilationOptions& opts) {
  // Setup LLVM target for code generation.
  InitializeCompiler();

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(mlir_module, "<unknown>"),
      llvm::SMLoc());

  // Register MLIR dialects supported by the compiled kernels.
  mlir::MLIRContext context;
  context.getDialectRegistry()
      .insert<mlir::async::AsyncDialect, mlir::linalg::LinalgDialect,
              mlir::scf::SCFDialect, mlir::StandardOpsDialect,
              mlir::LLVM::LLVMDialect>();

  // Parse a kernel source code into the MLIR Module.
  mlir::OwningModuleRef module(mlir::parseSourceFile(source_mgr, &context));
  if (!module) return MakeStringError("failed to parse kernel source");

  // Lower kernel IR from high level dialects to the MLIR LLVM Dialect.
  if (failed(LowerToLlvm(&context, *module, opts)))
    return MakeStringError("Failed to lower module to LLVM");

  // Prepare JIT target machine for code generation.
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder) return builder.takeError();

  auto target_machine = builder->createTargetMachine();
  if (!target_machine) return target_machine.takeError();

  // Link with shared libraries for symbol resolution.
  llvm::SmallVector<llvm::StringRef, 4> libs;

  // Additional LLVM passes to run.
  llvm::SmallVector<const llvm::PassInfo*, 4> passes;
  auto transformer = mlir::makeLLVMPassesTransformer(passes, /*mbOptLevel=*/2,
                                                     target_machine->get());

  // Build MLIR exection engine.
  auto engine =
      mlir::ExecutionEngine::create(*module, /*llvmModuleBuilder=*/nullptr,
                                    transformer, opts.jit_code_opt_level, libs);
  if (!engine) return engine.takeError();

  // Register Async Runtime API intrinsics.
  (*engine)->registerSymbols(AsyncRuntimeApiSymbolMap);

  return CompilationResult(entrypoint, std::move(*engine));
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
