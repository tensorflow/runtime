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

#include <cstdint>

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
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tfrt/cpu/jit/async_runtime.h"
#include "tfrt/cpu/jit/async_runtime_api.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/host_buffer.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/string_util.h"
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

using CallFrame = CompilationResult::CallFrame;

//----------------------------------------------------------------------------//
// CompilationResult arguments packing.
//----------------------------------------------------------------------------//

// TODO(ezhulenev): Currently codegen supports only F32 data type for
// simplicity. Add support for all other number data types when the overall
// code structure will become stable.

// TODO(ezhulenev): Add support UnrankedMemrefType arguments and results.

static bool IsValidMemref(mlir::Type type) {
  auto memref = type.dyn_cast<mlir::MemRefType>();
  if (!memref) return false;

  mlir::Type elt = memref.getElementType();
  return elt.isF32();
}

static bool IsValidRetType(mlir::Type type) {
  if (type.isa<mlir::async::TokenType>()) return true;

  if (auto value = type.dyn_cast<mlir::async::ValueType>())
    return IsValidMemref(value.getValueType());

  return false;
}

/*static*/ Error CompilationResult::VerifyEntrypointSignature(
    mlir::FunctionType signature) {
  // Arguments must be ranked memrefs of supported types.
  for (unsigned i = 0; i < signature.getNumInputs(); ++i)
    if (!IsValidMemref(signature.getInput(i)))
      return MakeStringError("input #", i, " must be a ranked memref type");

  // Return value must be async tokens or async values of valid memrefs.
  for (unsigned i = 0; i < signature.getNumResults(); ++i)
    if (!IsValidRetType(signature.getResult(i)))
      return MakeStringError(
          "result #", i,
          " must be an async value of ranked memref type or async token");

  return Error::success();
}

// -------------------------------------------------------------------------- //
// Converting from runtime buffers (aka Tensors) to Memref descriptors.
// -------------------------------------------------------------------------- //

Error VerifyMemrefOperand(mlir::MemRefType type, MemrefDesc memref) {
  if (memref.sizes.size() != type.getRank())
    return MakeStringError("operand rank does not match expected input rank: ",
                           memref.sizes.size(), " vs ", type.getRank());

  for (unsigned d = 0; d < memref.sizes.size(); ++d) {
    ssize_t operand_dim = memref.sizes[d];
    ssize_t memref_dim = type.getDimSize(d);
    if (operand_dim != memref_dim && !type.isDynamicDim(d))
      return MakeStringError("operand dimension #", d,
                             " does not match expected input dimension: ",
                             operand_dim, " vs ", memref_dim);
  }

  // TODO(ezhulenev): Verify memref element type.
  return Error::success();
}

Expected<MemrefDesc> ConvertTensorToMemrefDesc(mlir::MemRefType type,
                                               const Tensor& tensor) {
  if (auto* dht = dyn_cast<DenseHostTensor>(&tensor)) {
    MemrefDesc memref;
    memref.data = const_cast<void*>(dht->data());
    memref.offset = 0;
    dht->shape().GetDimensions(&memref.sizes);
    dht->shape().GetStrides(&memref.strides);
    if (auto err = VerifyMemrefOperand(type, memref)) return std::move(err);
    return memref;
  }

  return MakeStringError("unsupported tensor type: ", tensor.tensor_type());
}

// -------------------------------------------------------------------------- //
// Helper functions to pack compiled kernel operands.
// -------------------------------------------------------------------------- //

// Unpack `memref` argument into pointers to the data to be compatible with
// compiled MLIR function ABI.
static void AddMemrefArgument(const MemrefDesc& memref,
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
// arguments are alive when we call compiled kernel. Also allocates storage
// for returned values, which are passed to the compiled kernel as return value
// arguments.
//
// Returns compiled function arguments as `void*` type erased pointers. These
// pointers are pointing to the arguments that are stored in the CallFrame
// instance. See mlir::ExecutionEngine `packFunctionArguments` for the details.
static Expected<llvm::SmallVector<void*, 32>> InitializeCallFrame(
    mlir::FunctionType signature, ArrayRef<MemrefDesc> operands,
    CallFrame* call_frame) {
  llvm::SmallVector<void*, 32> args;

  // Make sure that we call the kernel with the correct number of operands.
  if (operands.size() != signature.getNumInputs())
    return MakeStringError(
        "number of operands must match the number of inputs: ", operands.size(),
        " vs ", signature.getNumInputs());

  // Pack all Memref operands as pointers to the call frame arguments.
  for (const MemrefDesc& desc : operands) {
    call_frame->memref_args.emplace_back(desc);
    AddMemrefArgument(call_frame->memref_args.back(), &args);
  }

  // Prepare storage to keep all the returned values.
  call_frame->async_rets.resize(signature.getNumResults());
  for (int i = 0; i < signature.getNumResults(); ++i)
    args.push_back(&call_frame->async_rets[i]);

  return args;
}

// -------------------------------------------------------------------------- //
// CompilationResult return values unpacking.
// -------------------------------------------------------------------------- //

// Extracts ranked memref from the async value storage, and emplaces it as a
// DenseHostTensor into the `dst` async value.
//
// TODO(ezhulenev): Currently this emplacer transfers ownership of the memref
// to the DenseHostTensor. This is not correct in general, because memref
// does not imply ownership, for example it can be one of the forwarded inputs
// or a global memref that is owned by the compiled kernel.
struct EmplaceDenseHostTensor {
  using ResultType = DenseHostTensor;

  template <typename T, int rank>
  static void Emplace(void* storage, AsyncValue* dst) {
    auto* memref = static_cast<StridedMemRefType<T, rank>*>(storage);

    TensorMetadata metadata(GetDType<T>(), memref->sizes);
    dst->emplace<DenseHostTensor>(
        metadata, HostBuffer::CreateFromExternal(
                      memref->data, metadata.GetHostSizeInBytes(),
                      [ptr = memref->basePtr](void*, size_t) { free(ptr); }));
  }
};

mlir::LogicalResult ReturnChain(RemainingResults results, unsigned result_index,
                                mlir::Type type, void* ret) {
  if (!type.isa<mlir::async::TokenType>()) return mlir::failure();

  auto* token = static_cast<mlir::runtime::AsyncToken*>(ret);
  results[result_index] = ConvertAsyncTokenToChain(token);
  return mlir::success();
}

mlir::LogicalResult ReturnDenseHostTensor(RemainingResults results,
                                          unsigned result_index,
                                          mlir::Type type, void* ret) {
  return ReturnStridedMemRef<EmplaceDenseHostTensor>(results, result_index,
                                                     type, ret);
}

ReturnValueConverter::ReturnValueConverter(RemainingResults results)
    : results_(results) {
  AddConversion([](RemainingResults results, unsigned i, mlir::Type t, void*) {
    results.EmitErrorAt(i, StrCat("unsupported return type: ", t));
    return mlir::failure();
  });
}

mlir::LogicalResult ReturnValueConverter::ReturnValue(unsigned result_index,
                                                      mlir::Type type,
                                                      void* ret) const {
  for (auto& convert : llvm::reverse(conversion_callbacks_))
    if (mlir::succeeded(convert(results_, result_index, type, ret)))
      return mlir::success();
  return mlir::failure();
}

void ReturnValueConverter::EmitErrors(RCReference<ErrorAsyncValue>& error) {
  for (size_t i = 0; i < results_.size(); ++i) results_[i] = error.CopyRef();
}

// -------------------------------------------------------------------------- //
// Execute compiled function with kernel operands.
// -------------------------------------------------------------------------- //

void EmitErrors(RemainingResults results, Error error,
                const ExecutionContext& exec_ctx) {
  auto async_error = EmitErrorAsync(exec_ctx, std::move(error));
  for (int i = 0; i < results.size(); ++i) results[i] = async_error.CopyRef();
}

Error EmitErrors(ReturnValueConverter results, Error error,
                 const ExecutionContext& exec_ctx) {
  auto async_error = EmitErrorAsync(exec_ctx, StrCat(error));
  results.EmitErrors(async_error);
  return error;
}

// TODO(ezhulenev): Execute should override alloc/free function calls used by
// codegened kernels to allocate/deallocate memrefs at runtime to use the host
// context allocator.

Error CompilationResult::Execute(ArrayRef<MemrefDesc> operands,
                                 ReturnValueConverter results,
                                 const ExecutionContext& exec_ctx) const {
  // CallFrame can be allocated on the stack because compiled function will
  // unpack all the arguments it needs, and async regions will not access
  // the data after the initial function will return the result.
  CallFrame call_frame;

  // Compiled function takes arguments and results as `void**` type erased
  // pointer. See mlir::ExecutionEngine `packFunctionArguments` for the details.
  Expected<llvm::SmallVector<void*, 32>> args =
      InitializeCallFrame(signature_, operands, &call_frame);
  if (auto err = args.takeError())
    return EmitErrors(results, std::move(err), exec_ctx);

  // Set the AsyncRuntime context to be used by all async tasks.
  ResourceContext* res_ctx = exec_ctx.resource_context();
  AsyncRuntime* runtime = res_ctx->GetOrCreateResource<AsyncRuntime>(
      "cpurt.runtime", exec_ctx.host());
  SetAsyncRuntimeContext(runtime);

  // Call the compiled function.
  (*fptr_)(args->data());

  // Convert compiled function return values into results.
  auto ret_types = signature_.getResults();
  bool converted = llvm::all_of(llvm::enumerate(ret_types), [&](auto tuple) {
    unsigned i = tuple.index();
    mlir::Type type = tuple.value();
    void* ret = call_frame.async_rets[i];
    return mlir::succeeded(results.ReturnValue(i, type, ret));
  });

  if (!converted)
    return MakeStringError("failed to convert all returned values");
  else
    return Error::success();
}

mlir::FunctionType CompilationResult::signature() const { return signature_; }

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

namespace {
// Expands operations that could not be lowered to LLVM direcly.
struct ExpandOpsPass
    : public mlir::PassWrapper<ExpandOpsPass, mlir::FunctionPass> {
  void runOnFunction() override;
};
}  // namespace

void ExpandOpsPass::runOnFunction() {
  mlir::MLIRContext* ctx = &getContext();
  mlir::OwningRewritePatternList patterns;
  mlir::populateExpandTanhPattern(patterns, ctx);
  if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<ExpandOpsPass> CreateExpandOpsPass() {
  return std::make_unique<ExpandOpsPass>();
}

static void InitializeCompiler() {
  static const bool initialized = ([] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  })();
  (void)initialized;
}

static void SetupPassDebugging(mlir::MLIRContext* context,
                               mlir::PassManager& pm) {
  // Print IR after all passes.
  if (DebugCpurtCompile()) {
    context->disableMultithreading();
    pm.enableIRPrinting([](mlir::Pass*, mlir::Operation*) { return false; },
                        [](mlir::Pass*, mlir::Operation*) { return true; },
                        /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false, llvm::errs());
  }
}

// Runs the custom pipeline that lowers loaded module to dialects supported by
// the CPURT (Linalg on buffers).
static mlir::LogicalResult LowerToCpurt(mlir::ModuleOp module,
                                        const CompilationOptions& opts) {
  if (!opts.register_pass_pipeline) return mlir::success();

  mlir::PassManager pm(module.getContext());
  SetupPassDebugging(module.getContext(), pm);
  opts.register_pass_pipeline(pm);
  return pm.run(module);
}

// Runs the pipeline to lower kernel IR to LLVM dialect.
static mlir::LogicalResult LowerToLlvm(mlir::ModuleOp module,
                                       const CompilationOptions& opts) {
  mlir::PassManager pm(module.getContext());
  SetupPassDebugging(module.getContext(), pm);

  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // TODO(ezhulenev): Move this to a pipeline exposed upstream when it will
  // stabilize, e.g. `LinalgToAsyncRuntime`.

  // Convert all linalg operations to parallel loops, and then add async
  // operations to actually execute them in parallel using the async runtime.
  mlir::OpPassManager& fpm = pm.nest<mlir::FuncOp>();
  fpm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
  // TODO(ezhulenev): Currently async.execute region can call a function with
  // an async.await inside, and this leads to blocking await inside a thread
  // managed by the concurrent work queue.
  // fpm.addPass(mlir::createAsyncParallelForPass(opts.num_worker_threads));
  fpm.addPass(mlir::createAsyncRefCountingPass());
  fpm.addPass(mlir::createAsyncRefCountingOptimizationPass());
  fpm.addPass(mlir::createStdExpandOpsPass());
  fpm.addPass(CreateExpandOpsPass());

  // Lower from high level async operations to async runtime.
  pm.addPass(mlir::createAsyncToAsyncRuntimePass());

  // Lower everything down to LLVM dialect.
  mlir::LowerToLLVMOptions lower_to_llvm_opts;
  pm.addPass(mlir::createConvertAsyncToLLVMPass());
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createLowerToLLVMPass(lower_to_llvm_opts));

  return pm.run(module);
}

static Expected<mlir::FuncOp> ResolveEntrypointFunction(
    mlir::ModuleOp module, string_view entrypoint) {
  // Find the original entryupoint function.
  auto entry_func = module.lookupSymbol<mlir::FuncOp>(entrypoint);
  if (!entry_func) return MakeStringError("entrypoint function not found");

  // Maybe resolve the corert entrypoint function referenced by the original
  // entrypoint function.
  if (auto ref = entry_func->getAttrOfType<mlir::SymbolRefAttr>(
          "cpurt.corert.entrypoint")) {
    auto corert_func = module.lookupSymbol<mlir::FuncOp>(ref);
    if (!corert_func) return MakeStringError("entrypoint function not found");
    return corert_func;
  }

  return entry_func;
}

Expected<CompilationResult> CompileKernelMlirModule(
    string_view mlir_module, string_view entrypoint,
    const CompilationOptions& opts) {
  // Setup LLVM target for code generation.
  InitializeCompiler();

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(mlir_module, "cpurt.kernel"),
      llvm::SMLoc());

  // Register MLIR dialects supported by the compiled kernels.
  mlir::DialectRegistry registry;
  registry.insert<mlir::async::AsyncDialect, mlir::linalg::LinalgDialect,
                  mlir::scf::SCFDialect, mlir::StandardOpsDialect,
                  mlir::LLVM::LLVMDialect>();

  // Register additional dialects provided via compilation options.
  if (opts.register_dialects) opts.register_dialects(registry);

  auto context = std::make_unique<mlir::MLIRContext>(registry);

  // Collect all diagnostics emited while lowering kernel module to LLVM.
  std::string diagnostic_str;
  llvm::raw_string_ostream os(diagnostic_str);
  mlir::SourceMgrDiagnosticHandler handler(source_mgr, context.get(), os);

  auto error = [&](string_view message) -> llvm::Error {
    return MakeStringError(message, ":\n", diagnostic_str);
  };

  // Parse a kernel source code into the MLIR Module.
  mlir::OwningModuleRef module(
      mlir::parseSourceFile(source_mgr, context.get()));
  if (!module) return error("failed to parse kernel source");

  // Lower loaded module to dialects supported by the CPURT to LLVM pipeline.
  if (failed(LowerToCpurt(*module, opts)))
    return error("failed to lower module to CPURT dialects");

  // Verify entrypoint function signature.
  auto entry_func = ResolveEntrypointFunction(*module, entrypoint);
  if (auto err = entry_func.takeError()) return std::move(err);

  std::string entry_name = entry_func->getName().str();
  mlir::FunctionType entry_signature = entry_func->getType();
  if (auto err = CompilationResult::VerifyEntrypointSignature(entry_signature))
    return std::move(err);

  // Lower kernel IR from high level dialects to the MLIR LLVM Dialect.
  if (failed(LowerToLlvm(*module, opts)))
    return error("failed to lower module to LLVM");

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

  return CompilationResult(std::move(context), std::move(*engine),
                           entry_signature, entry_name);
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
