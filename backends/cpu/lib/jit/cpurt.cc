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
#include "mlir/Transforms/Passes.h"
#include "tfrt/cpu/jit/async_runtime.h"
#include "tfrt/cpu/jit/async_runtime_api.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/host_buffer.h"
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

static bool IsValidRetType(mlir::Type type, ExecutionMode mode) {
  // Both native and CoreRT execution modes support async memrefs.
  if (auto value = type.dyn_cast<mlir::async::ValueType>())
    return IsValidMemref(value.getValueType());

  // In native execution mode compiled kernel can also return async tokens to
  // signal kernel completion if it is bufferized.
  return mode == ExecutionMode::kNative && type.isa<mlir::async::TokenType>();
}

/*static*/ Error CompilationResult::VerifyEntrypointSignature(
    mlir::FunctionType signature, ExecutionMode mode) {
  // Arguments must be ranked memrefs of supported types.
  for (unsigned i = 0; i < signature.getNumInputs(); ++i)
    if (!IsValidMemref(signature.getInput(i)))
      return MakeStringError("input #", i, " must be a ranked memref type");

  // Return value must be async tokens or async values of valid memrefs.
  for (unsigned i = 0; i < signature.getNumResults(); ++i)
    if (!IsValidRetType(signature.getResult(i), mode))
      return MakeStringError("result #", i,
                             " must be an async value of ranked memref type or "
                             "async token (only in native execution mode)");

  return Error::success();
}

static Error VerifyTensorOperand(mlir::MemRefType memref_type,
                                 const DenseHostTensor* tensor) {
  if (tensor->shape().GetRank() != memref_type.getRank())
    return MakeStringError("operand rank does not match expected input rank: ",
                           tensor->shape().GetRank(), " vs ",
                           memref_type.getRank());

  for (unsigned d = 0; d < tensor->shape().GetRank(); ++d) {
    ssize_t tensor_dim = tensor->shape().GetDimensionSize(d);
    ssize_t memref_dim = memref_type.getDimSize(d);
    if (tensor_dim != memref_dim && !memref_type.isDynamicDim(d))
      return MakeStringError("operand dimension #", d,
                             " does not match expected input dimension: ",
                             tensor_dim, " vs ", memref_dim);
  }

  return Error::success();
}

// Converts Tensor to Memref argument for supported tensor types. Returns
// error otherwise.
static Expected<MemrefArg> ConvertTensorToMemrefArg(mlir::Type type,
                                                    const Tensor& tensor) {
  auto memref_type = type.cast<mlir::MemRefType>();

  if (auto* dht = dyn_cast<DenseHostTensor>(&tensor)) {
    if (auto err = VerifyTensorOperand(memref_type, dht)) return std::move(err);
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
// arguments are alive when we call compiled kernel. Also allocates storage
// for returned values, which are passed to the compiled kernel as return value
// arguments.
//
// Returns compiled function arguments as `void*` type erased pointers. These
// pointers are pointing to the arguments that are stored in the CallFrame
// instance. See mlir::ExecutionEngine `packFunctionArguments` for the details.
static Expected<llvm::SmallVector<void*, 32>> InitializeCallFrame(
    mlir::FunctionType signature, RepeatedArguments<Tensor> operands,
    CallFrame* call_frame) {
  llvm::SmallVector<void*, 32> args;

  // Make sure that we call the kernel with the correct number of operands.
  if (operands.size() != signature.getNumInputs())
    return MakeStringError(
        "number of operands must match the number of inputs: ", operands.size(),
        " vs ", signature.getNumInputs());

  // Pack all Tensor operands as memref arguments.
  for (unsigned i = 0; i < operands.size(); ++i) {
    Expected<MemrefArg> memref =
        ConvertTensorToMemrefArg(signature.getInput(i), operands[i]);
    if (auto err = memref.takeError()) return std::move(err);
    call_frame->memref_args.push_back(*memref);
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
template <typename T, int rank>
static void EmplaceDenseHostTensor(void* storage, AsyncValue* dst) {
  auto* memref = static_cast<StridedMemRefType<T, rank>*>(storage);

  TensorMetadata metadata(GetDType<T>(), memref->sizes);
  dst->emplace<DenseHostTensor>(
      metadata, HostBuffer::CreateFromExternal(
                    memref->data, metadata.GetHostSizeInBytes(),
                    [ptr = memref->basePtr](void*, size_t) { free(ptr); }));
}

static void ReturnChain(RemainingResults results, unsigned result_index,
                        void* ret) {
  auto* token = static_cast<mlir::runtime::AsyncToken*>(ret);
  results[result_index] = ConvertAsyncTokenToChain(token);
}

static void ReturnDenseHostTensor(RemainingResults results,
                                  unsigned result_index,
                                  mlir::async::ValueType value_type,
                                  void* ret) {
  auto* value = static_cast<mlir::runtime::AsyncValue*>(ret);
  auto& dht = results.AllocateAt<DenseHostTensor>(result_index);

  // We already verified that return value is an async value of memref.
  auto memref = value_type.getValueType().cast<mlir::MemRefType>();
  auto element_type = memref.getElementType();

  // Dispatch to the correct extract function based on rank.
  auto rank_dispatch = [&](auto type_tag) {
    using T = decltype(type_tag);
    int64_t rank = memref.getRank();

    if (rank == 1)
      ExtractAsyncValue(value, dht.get(), EmplaceDenseHostTensor<T, 1>);
    else if (rank == 2)
      ExtractAsyncValue(value, dht.get(), EmplaceDenseHostTensor<T, 2>);
    else if (rank == 3)
      ExtractAsyncValue(value, dht.get(), EmplaceDenseHostTensor<T, 3>);
    else if (rank == 4)
      ExtractAsyncValue(value, dht.get(), EmplaceDenseHostTensor<T, 4>);
    else if (rank == 5)
      ExtractAsyncValue(value, dht.get(), EmplaceDenseHostTensor<T, 5>);
    else
      // TODO(ezhulenev): Because ExtractAsyncValue takes a llvm::function_ref
      // we can't pass a runtime arguments to emplace functions via lambda
      // capture, because the value might become available asynchronously and
      // this will lead to use after free. Consider adding an std::function
      // alternative for ranks higher then 5? Lambdas with small captures should
      // be stack allocated anyway, however it is implementation defined.
      dht->SetError({"unsupported rank", ErrorCode::kInvalidArgument});
  };

  // Dispatch based on the memref element type.
  if (element_type.isF32())
    rank_dispatch(float{});
  else
    dht->SetError({"unsupported element type", ErrorCode::kInvalidArgument});
}

// -------------------------------------------------------------------------- //
// Execute compiled function with kernel operands.
// -------------------------------------------------------------------------- //

// TODO(ezhulenev): Execute should override alloc/free function calls used by
// codegened kernels to allocate/deallocate memrefs at runtime to use the host
// context allocator.

Error CompilationResult::Execute(RepeatedArguments<Tensor> operands,
                                 RemainingResults results,
                                 const ExecutionContext& exec_ctx) const {
  // CallFrame can be allocated on the stack because compiled function will
  // unpack all the arguments it needs, and async regions will not access
  // the data after the initial function will return the result.
  CallFrame call_frame;

  // Compiled function takes arguments and results as `void**` type erased
  // pointer. See mlir::ExecutionEngine `packFunctionArguments` for the details.
  Expected<llvm::SmallVector<void*, 32>> args =
      InitializeCallFrame(signature_, operands, &call_frame);
  if (auto err = args.takeError()) return err;

  // Set the AsyncRuntime context to be used by all async tasks.
  ResourceContext* res_ctx = exec_ctx.resource_context();
  AsyncRuntime* runtime = res_ctx->GetOrCreateResource<AsyncRuntime>(
      "cpurt.runtime", exec_ctx.host());
  SetAsyncRuntimeContext(runtime);

  // Call the compiled function.
  (*fptr_)(args->data());

  // Unpack compiled function return values into results.
  for (unsigned i = 0; i < signature_.getNumResults(); ++i) {
    mlir::Type type = signature_.getResult(i);

    if (type.isa<mlir::async::TokenType>())
      ReturnChain(results, i, call_frame.async_rets[i]);
    else if (auto value_type = type.dyn_cast<mlir::async::ValueType>())
      ReturnDenseHostTensor(results, i, value_type, call_frame.async_rets[i]);
    else
      return MakeStringError("unsupported return type: ", type);
  }

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
  auto context = std::make_unique<mlir::MLIRContext>();
  context->getDialectRegistry()
      .insert<mlir::async::AsyncDialect, mlir::linalg::LinalgDialect,
              mlir::scf::SCFDialect, mlir::StandardOpsDialect,
              mlir::LLVM::LLVMDialect>();

  // Parse a kernel source code into the MLIR Module.
  mlir::OwningModuleRef module(
      mlir::parseSourceFile(source_mgr, context.get()));
  if (!module) return MakeStringError("failed to parse kernel source");

  // Verify entrypoint function signature.
  auto entry_func = module->lookupSymbol<mlir::FuncOp>(entrypoint);
  if (!entry_func) return MakeStringError("entrypoint function not found");

  mlir::FunctionType entry_signature = entry_func.getType();
  if (auto err = CompilationResult::VerifyEntrypointSignature(
          entry_signature, opts.execution_mode))
    return std::move(err);

  // Lower kernel IR from high level dialects to the MLIR LLVM Dialect.
  if (failed(LowerToLlvm(context.get(), *module, opts)))
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

  return CompilationResult(std::move(context), std::move(*engine),
                           entry_signature, entrypoint);
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
