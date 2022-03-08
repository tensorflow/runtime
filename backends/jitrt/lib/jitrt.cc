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

//===- jitrt.cc - ---------------------------------------------------------===//
// Support library for implementing TFRT kernels that do JIT compilation using
// MLIR framework.
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/jitrt.h"

#include <sys/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_buffer.h"
#include "tfrt/jitrt/async_runtime.h"
#include "tfrt/jitrt/async_runtime_api.h"
#include "tfrt/jitrt/constraints.h"
#include "tfrt/jitrt/runtime.h"
#include "tfrt/jitrt/specialization.h"
#include "tfrt/jitrt/support.h"
#include "tfrt/jitrt/symbolic_shape.h"
#include "tfrt/jitrt/transforms/rt_passes.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor.h"

namespace tfrt {
namespace jitrt {

// PRE-C++17: Static constexpr class members are required to have a definition.
constexpr int64_t MemrefType::kDynamicSize;

// Enable IR printing during the kernel compilation pipeline execution.
static bool DebugJitrtCompile() {
#if defined(DEBUG_JITRT)
  return true;
#else
  return false;
#endif
}

//----------------------------------------------------------------------------//
// Register MLIR C Runner Utils symbols with JIT execution engine.
//----------------------------------------------------------------------------//

static llvm::orc::SymbolMap CRunnerUtilsSymbolMap(llvm::orc::MangleAndInterner);

//----------------------------------------------------------------------------//
// Types for the codegen<->runtime integration, see API implementation below.
//----------------------------------------------------------------------------//

namespace runtime {

// Runtime KernelContext encapsulates all the JitRT data that is required to
// implement codegen<->runtime API.
struct KernelContext {
  // Results memory layout is owned by the executable, and stays alive after
  // the kernel function execution completes.
  const Executable::ResultsMemoryLayout* results_memory_layout;

  // CallFrame life time bound to the kernel function execution and destroyed
  // immediately when the function returns. Only the kernel function itself
  // reads the arguments and writes to the function results storage.
  Executable::CallFrame* call_frame;

  // Tracks whether any of the outputs were set.
  bool has_set_outputs = false;
};

llvm::orc::SymbolMap RuntimeApiSymbolMap(llvm::orc::MangleAndInterner);

}  // namespace runtime

//----------------------------------------------------------------------------//
// Get compiled function results memory layout.
//----------------------------------------------------------------------------//

Expected<Executable::ResultsMemoryLayout> Executable::GetResultsMemoryLayout(
    const FunctionType& signature) {
  // Size of the memory block required for storing results, and offsets for
  // each function result.
  bool has_async_results = false;
  size_t results_size_bytes = 0;
  llvm::SmallVector<size_t> results_offsets_bytes;
  results_offsets_bytes.reserve(signature.num_results());

  // Allocate `size_bytes` block of memory to store the function result.
  auto allocate_result = [&](size_t size_bytes) {
    results_offsets_bytes.emplace_back(results_size_bytes);
    results_size_bytes += size_bytes;
  };

  // Verify all result types and record memory requirements.
  for (unsigned i = 0; i < signature.num_results(); ++i) {
    auto* type = signature.result(i);

    // Async tokens stored as void* pointers.
    if (llvm::isa<AsyncTokenType>(type)) {
      allocate_result(sizeof(void*));
      has_async_results = true;
      continue;
    }

    // Async values stored as void* pointers.
    if (llvm::isa<AsyncValueType>(type)) {
      allocate_result(sizeof(void*));
      has_async_results = true;
      continue;
    }

    // Memrefs are stored as StridedMemref<T, rank> type:
    //   basePtr, data, offset, sizes[rank], strides[rank]
    if (auto* memref = llvm::dyn_cast<MemrefType>(type)) {
      allocate_result(/*pointers*/ 2 * sizeof(void*) +
                      /*offset*/ sizeof(int64_t) +
                      /*sizes/strides*/ sizeof(int64_t) * 2 * memref->rank());
      continue;
    }

    return MakeStringError("unknown result #", i,
                           " type memory layout: ", *type);
  }

  return ResultsMemoryLayout{has_async_results, results_size_bytes,
                             std::move(results_offsets_bytes)};
}

// -------------------------------------------------------------------------- //
// Converting from runtime buffers (aka Tensors) to Memref descriptors.
// -------------------------------------------------------------------------- //

Expected<MemrefDesc> ConvertTensorToMemrefDesc(const Tensor& tensor) {
  if (auto* dht = dyn_cast<DenseHostTensor>(&tensor)) {
    MemrefDesc memref;
    memref.dtype = dht->dtype();
    memref.data = const_cast<void*>(dht->data());
    memref.offset = 0;
    dht->shape().GetDimensions(&memref.sizes);
    dht->shape().GetStrides(&memref.strides);
    return {std::move(memref)};
  }

  return MakeStringError("unsupported tensor type: ", tensor.tensor_type());
}

// -------------------------------------------------------------------------- //
// Executable CallFrame initialization.
// -------------------------------------------------------------------------- //

// Returns the number of call frame arguments required to pass the `memref` to
// the compiled kernel.
static size_t GetArgsCount(const MemrefDesc& memref) {
  // Memref layout: 2 pointers + offset + rank * (size + stride)
  return 3 + 2 * memref.sizes.size();
}

// Returns the number of call frame arguments required to pass all operands
// to the compiled kernel.
static size_t GetArgsCount(ArrayRef<MemrefDesc> operands) {
  size_t n = 0;
  for (const MemrefDesc& memref : operands) n += GetArgsCount(memref);
  return n;
}

// Unpack `memref` argument into pointers to the data to be compatible with
// compiled MLIR function ABI.
static void AddMemrefArgument(const MemrefDesc& memref, size_t* offset,
                              llvm::SmallVectorImpl<void*>* args) {
  assert(memref.sizes.size() == memref.strides.size());

  auto* storage = args->data() + *offset;
  auto add_arg = [&](const void* p) {
    *storage = const_cast<void*>(p);
    ++storage;
    ++*offset;
  };

  add_arg(&memref.data);  // memref.basePtr
  add_arg(&memref.data);  // memref.data
  add_arg(&memref.offset);
  for (const Index& size : memref.sizes) add_arg(&size);
  for (const Index& stride : memref.strides) add_arg(&stride);
}

Error Executable::InitializeCallFrame(ArrayRef<MemrefDesc> operands,
                                      CallFrame* call_frame) const {
  // TODO(ezhulenev): If executable is specialized for operands shapes then
  // there is no need to verify them once more here. However currently we rely
  // on a hash code to look up specializations, and this can lead to collisions.

  // Make sure that we call the kernel with the correct number of operands.
  // We subtract one operand from the signature because it corresponds to the
  // context that we prepend to the given operands.
  if (LLVM_UNLIKELY(operands.size() != runtime_signature_.num_operands() - 1))
    return MakeStringError(
        "number of operands doesn't match the function signature: ",
        operands.size(), " vs ", runtime_signature_.num_operands() - 1);

  // Verify that all operands passed at runtime are compatible with compiled
  // function signature.
  auto kctx = dyn_cast<KernelContextOperandType>(runtime_signature_.operand(0));
  if (LLVM_UNLIKELY(!kctx)) {
    return MakeStringError(
        "expected KernelContext in first argument of "
        "signature, got: ",
        runtime_signature_.operand(0));
  }

  // We use 0-based index for operands, because the kernel context operand is an
  // internal implementation detail, and in case of an error users should get
  // back operand index corresponding to the user provided signature.
  for (unsigned i = 0; i < operands.size(); ++i) {
    unsigned idx = i + 1;  // use 1-based index to fetch runtime operand

    if (auto* memref = dyn_cast<MemrefType>(runtime_signature_.operand(idx))) {
      if (auto err = VerifyMemrefOperand(i, *memref, operands[i])) return err;
    } else {
      return MakeStringError("expected memref operand at #", i,
                             ", got: ", *runtime_signature_.operand(i));
    }
  }

  size_t n_args_elems = 1 + GetArgsCount(operands);
  call_frame->args.resize_for_overwrite(n_args_elems);

  // Add a placeholder for the kernel context as the first argument.
  call_frame->args[0] = nullptr;

  // Keep offset of the next argument in the `args` array, and update it every
  // time we pack a new argument.
  size_t offset = 1;

  // Pack all Memref operands as pointers to the call frame arguments.
  for (const MemrefDesc& desc : operands)
    AddMemrefArgument(desc, &offset, &call_frame->args);

  assert(offset == n_args_elems &&
         "reserved number of args must match the argument offset");

  // Allocate storage for results.
  call_frame->results.resize_for_overwrite(results_memory_layout_.size);

  // Mark results memory initialized to supress potential msan errors.
  TFRT_MSAN_MEMORY_IS_INITIALIZED(call_frame->results.data(),
                                  call_frame->results.size());

  return Error::success();
}

// -------------------------------------------------------------------------- //
// Executable return values unpacking.
// -------------------------------------------------------------------------- //

void ReturnValueConverterBase::ReturnErrors(
    RCReference<ErrorAsyncValue> error) const {
  results_[0] = std::move(error);
  for (size_t i = 1; i < results_.size(); ++i) results_[i] = results_[0];
}

namespace {
// Do not record any operands information for results conversion.
struct ConversionCtx {};

template <typename T, int rank>
static ArrayRef<int64_t> Sizes(StridedMemRefType<T, rank>* memref) {
  return llvm::makeArrayRef(memref->sizes);
}

template <typename T>
static ArrayRef<int64_t> Sizes(StridedMemRefType<T, 0>* memref) {
  return {};
}

// The returned memref can point into statically allocated memory that we can't
// pass to `free` (memref.global). The LLVM lowering of `memref.global` sets the
// allocated pointer to the magic value 0xDEADBEEF.
template <typename T, int rank>
static bool IsStaticStorageDuration(StridedMemRefType<T, rank>* memref) {
  return reinterpret_cast<std::intptr_t>(memref->basePtr) == 0xDEADBEEF;
}

// Converts StridedMemref to the DenseHostTensor. This struct satisfies
// ReturnStridedMemref's concept (see jitrt.h).
//
// This converter always creates a new DenseHostTensor from the memref, and it
// must be used only when it is guaranteed that the compiled region can't
// return global constant memref or forward one of the operands.
struct ConvertDenseHostTensor {
  using ResultType = DenseHostTensor;
  using ConversionContext = ConversionCtx;

  template <typename T, int rank>
  static DenseHostTensor Convert(ConversionContext& ctx, void* memref_ptr) {
    auto* memref = static_cast<StridedMemRefType<T, rank>*>(memref_ptr);
    TFRT_MSAN_MEMORY_IS_INITIALIZED(memref, sizeof(StridedMemRefType<T, rank>));
    TensorMetadata metadata(GetDType<T>(), Sizes(memref));
    TFRT_MSAN_MEMORY_IS_INITIALIZED(memref->data,
                                    metadata.GetHostSizeInBytes());

    // Deallocate memref only if it has dynamic storage duration.
    void* ptr = IsStaticStorageDuration(memref) ? nullptr : memref->basePtr;
    HostBuffer::Deallocator deallocator = [ptr](void*, size_t) { free(ptr); };

    return DenseHostTensor(
        metadata, HostBuffer::CreateFromExternal(memref->data,
                                                 metadata.GetHostSizeInBytes(),
                                                 std::move(deallocator)));
  }
};
}  // namespace

namespace internal {

mlir::LogicalResult ReturnAsyncToken(RemainingResults results,
                                     unsigned result_index, const Type* type,
                                     const Type* runtime_type,
                                     void* result_ptr) {
  if (!isa<AsyncTokenType>(type)) return mlir::failure();

  // Load the pointer to the async token from a pointer to result storage.
  TFRT_MSAN_MEMORY_IS_INITIALIZED(result_ptr, sizeof(void*));
  void* ret = *reinterpret_cast<void**>(result_ptr);
  auto* token = static_cast<mlir::runtime::AsyncToken*>(ret);
  results[result_index] = ConvertAsyncTokenToChain(token);
  return mlir::success();
}

mlir::LogicalResult ReturnAsyncMemrefAsDenseHostTensor(RemainingResults results,
                                                       unsigned result_index,
                                                       const Type* type,
                                                       const Type* runtime_type,
                                                       void* result_ptr) {
  ConversionCtx ctx;
  return ReturnAsyncStridedMemref<ConvertDenseHostTensor>(
      ctx, results, result_index, type, runtime_type, result_ptr);
}

mlir::LogicalResult ReturnMemrefAsDenseHostTensor(RemainingResults results,
                                                  unsigned result_index,
                                                  const Type* type,
                                                  const Type* runtime_type,
                                                  void* result_ptr) {
  ConversionCtx ctx;
  return ReturnStridedMemref<ConvertDenseHostTensor>(
      ctx, results, result_index, type, runtime_type, result_ptr);
}

}  // namespace internal

// -------------------------------------------------------------------------- //
// Execute compiled function with kernel operands.
// -------------------------------------------------------------------------- //

void ReturnErrors(RemainingResults results, Error error) {
  auto async_error = MakeErrorAsyncValueRef(StrCat(error));
  for (int i = 0; i < results.size(); ++i) results[i] = async_error;
}

void ReturnErrors(RemainingResults results, DecodedDiagnostic error) {
  return ReturnErrors(results, MakeStringError(error));
}

Error ReturnErrors(const ReturnValueConverterBase& results, Error error) {
  auto async_error = MakeErrorAsyncValueRef(StrCat(error));
  results.ReturnErrors(async_error);
  return error;
}

// TODO(ezhulenev): Execute should override alloc/free function calls used by
// codegened kernels to allocate/deallocate memrefs at runtime to use the host
// context allocator.

Error Executable::Execute(ArrayRef<MemrefDesc> operands,
                          const ReturnValueConverterBase& results,
                          const ExecuteOpts& opts) const {
  // CallFrame can be allocated on the stack because compiled function will
  // unpack all the arguments it needs, and async regions will not access
  // the data after the initial function will return the result.
  CallFrame call_frame;

  // Touch every byte of the memref arguments, to trigger memory sanitizer error
  // if some of the memrefs are already deallocated. Unfortunatelly sanitizers
  // do not work inside the JIT compiled code, and compiled kernels still can do
  // out of bounds memory access, however this sanity check allows to catch
  // obvious errors earlier.
#if defined(MEMORY_SANITIZER)
  auto do_not_optimize = [&](const auto& value) -> void {
    asm volatile("" : : "r,m"(value) : "memory");
  };

  for (const MemrefDesc& memref : operands) {
    Index size_in_bytes = GetHostSize(memref.dtype);
    for (Index size : memref.sizes) size_in_bytes *= size;

    uint8_t* data = static_cast<uint8_t*>(memref.data);
    for (Index i = 0; i < size_in_bytes; ++i) {
      uint8_t value = data[i];
      do_not_optimize(value);
    }
  }
#endif

  // Compiled function takes arguments and results as `void**` type erased
  // pointer. See mlir::ExecutionEngine `packFunctionArguments` for the details.
  if (auto err = InitializeCallFrame(operands, &call_frame))
    return ReturnErrors(results, std::move(err));

  Execute(call_frame, opts);

  // Convert compiled function return values into results.
  if (auto err = ReturnResults(results, &call_frame)) return err;

  return Error::success();
}

void Executable::Execute(CallFrame& call_frame, const ExecuteOpts& opts) const {
  // Set the AsyncRuntime to be used by all async tasks spawned by the compiled
  // kernel function.
  SetAsyncRuntime(AsyncRuntime(opts.async_task_runner));

  // Runtime kernel context can be used only by the entrypoint function (kernel
  // function) and can be safely allocated on the stack.
  runtime::KernelContext kernel_context;
  kernel_context.results_memory_layout = &results_memory_layout_;
  kernel_context.call_frame = &call_frame;

  // Override the kernel context argument.
  runtime::KernelContext* kernel_context_ptr = &kernel_context;
  assert(!call_frame.args.empty() && "call frame arguments must be non-empty");
  assert(call_frame.args[0] == nullptr && "expected to see a placeholder");
  call_frame.args[0] = &kernel_context_ptr;

  // Call the compiled function.
  (*fptr_)(call_frame.args.data());
}

Error Executable::ReturnResults(const ReturnValueConverterBase& results,
                                CallFrame* call_frame) const {
  // If execution failed, forward error to all results.
  if (call_frame->is_error) {
    results.ReturnErrors(MakeErrorAsyncValueRef(
        StrCat("compiled kernel run time error: ", call_frame->error)));
    return Error::success();
  }

  // Try to convert results using registered conversion functions.
  bool converted = true;

  for (unsigned i = 0; i < runtime_signature_.num_results(); ++i) {
    const Type* type = signature_.result(i);
    const Type* runtime_type = runtime_signature_.result(i);
    void* ret = &call_frame->results[results_memory_layout_.offsets[i]];
    bool res = mlir::succeeded(results.ReturnValue(i, type, runtime_type, ret));
    converted = converted && res;
  }

  if (LLVM_UNLIKELY(!converted))
    return MakeStringError("failed to convert all returned values");
  else
    return Error::success();
}

unsigned Executable::num_results() const {
  return runtime_signature_.num_results();
}

std::chrono::milliseconds Executable::time_to_compile() const {
  return time_to_compile_;
}

//----------------------------------------------------------------------------//
// Default calling convention for kernels compiled for JitRt.
//----------------------------------------------------------------------------//

using CallingConvention = CompilationOptions::CallingConvention;

/*static*/ CallingConvention CompilationOptions::DefaultCallingConvention() {
  return [](mlir::FunctionType func) {
    mlir::MLIRContext* ctx = func.getContext();

    llvm::SmallVector<mlir::Type> inputs = {KernelContextType::get(ctx)};
    inputs.reserve(1 + func.getNumInputs());
    llvm::append_range(inputs, func.getInputs());

    return mlir::FunctionType::get(ctx, inputs, func.getResults());
  };
}

/*static*/ CallingConvention CompilationOptions::DefaultCallingConvention(
    mlir::TypeConverter type_converter) {
  return [c = std::move(type_converter)](mlir::FunctionType func) mutable {
    mlir::MLIRContext* ctx = func.getContext();

    // Track if all type conversions were successful.
    bool failed_conversion = false;
    auto convert = [&](mlir::Type type) -> mlir::Type {
      auto converted = c.convertType(type);
      if (!converted) failed_conversion = true;
      return converted;
    };

    // Add kernel context as the first argument.
    llvm::SmallVector<mlir::Type> inputs = {KernelContextType::get(ctx)};
    inputs.reserve(1 + func.getNumInputs());
    llvm::transform(func.getInputs(), std::back_inserter(inputs), convert);

    // Apply type conversion to all results types.
    llvm::SmallVector<mlir::Type> results;
    results.reserve(func.getNumResults());
    llvm::transform(func.getResults(), std::back_inserter(results), convert);

    // Return null if any of the type conversions failed.
    if (failed_conversion) return mlir::FunctionType();

    return mlir::FunctionType::get(ctx, inputs, results);
  };
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

static void SetupPassDebugging(mlir::MLIRContext* context,
                               mlir::PassManager& pm) {
  // Print IR after all passes.
  if (DebugJitrtCompile()) {
    context->disableMultithreading();
    pm.enableIRPrinting([](mlir::Pass*, mlir::Operation*) { return false; },
                        [](mlir::Pass*, mlir::Operation*) { return true; },
                        /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false,
                        /*printAfterOnlyOnFailure=*/false, llvm::errs());
  }
}

static mlir::LogicalResult RunPipeline(
    mlir::ModuleOp module,
    const std::function<void(mlir::PassManager&)>& create_pipeline) {
  if (!create_pipeline) return mlir::success();

  mlir::PassManager pm(module.getContext());
  SetupPassDebugging(module.getContext(), pm);
  create_pipeline(pm);

  return pm.run(module);
}

// Runs the user-provided compilation pipeline to compile the module to LLVM.
static mlir::LogicalResult RunCompilationPipeline(
    mlir::ModuleOp module, const CompilationOptions& opts) {
  return RunPipeline(module, opts.create_compilation_pipeline);
}

// Runs the user-provided specialization pipeline.
static mlir::LogicalResult RunSpecializationPipeline(
    mlir::ModuleOp module, const CompilationOptions& opts) {
  return RunPipeline(module, opts.create_specialization_pipeline);
}

//----------------------------------------------------------------------------//
// JitCompilationContext to manage specialization and compilation.
//----------------------------------------------------------------------------//

using SymbolicShape = SymbolicShapesResolver::SymbolicShape;

namespace {
// JitCompilationContext manages parsing, specialization and compilation of a
// single compiled module. It owns the MLIR context where the module is created,
// and handlers to capture all diagnostics messages.
class JitCompilationContext {
 public:
  // Instantiates JIT compilation context from the serialized mlir source.
  static Expected<std::unique_ptr<JitCompilationContext>> Instantiate(
      CompilationOptions opts, string_view mlir_module, string_view entrypoint);

  // Makes an executable from the JIT compilation context. This is the end of
  // life for the compilation context, it effectively converts the MLIR module
  // to the executable (function pointer) using LLVM JIT code generation.
  // Optional specialization identifier specifies if the compiled executable is
  // a default one, or a specialization.
  static Expected<Executable> Compile(
      std::unique_ptr<JitCompilationContext>, string_view memory_region_name,
      Optional<size_t> specialization = llvm::None);

  template <typename OriginalError>
  llvm::Error Error(OriginalError original_error) {
    return MakeStringError(original_error, ":\n", diagnostic_);
  }

  llvm::StringRef name() const {
    return module().getName().getValueOr("<unknown>");
  }

  mlir::ModuleOp module() const {
    assert(module_ && "failed to parse the mlir module");
    return *module_;
  }

  mlir::FuncOp entrypoint() const {
    assert(entrypoint_ && "failed to resolve entrypoint function");
    return entrypoint_;
  }

  // Specialize compiled module to the operands:
  //
  // - update all unknown dimensions according to the resolved symbolic shapes
  // - attach symbolic shape attribute to the operands
  // - sink small constants into the function body
  //
  // After entrypoint signature is updated, and all constant operands
  // materialized in the function body, runs the user-provided specialization
  // pipeline to optimize the module based on the new information in the IR.
  //
  // Returns error if operands are not compatible with compiled module
  // entrypoint signature.
  llvm::Error Specialize(ArrayRef<MemrefDesc> operands,
                         ArrayRef<SymbolicShape> symbolic_shapes,
                         ArrayRef<OperandConstraint> constraints,
                         const SpecializationListener* listener);

  const CompilationOptions& options() const { return opts_; }

 private:
  JitCompilationContext(CompilationOptions opts, string_view mlir_module,
                        string_view entrypoint);

  CompilationOptions opts_;
  std::unique_ptr<mlir::MLIRContext> context_;
  std::string diagnostic_;
  llvm::raw_string_ostream diagnostic_os_;
  llvm::SourceMgr source_mgr_;
  mlir::SourceMgrDiagnosticHandler handler_;
  mlir::OwningOpRef<mlir::ModuleOp>
      module_;               // can be null if failed to parse the module
  mlir::FuncOp entrypoint_;  // can be null if failed to parse the module
  bool specialized_;
};
}  // namespace

// Creates a new MLIR Context and registers all the dialects that are expected
// in the compiled module.
static std::unique_ptr<mlir::MLIRContext> CreateMlirContext(
    const CompilationOptions& opts) {
  mlir::DialectRegistry registry;

  // Call user-provided callback to register all required dialects.
  if (opts.register_dialects) opts.register_dialects(registry);

  // TODO(ezhulenev): Wrap host context work queue into the llvm ThreadPool API
  // and pass it to all MLIR contexts.
  auto ctx = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
  ctx->loadAllAvailableDialects();
  return ctx;
}

JitCompilationContext::JitCompilationContext(CompilationOptions opts,
                                             string_view mlir_module,
                                             string_view entrypoint)
    : opts_(std::move(opts)),
      context_(CreateMlirContext(opts_)),
      diagnostic_os_(diagnostic_),
      handler_(source_mgr_, context_.get(), diagnostic_os_),
      specialized_(false) {
  source_mgr_.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(mlir_module, "jitrt.kernel"),
      llvm::SMLoc());
  module_ = mlir::parseSourceFile<mlir::ModuleOp>(source_mgr_, context_.get());
  if (module_) entrypoint_ = module_->lookupSymbol<mlir::FuncOp>(entrypoint);
}

/*static*/ Expected<std::unique_ptr<JitCompilationContext>>
JitCompilationContext::Instantiate(CompilationOptions opts,
                                   string_view mlir_module,
                                   string_view entrypoint) {
  std::unique_ptr<JitCompilationContext> context(
      new JitCompilationContext(std::move(opts), mlir_module, entrypoint));
  if (!context->module_)
    return context->Error("failed to parse the mlir source");
  if (!context->entrypoint_)
    return context->Error("failed to resolve entrypoint function");
  return {std::move(context)};
}

/*static*/ Expected<Executable> JitCompilationContext::Compile(
    std::unique_ptr<JitCompilationContext> ctx, string_view memory_region_name,
    Optional<size_t> specialization) {
  mlir::FuncOp entry_func = ctx->entrypoint();
  std::string entrypoint = entry_func.getName().str();

  // We track end-to-end time to compile the final executable.
  auto compilation_start = std::chrono::steady_clock::now();

  // Get the signature of the entrypoint function.
  auto signature = FunctionType::Convert(entry_func.getType());
  if (auto err = signature.takeError()) return std::move(err);

  // Get the calling convention for the entrypoint function.
  if (!ctx->options().calling_convention)
    return ctx->Error("calling convention is not defined");

  // Calling convention conversion can fail if some types are not supported.
  auto runtime_type = ctx->options().calling_convention(entry_func.getType());
  if (!runtime_type)
    return ctx->Error("calling convention failed to convert entrypoint type");

  // Get the runtime signature of the entrypoint function.
  auto runtime_signature = FunctionType::Convert(runtime_type);
  if (auto err = runtime_signature.takeError()) return std::move(err);

  // Get the memory layout for returning function results.
  auto results_memory_layout =
      Executable::GetResultsMemoryLayout(*runtime_signature);
  if (auto err = results_memory_layout.takeError()) return std::move(err);

  // Mark entrypoint function with a JitRt attribute, so it can be converted
  // to a kernel function (see `rt-to-kernel-function` pass).
  auto unit_attr = mlir::UnitAttr::get(entry_func.getContext());
  entry_func->setAttr(kJitRtEntrypointAttrName, unit_attr);

  // Run the compilation pipeline to lower the module to LLVM dialect.
  if (failed(RunCompilationPipeline(ctx->module(), ctx->options())))
    return ctx->Error("failed to run compilation pipeline");

  // Prepare JIT target machine for code generation.
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder) return builder.takeError();
  // Initialize asm parser and printer so that we can handle the inline assembly
  // generated in MLIR for some operations.
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  auto target_machine = builder->createTargetMachine();
  if (!target_machine) return target_machine.takeError();

  // Link with shared libraries for symbol resolution.
  llvm::SmallVector<llvm::StringRef, 4> libs;

  // Additional LLVM passes to run.
  llvm::SmallVector<const llvm::PassInfo*, 4> passes;
  auto transformer = mlir::makeLLVMPassesTransformer(passes, /*mbOptLevel=*/2,
                                                     target_machine->get());

  // Build MLIR execution engine.
  mlir::ExecutionEngineOptions engine_options;
  engine_options.transformer = transformer;
  engine_options.jitCodeGenOptLevel = ctx->options().jit_code_opt_level;
  engine_options.sharedLibPaths = libs;

  // Escape slashes, substituting them with double underscores.
  // The profiler's UI might interpret slashes as callchain separators,
  // whereas we want the region name to be shown in full.
  auto escape_region_name = [](llvm::StringRef str) -> std::string {
    llvm::SmallVector<llvm::StringRef> vec;
    for (llvm::StringRef sub : llvm::split(str, '/')) {
      vec.push_back(sub);
    }
    return llvm::join(vec, "__");
  };
  std::string mapper_name = llvm::formatv(
      "/jitrt{0}{1}:@{2}:{3}", memory_region_name.empty() ? "" : ":",
      escape_region_name(memory_region_name), entrypoint,
      specialization.hasValue() ? "specialized" : "default");

  std::unique_ptr<JitRtMemoryMapper> memory_mapper =
      JitRtMemoryMapper::Create(std::move(mapper_name));
  engine_options.sectionMemoryMapper = memory_mapper.get();
  auto engine = mlir::ExecutionEngine::create(ctx->module(), engine_options);
  if (auto err = engine.takeError()) return std::move(err);

  // Register MLIR C Runner API intrinsics (defined in CRunnerUtils).
  (*engine)->registerSymbols(CRunnerUtilsSymbolMap);
  // Register Async Runtime API intrinsics.
  (*engine)->registerSymbols(AsyncRuntimeApiSymbolMap);
  // Register Runtime API intrinsics (host runtime integration).
  (*engine)->registerSymbols(runtime::RuntimeApiSymbolMap);
  // Register memory allocation functions (malloc, free, ...).
  (*engine)->registerSymbols(AsyncRuntimeMemoryAllocationSymbolMap);

  // Trigger compilation by looking up the entrypoint function in the engine.
  Expected<Executable::KernelFunctionPtr> kernel_fn =
      (*engine)->lookupPacked(entrypoint);
  if (auto err = kernel_fn.takeError()) return std::move(err);

  // At this point compilation is completed, and all symbols in the LLVM module
  // materialized as addresses (entrypoint is an executable function pointer).
  auto time_to_compile = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - compilation_start);

  return Executable(
      ctx->name().str(), std::move(memory_mapper), std::move(*engine),
      *kernel_fn, std::move(*signature), std::move(*runtime_signature),
      std::move(*results_memory_layout), specialization, time_to_compile);
}

llvm::Error JitCompilationContext::Specialize(
    ArrayRef<MemrefDesc> operands, ArrayRef<SymbolicShape> symbolic_shapes,
    ArrayRef<OperandConstraint> constraints,
    const SpecializationListener* listener) {
  assert(!specialized_ && "can specialize executable only once");
  specialized_ = true;

  mlir::FuncOp func = entrypoint();

  // Update function signature and sink constant operands into the body.
  if (auto err = SpecializeFunction(func, operands, symbolic_shapes,
                                    constraints, listener)) {
    // No need to call this->Error() because we don't have diagnostic to report
    // in case of a failed specialization.
    return MakeStringError("failed to specialize: ", err);
  }

  // Run the user-provided specialization pipeline to take advantage of the
  // specialized operands and sunk constants.
  if (failed(RunSpecializationPipeline(*module_, opts_)))
    return Error("failed to run specialization pipeline");

  return Error::success();
}

//----------------------------------------------------------------------------//
// JitExecutable implementation.
//----------------------------------------------------------------------------//

using Specialization = CompilationOptions::Specialization;

static bool IsSpecializationOnly(ArrayRef<OperandConstraint> constraints) {
  return llvm::any_of(constraints, [](OperandConstraint constraint) {
    return constraint != OperandConstraint::kResolved;
  });
}

static bool HasValueConstraints(ArrayRef<OperandConstraint> constraints) {
  return llvm::any_of(constraints, [](OperandConstraint constraint) {
    return constraint == OperandConstraint::kValue;
  });
}

// Returns true if all function operands have statically known shape.
static bool HasStaticShapeOperands(const FunctionType& signature) {
  auto is_static = [](ArrayRef<Index> sizes) -> bool {
    return llvm::none_of(sizes, mlir::ShapedType::isDynamic);
  };

  auto is_shaped_static = [&](auto* type) -> Optional<bool> {
    if (auto* memref = dyn_cast<MemrefType>(type))
      return is_static(memref->sizes());

    if (auto* tensor = dyn_cast<RankedTensorType>(type))
      return is_static(tensor->sizes());

    return llvm::None;
  };

  for (unsigned i = 0; i < signature.num_operands(); ++i) {
    const Type* type = signature.operand(i);

    // Get the underlying value type from the async value.
    while (auto* value = dyn_cast<AsyncValueType>(type))
      type = &value->value_type();

    // Skip types that do not have shape.
    if (isa<AsyncTokenType, KernelContextOperandType>(type)) continue;

    // Unranked types do not have statically known shape.
    if (isa<UnrankedTensorType, UnrankedMemrefType>(type)) return false;

    // Check if the type is a shaped type with static sizes.
    if (Optional<bool> shaped_static = is_shaped_static(type)) {
      if (*shaped_static) continue;
      return false;
    }

    assert(false && "unsupported operand type");
    return false;
  }

  return true;
}

/*static*/ void JitExecutable::InlineCompilationTaskRunner(
    size_t num_specializations, ArrayRef<OperandConstraint> constraints,
    ArrayRef<MemrefDesc> operands, TaskFunction task, UserData user_data) {
  task();
}

/*static*/ Expected<JitExecutable> JitExecutable::Instantiate(
    string_view mlir_module, string_view entrypoint,
    CompilationOptions compilation_opts, string_view memory_region_name,
    CompilationTaskRunner runner) {
  // Set up LLVM target for code generation.
  InitializeCompiler();

  // Try to instantiate compilation context from the mlir source.
  Expected<std::unique_ptr<JitCompilationContext>> ctx =
      JitCompilationContext::Instantiate(compilation_opts, mlir_module,
                                         entrypoint);
  if (auto err = ctx.takeError()) return std::move(err);

  // Get resolved operands constraints for the entrypoint function.
  auto constraints = GetOperandsConstraints((*ctx)->entrypoint());
  if (auto err = constraints.takeError()) return std::move(err);

  // Get the entrypoint function signature, it will be later required to
  // compute the specialized function signature from the operands at runtime.
  auto signature = FunctionType::Convert((*ctx)->entrypoint().getType());
  if (auto err = signature.takeError()) return std::move(err);

  // If all of the operands have static shape, then we can always use default
  // binary for execution (unless specialization is explicitly required by the
  // operands constraints).
  if (HasStaticShapeOperands(*signature) && !IsSpecializationOnly(*constraints))
    compilation_opts.specialization = Specialization::kDisabled;

  // Return an error if specialization is explicitly disabled, yet some of
  // the operands have unresolved constraints.
  if (compilation_opts.specialization == Specialization::kDisabled &&
      IsSpecializationOnly(*constraints))
    return MakeStringError(
        "compilation options disabled specialization, yet operands have "
        "unresolved constraints: ",
        *constraints);

  // If the module must be specialized, return JitExecutable without a default
  // compiled executable.
  if (compilation_opts.specialization == Specialization::kAlways ||
      IsSpecializationOnly(*constraints))
    return JitExecutable(mlir_module, entrypoint, memory_region_name,
                         std::move(compilation_opts), std::move(*constraints),
                         std::move(*signature),
                         /*default_executable=*/llvm::None, std::move(runner));

  // Otherwise try to compile the default executable.
  Expected<Executable> executable =
      JitCompilationContext::Compile(std::move(*ctx), memory_region_name);
  if (auto err = executable.takeError()) return std::move(err);

  return JitExecutable(mlir_module, entrypoint, memory_region_name,
                       std::move(compilation_opts), std::move(*constraints),
                       std::move(*signature), std::move(*executable),
                       std::move(runner));
}

JitExecutable::JitExecutable(string_view mlir_module, string_view entrypoint,
                             string_view memory_region_name,
                             CompilationOptions compilation_opts,
                             ArrayRef<OperandConstraint> constraints,
                             FunctionType signature,
                             Optional<Executable> default_executable,
                             CompilationTaskRunner runner)
    : mlir_module_(mlir_module.str()),
      entrypoint_(entrypoint.str()),
      memory_region_name_(memory_region_name.str()),
      compilation_opts_(std::move(compilation_opts)),
      constraints_(constraints.begin(), constraints.end()),
      has_value_constraints_(HasValueConstraints(constraints_)),
      signature_(std::move(signature)),
      symbolic_shapes_resolver_(signature_, constraints_),
      has_default_executable_(default_executable.hasValue()),
      runner_(std::move(runner)),
      specializations_(std::make_unique<Specializations>()) {
  // Initialize default executable if it is available.
  if (has_default_executable_) {
    default_executable_ =
        MakeAvailableAsyncValueRef<Executable>(std::move(*default_executable));
  } else {
    default_executable_ =
        MakeErrorAsyncValueRef("default executable is not available");
  }
}

AsyncValuePtr<Executable> JitExecutable::DefaultExecutable() const {
  return default_executable_.AsPtr();
}

ArrayRef<OperandConstraint> JitExecutable::constraints() const {
  return constraints_;
}

// Combines `hash` with a hash value computed from a value constrained operands.
static llvm::hash_code CombineWithValueConstraineOperands(
    llvm::hash_code hash, ArrayRef<MemrefDesc> operands,
    ArrayRef<OperandConstraint> constraints) {
  for (int i = 0; i < constraints.size(); ++i) {
    if (LLVM_LIKELY(constraints[i] != OperandConstraint::kValue)) continue;

    const MemrefDesc& operand = operands[i];
    const auto* data = static_cast<uint8_t*>(operand.data);
    size_t rank = operand.sizes.size();
    assert(rank == 0 || rank == 1);
    size_t num_values = rank == 0 ? 1 : operand.sizes[0];
    Index len = num_values * GetHostSize(operand.dtype);
    hash = llvm::hash_combine(hash, llvm::hash_combine_range(data, data + len));
  }
  return hash;
}

// TODO(ezhulenev): The fast path should be free of mutex to find the
// pre-compiled specialization. Maybe use atomic pointers (multiple atomic
// pointers?) to keep the most commonly used specialization available without
// doing a lookup in the AsyncValuesCache.
//
// TODO(ezhulenev): The number of specializations should be bounded, ideally we
// should only keep N most common specializations, and for everything else
// fall back on the default executable. However what to do if default executable
// is not available, and the number of specializations is above N?
Expected<AsyncValuePtr<Executable>> JitExecutable::GetExecutable(
    ArrayRef<MemrefDesc> operands, UserData user_data,
    const SpecializationListener* listener) {
  // Do not try to compile specialized executable if it is explicitly disabled.
  if (compilation_opts_.specialization == Specialization::kDisabled)
    return DefaultExecutable();

  // Resolve symbolic shapes hash based on the static and runtime information.
  //
  // We rely on the hash code to find the specialized executable. In case of
  // a collision (practically impossible) incompatible operands will be rejected
  // by the executable operands verification.
  mlir::FailureOr<llvm::hash_code> hash =
      symbolic_shapes_resolver_.ResolveHash(operands);

  // If we failed to resolve the symbolic shapes hash, then we need to verify
  // all the operands to find the mismatch and report it to the user.
  if (LLVM_UNLIKELY(mlir::failed(hash))) {
    for (unsigned i = 0; i < operands.size(); ++i) {
      auto* type = signature_.operand(i);

      if (auto* memref = dyn_cast<MemrefType>(type)) {
        if (auto err = VerifyMemrefOperand(i, *memref, operands[i]))
          return std::move(err);

      } else if (auto* tensor = dyn_cast<RankedTensorType>(type)) {
        if (auto err = VerifyMemrefOperand(i, *tensor, operands[i]))
          return std::move(err);

      } else {
        return MakeStringError("expected memref operand at #", i,
                               ", got: ", *signature_.operand(i));
      }
    }

    assert(false && "failed to detect incorrect operand");
    return MakeStringError("failed to resolve symbolic shapes");
  }

  // Combine with a hash value computed from the value constrained operands.
  if (LLVM_UNLIKELY(has_value_constraints_))
    *hash = CombineWithValueConstraineOperands(*hash, operands, constraints_);

  // Maybe return Executable from the cache.
  if (auto cached = specializations_->Find(*hash)) {
    // Always use specialized kernel if required by the compilation options.
    if (compilation_opts_.specialization == Specialization::kAlways)
      return cached;

    // Fall back on default executable if the specialization is not yet
    // available.
    if (has_default_executable_ && !cached.IsAvailable())
      return DefaultExecutable();

    return cached;
  }

  // Instantiation from the source and specialization are cheap, so we do it in
  // the caller thread. We only use compilation runner for expensive part.

  // Try to instantiate compilation context from the mlir source.
  Expected<std::unique_ptr<JitCompilationContext>> ctx =
      JitCompilationContext::Instantiate(compilation_opts_, mlir_module_,
                                         entrypoint_);

  if (auto err = ctx.takeError()) {
    assert(false && "parsing mlir module must always succeed at this point");
    return std::move(err);
  }

  // Specialize executable to the concrete operands.
  mlir::FailureOr<llvm::SmallVector<SymbolicShape>> symbolic_shapes =
      symbolic_shapes_resolver_.Resolve(operands);
  if (auto err = (*ctx)->Specialize(operands, *symbolic_shapes, constraints_,
                                    listener)) {
    return MakeStringError("failed to specialize executable: ", err);
  }

  // Allocate a placeholder for the compiled specialization only after we are
  // ready to dispatch the compilation task.
  Specializations::Entry entry = specializations_->Allocate(*hash);

  // We lost the race; some other invocation will do the compilation.
  if (!entry.allocated) return entry.ptr;

  // Get the specialization id from the size of the specializations cache.
  size_t specialization = entry.size - 1;

  // Construct the task that will do the specialized executable compilation.
  auto compile = TaskFunction([ctx = std::move(*ctx), ref = entry.ptr.CopyRef(),
                               memory_region_name = memory_region_name_,
                               specialization]() mutable {
    Expected<Executable> executable = JitCompilationContext::Compile(
        std::move(ctx), memory_region_name, specialization);

    // Set the allocated entry async value state to error or concrete.
    if (auto err = executable.takeError()) {
      ref.SetError(std::move(err));
    } else {
      ref.emplace(std::move(*executable));
    }
  });

  // Offload specialization compilation to the user provided runner.
  runner_(specialization, constraints_, operands, std::move(compile),
          user_data);

  // Use the default executable while we are compiling a specialized version if
  // this is not explicitly disabled by the compilation options.
  if (compilation_opts_.specialization == Specialization::kAlways)
    return entry.ptr;
  else
    return has_default_executable_ ? DefaultExecutable() : entry.ptr;
}

AsyncValueRef<Chain> JitExecutable::AllExecutablesCompiled() const {
  return specializations_->AllAvailable();
}

//----------------------------------------------------------------------------//
// Register MLIR C Runner Utils symbols with JIT execution engine.
//----------------------------------------------------------------------------//

static llvm::orc::SymbolMap CRunnerUtilsSymbolMap(
    llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;

  auto bind = [&](llvm::StringRef name, auto symbol_ptr) {
    symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(symbol_ptr), llvm::JITSymbolFlags());
  };

  bind("memrefCopy", &memrefCopy);

  return symbol_map;
}

//----------------------------------------------------------------------------//
// Implement API for codegen <-> runtime integration defined in runtime header.
//----------------------------------------------------------------------------//

namespace runtime {

extern "C" void* runtimeGetResultStorage(KernelContext* ctx, int64_t index) {
  assert(ctx && "kernel context must be not null");
  assert(!ctx->call_frame->is_error && "error must not be set");
  size_t offset = ctx->results_memory_layout->offsets[index];
  assert(offset < ctx->call_frame->results.size() && "offset is out of bounds");
  ctx->has_set_outputs = true;
  return &ctx->call_frame->results[offset];
}

extern "C" void runtimeSetError(KernelContext* ctx, const char* error) {
  assert(ctx && "kernel context must be not null");
  assert(error && "runtime error must be not null");
  assert(!ctx->call_frame->is_error && "error must be set only once");
  assert(!ctx->has_set_outputs && "outputs must be undefined");
  ctx->call_frame->is_error = true;
  ctx->call_frame->error = {error};
}

llvm::orc::SymbolMap RuntimeApiSymbolMap(llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;

  auto bind = [&](llvm::StringRef name, auto symbol_ptr) {
    symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(symbol_ptr), llvm::JITSymbolFlags());
  };

  bind("runtimeGetResultStorage", &runtimeGetResultStorage);
  bind("runtimeSetError", &runtimeSetError);

  return symbol_map;
}

}  // namespace runtime

}  // namespace jitrt
}  // namespace tfrt
