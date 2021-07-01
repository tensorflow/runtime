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
// Support library for implementing TFRT kernels that do JIT compilation using
// MLIR framework.
//===----------------------------------------------------------------------===//

#include "tfrt/cpu/jit/cpurt.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ArmSVE/ArmSVEToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tfrt/cpu/jit/async_runtime.h"
#include "tfrt/cpu/jit/async_runtime_api.h"
#include "tfrt/cpu/jit/cpurt_support.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/host_buffer.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/mutex.h"
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

using CallFrame = Executable::CallFrame;
using ResultsMemoryLayout = Executable::ResultsMemoryLayout;

raw_ostream& operator<<(raw_ostream& os, const MemrefDesc& desc) {
  auto print_arr = [&](string_view name, ArrayRef<ssize_t> arr) {
    os << " " << name << ": [";
    if (!arr.empty()) {
      os << arr[0];
      for (int i = 1; i < arr.size(); ++i) os << ", " << arr[i];
    }
    os << "]";
  };

  os << "MemrefDesc: offset: " << desc.offset;
  print_arr("sizes", desc.sizes);
  print_arr("strides", desc.strides);

  return os;
}

//----------------------------------------------------------------------------//
// Verify compiled function signature and pre-compute memory layout for results.
//----------------------------------------------------------------------------//

// TODO(ezhulenev): Add support UnrankedMemrefType arguments and results.
static bool IsValidMemref(mlir::Type type) {
  auto memref = type.dyn_cast<mlir::MemRefType>();
  return static_cast<bool>(memref);
}

// Verifies that all function operands are supported at runtime.
static Error VerifyEntrypointOperands(mlir::FunctionType signature) {
  for (unsigned i = 0; i < signature.getNumInputs(); ++i)
    if (!IsValidMemref(signature.getInput(i)))
      return MakeStringError("input #", i, " must be a ranked memref type");

  return Error::success();
}

Expected<ResultsMemoryLayout> Executable::VerifyEntrypointSignature(
    mlir::FunctionType signature) {
  // Check if function operands are compatible with code generation.
  if (auto err = VerifyEntrypointOperands(signature)) return std::move(err);

  // Size of the memory block required for storing results, and offsets for
  // each function result.
  bool has_async_results = false;
  size_t results_size_bytes = 0;
  llvm::SmallVector<size_t> results_offsets_bytes;
  results_offsets_bytes.reserve(signature.getNumResults());

  // Allocate `size_bytes` block of memory to store the function result.
  auto allocate_result = [&](size_t size_bytes) {
    results_offsets_bytes.emplace_back(results_size_bytes);
    results_size_bytes += size_bytes;
  };

  // Verify all result types and record memory requirements.
  for (unsigned i = 0; i < signature.getNumResults(); ++i) {
    mlir::Type type = signature.getResult(i);

    // Async tokens stored as void* pointers.
    if (type.isa<mlir::async::TokenType>()) {
      allocate_result(sizeof(void*));
      has_async_results = true;
      continue;
    }

    // Async values stored as void* pointers.
    if (auto value = type.dyn_cast<mlir::async::ValueType>()) {
      if (!IsValidMemref(value.getValueType()))
        return MakeStringError(
            "result #", i, " async value payload type must be a valid memref");

      allocate_result(sizeof(void*));
      has_async_results = true;
      continue;
    }

    // Memrefs are stored as StridedMemref<T, rank> type:
    //   basePtr, data, offset, sizes[rank], strides[rank]
    if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
      if (!IsValidMemref(type))
        return MakeStringError("result #", i, " is not a valid memref");

      allocate_result(/*pointers*/ 2 * sizeof(void*) +
                      /*offset*/ sizeof(int64_t) +
                      /*sizes/strides*/ sizeof(int64_t) * 2 * memref.getRank());
      continue;
    }

    return MakeStringError("unsupported result type: ", type);
  }

  return ResultsMemoryLayout{has_async_results, results_size_bytes,
                             std::move(results_offsets_bytes)};
}

// -------------------------------------------------------------------------- //
// Converting from runtime buffers (aka Tensors) to Memref descriptors.
// -------------------------------------------------------------------------- //

static Error VerifyDType(mlir::Type input_type, DType operand_type) {
  assert(operand_type != DType::Invalid && "invalid operand type");

  auto verify = [&](DType expected_input_type) -> Error {
    if (operand_type != expected_input_type)
      return MakeStringError("operand type does not match input type: ",
                             operand_type, " vs ", DType(expected_input_type));
    return Error::success();
  };

  // TODO(ezhulenev): Implement type dispatching to connect MLIR with TFRT.
  if (input_type.isF32()) return verify(DType::F32);
  if (input_type.isInteger(1)) return verify(DType::I1);
  if (input_type.isInteger(32)) return verify(DType::I32);
  if (input_type.isInteger(64)) return verify(DType::I64);

  return MakeStringError("unexpected input type: ", input_type);
}

static Error VerifyMemrefOperand(mlir::ShapedType type,
                                 const MemrefDesc& memref) {
  // Check that memref data type matches operand element type.
  if (auto err = VerifyDType(type.getElementType(), memref.dtype)) return err;

  // For unranked operands we only verify the element type.
  if (!type.hasRank()) return Error::success();

  ArrayRef<int64_t> expected_shape = type.getShape();

  // Check that memref rank is the same as operand rank.
  if (memref.sizes.size() != expected_shape.size())
    return MakeStringError("operand rank does not match expected input rank: ",
                           memref.sizes.size(), " vs ", expected_shape.size());

  // Check that all statically known dimensions matches the memref dimensions.
  for (auto pair : llvm::enumerate(llvm::zip(memref.sizes, expected_shape))) {
    ssize_t operand_dim = std::get<0>(pair.value());
    ssize_t expected_dim = std::get<1>(pair.value());

    bool is_dynamic_dim = mlir::ShapedType::isDynamic(expected_dim);

    if (operand_dim != expected_dim && !is_dynamic_dim)
      return MakeStringError("operand dimension #", pair.index(),
                             " does not match expected input dimension: ",
                             operand_dim, " vs ", expected_dim);
  }

  return Error::success();
}

Error VerifyMemrefOperand(mlir::MemRefType type, const MemrefDesc& memref) {
  return VerifyMemrefOperand(type.cast<mlir::ShapedType>(), memref);
}

Error VerifyMemrefOperand(mlir::RankedTensorType type,
                          const MemrefDesc& memref) {
  return VerifyMemrefOperand(type.cast<mlir::ShapedType>(), memref);
}

Error VerifyMemrefOperand(mlir::UnrankedTensorType type,
                          const MemrefDesc& memref) {
  return VerifyMemrefOperand(type.cast<mlir::ShapedType>(), memref);
}

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

// Gets (copies) the values from `desc`, returning them in a DenseElementsAttr.
// If it cannot extract the values, returns an empty attribute.
static mlir::DenseElementsAttr GetMemrefValues(mlir::Builder* builder,
                                               mlir::ShapedType shaped_type,
                                               const MemrefDesc& desc) {
  size_t rank = desc.sizes.size();
  if (rank != 0 && rank != 1) return {};

  llvm::SmallVector<mlir::Attribute> attributes;
  size_t num_values = rank == 0 ? 1 : desc.sizes[0];
  switch (desc.dtype) {
    case DType::I32: {
      const auto* data = static_cast<TypeForDTypeKind<DType::I32>*>(desc.data);
      for (int i = 0; i < num_values; ++i) {
        attributes.push_back(builder->getI32IntegerAttr(data[i]));
      }
    } break;
    case DType::I64: {
      const auto* data = static_cast<TypeForDTypeKind<DType::I64>*>(desc.data);
      for (int i = 0; i < num_values; ++i) {
        attributes.push_back(builder->getI64IntegerAttr(data[i]));
      }
    } break;
    default:
      return {};
  }
  return mlir::DenseElementsAttr::get(shaped_type, attributes);
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
static void AddMemrefArgument(const MemrefDesc& memref,
                              llvm::SmallVectorImpl<void*>* args) {
  assert(memref.sizes.size() == memref.strides.size());

  size_t size = args->size();
  args->set_size(size + GetArgsCount(memref));

  auto* storage = &(*args)[size];
  auto add_arg = [&](const void* p) {
    *storage = const_cast<void*>(p);
    ++storage;
  };

  add_arg(&memref.data);  // memref.basePtr
  add_arg(&memref.data);  // memref.data
  add_arg(&memref.offset);
  for (const ssize_t& size : memref.sizes) add_arg(&size);
  for (const ssize_t& stride : memref.strides) add_arg(&stride);
}

Error Executable::InitializeCallFrame(ArrayRef<MemrefDesc> operands,
                                      CallFrame* call_frame) const {
  // TODO(ezhulenev): If executable is specialized for operands shapes then
  // there is no need to verify them once more here. However currently we rely
  // on a hash code to look up specializations, and this can lead to collisions.

  // Make sure that we call the kernel with the correct number of operands.
  if (operands.size() != signature_.getNumInputs())
    return MakeStringError(
        "number of operands must match the number of inputs: ", operands.size(),
        " vs ", signature_.getNumInputs());

  // Verify that all operands passed at runtime are compatible with compiled
  // function signature.
  for (int i = 0; i < operands.size(); ++i) {
    if (auto memref_ty = signature_.getInput(i).dyn_cast<mlir::MemRefType>()) {
      if (auto err = VerifyMemrefOperand(memref_ty, operands[i])) return err;
    } else {
      return MakeStringError("expected memref operand at #", i,
                             ", got: ", signature_.getInput(i));
    }
  }

  // Pack all Memref operands as pointers to the call frame arguments.
  call_frame->args.reserve(GetArgsCount(operands));
  for (const MemrefDesc& desc : operands)
    AddMemrefArgument(desc, &call_frame->args);

  // Allocate storage for results and add pointers to results into the `args`.
  call_frame->results.resize_for_overwrite(results_memory_layout_.size);
  for (auto offset : results_memory_layout_.offsets)
    call_frame->args.push_back(&call_frame->results[offset]);

  // Mark results memory initialized to supress potential msan errors.
  TFRT_MSAN_MEMORY_IS_INITIALIZED(call_frame->results.data(),
                                  call_frame->results.size());

  return Error::success();
}

// -------------------------------------------------------------------------- //
// Executable return values unpacking.
// -------------------------------------------------------------------------- //

ReturnValueConverterBase::ReturnValueConverterBase(RemainingResults results)
    : results_(results) {}

ReturnValueConverterBase::~ReturnValueConverterBase() {}

void ReturnValueConverterBase::EmitErrors(
    RCReference<ErrorAsyncValue>& error) const {
  for (size_t i = 0; i < results_.size(); ++i) results_[i] = error.CopyRef();
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
// ReturnStridedMemref's concept (see cpurt.h).
//
// This converter always creates a new DenseHostTensor from the memref, and it
// must be used only when it is guaranteed that the compiled region can't
// return global constant memref or forward one of the operands.
struct ConvertDenseHostTensor {
  using ResultType = DenseHostTensor;
  using ConversionContext = ConversionCtx;

  template <typename T, int rank>
  static DenseHostTensor Convert(const ConversionContext& ctx,
                                 void* memref_ptr) {
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
                                     unsigned result_index, mlir::Type type,
                                     void* result_ptr) {
  if (!type.isa<mlir::async::TokenType>()) return mlir::failure();

  // Load the pointer to the async token from a pointer to result storage.
  TFRT_MSAN_MEMORY_IS_INITIALIZED(result_ptr, sizeof(void*));
  void* ret = *reinterpret_cast<void**>(result_ptr);
  auto* token = static_cast<mlir::runtime::AsyncToken*>(ret);
  results[result_index] = ConvertAsyncTokenToChain(token);
  return mlir::success();
}

mlir::LogicalResult ReturnAsyncMemrefAsDenseHostTensor(RemainingResults results,
                                                       unsigned result_index,
                                                       mlir::Type type,
                                                       void* result_ptr) {
  return ReturnAsyncStridedMemref<ConvertDenseHostTensor>(
      {}, results, result_index, type, result_ptr);
}

mlir::LogicalResult ReturnMemrefAsDenseHostTensor(RemainingResults results,
                                                  unsigned result_index,
                                                  mlir::Type type,
                                                  void* result_ptr) {
  return ReturnStridedMemref<ConvertDenseHostTensor>({}, results, result_index,
                                                     type, result_ptr);
}

}  // namespace internal

// -------------------------------------------------------------------------- //
// Execute compiled function with kernel operands.
// -------------------------------------------------------------------------- //

void EmitErrors(RemainingResults results, Error error,
                const ExecutionContext& exec_ctx) {
  auto async_error = EmitErrorAsync(exec_ctx, std::move(error));
  for (int i = 0; i < results.size(); ++i) results[i] = async_error.CopyRef();
}

Error EmitErrors(const ReturnValueConverterBase& results, Error error,
                 const ExecutionContext& exec_ctx) {
  auto async_error = EmitErrorAsync(exec_ctx, StrCat(error));
  results.EmitErrors(async_error);
  return error;
}

// TODO(ezhulenev): Execute should override alloc/free function calls used by
// codegened kernels to allocate/deallocate memrefs at runtime to use the host
// context allocator.

Error Executable::Execute(ArrayRef<MemrefDesc> operands,
                          const ReturnValueConverterBase& results,
                          const ExecutionContext& exec_ctx,
                          const ExecuteOpts& opts) const {
  // CallFrame can be allocated on the stack because compiled function will
  // unpack all the arguments it needs, and async regions will not access
  // the data after the initial function will return the result.
  CallFrame call_frame;

  // Compiled function takes arguments and results as `void**` type erased
  // pointer. See mlir::ExecutionEngine `packFunctionArguments` for the details.
  if (auto err = InitializeCallFrame(operands, &call_frame))
    return EmitErrors(results, std::move(err), exec_ctx);

  Execute(call_frame, exec_ctx, opts);

  // Convert compiled function return values into results.
  if (auto err = ReturnResults(results, &call_frame)) return err;

  return Error::success();
}

void Executable::Execute(CallFrame& call_frame,
                         const ExecutionContext& exec_ctx,
                         const ExecuteOpts& opts) const {
  // Set the AsyncRuntime to be used by all async tasks spawned by the compiled
  // kernel function.
  SetAsyncRuntime({exec_ctx.host(), opts.async_runtime_worker_threads});

  // Call the compiled function.
  (*fptr_)(call_frame.args.data());
}

Error Executable::ReturnResults(const ReturnValueConverterBase& results,
                                CallFrame* call_frame) const {
  auto ret_types = signature_.getResults();

  bool converted = llvm::all_of(llvm::enumerate(ret_types), [&](auto tuple) {
    unsigned i = tuple.index();
    mlir::Type type = tuple.value();
    void* ret = &call_frame->results[results_memory_layout_.offsets[i]];
    return mlir::succeeded(results.ReturnValue(i, type, ret));
  });

  if (!converted)
    return MakeStringError("failed to convert all returned values");
  else
    return Error::success();
}

mlir::FunctionType Executable::signature() const { return signature_; }

//----------------------------------------------------------------------------//
// Setup MLIR pass pipeline to lower to LLVM dialect, and use ORC JIT to codegen
// functions at runtime.
//----------------------------------------------------------------------------//

namespace {
// Expand math operations to fast polynomial approximations.
struct MathApproximationPass
    : public mlir::PassWrapper<MathApproximationPass, mlir::FunctionPass> {
  void runOnFunction() override;
};

// Add alignment attribute to all `alloc` operations.
struct AlignedAllocationsPass
    : public mlir::PassWrapper<AlignedAllocationsPass, mlir::FunctionPass> {
  explicit AlignedAllocationsPass(int64_t alignment) : alignment(alignment) {}
  void runOnFunction() override;
  int64_t alignment;
};
}  // namespace

void MathApproximationPass::runOnFunction() {
  mlir::OwningRewritePatternList patterns(&getContext());
  mlir::populateMathPolynomialApproximationPatterns(patterns);
  if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<MathApproximationPass> CreateMathApproximationPass() {
  return std::make_unique<MathApproximationPass>();
}

void AlignedAllocationsPass::runOnFunction() {
  assert(alignment >= 0 && "alignment must be larger or equal to 0");
  if (alignment == 0) return;

  auto i64 = mlir::IntegerType::get(&getContext(), 64);
  auto alignment_attr = mlir::IntegerAttr::get(i64, alignment);

  getFunction().walk([&](mlir::memref::AllocOp alloc) {
    // Add alignment attribute only if the alignment attribute is missing or the
    // current alignment is smaller.
    if (!alloc.alignment().hasValue() || *alloc.alignment() < alignment)
      alloc.alignmentAttr(alignment_attr);
  });
}

std::unique_ptr<AlignedAllocationsPass> CreateAlignedAllocationsPass(
    int64_t alignment) {
  return std::make_unique<AlignedAllocationsPass>(alignment);
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
                        /*printAfterOnlyOnChange=*/false,
                        /*printAfterOnlyOnFailure=*/false, llvm::errs());
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

  // Convert all linalg operations to parallel loops.
  pm.addNestedPass<mlir::FuncOp>(
      mlir::createConvertLinalgToParallelLoopsPass());

  // Convert scf.parallel operations into async work sharding loops.
  if (opts.num_worker_threads > 1) {
    pm.addPass(mlir::createAsyncParallelForPass(
        /*asyncDispatch=*/true, /*numWorkerThreads=*/opts.num_worker_threads,
        /*targetBlockSize=*/15000));

    // Run canonicalization after async-parallel-for pass to remove async
    // operations that are not needed for executing small and cheap loops.
    pm.addPass(mlir::createCanonicalizerPass());
  }

  // Lower from high level async operations to async runtime.
  pm.addPass(mlir::createAsyncToAsyncRuntimePass());

  // Add async.runtime reference counting operations.
  pm.addPass(mlir::createAsyncRuntimePolicyBasedRefCountingPass());

  {
    mlir::OpPassManager& fpm = pm.nest<mlir::FuncOp>();

    // Optimize math operations.
    fpm.addPass(mlir::createStdExpandOpsPass());
    fpm.addPass(CreateMathApproximationPass());

    // Add alignment attribute to all memref allocations.
    fpm.addPass(CreateAlignedAllocationsPass(opts.alignment));
  }

  // Lower everything down to LLVM dialect.
  pm.addPass(mlir::createConvertLinalgToLLVMPass());
  pm.addPass(mlir::createConvertAsyncToLLVMPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());

  mlir::LowerVectorToLLVMOptions vector_to_llvm_opts;
  pm.addPass(mlir::createConvertVectorToLLVMPass());

  mlir::LowerToLLVMOptions lower_to_llvm_opts(module.getContext());
  pm.addPass(mlir::createLowerToLLVMPass(lower_to_llvm_opts));

  return pm.run(module);
}

//----------------------------------------------------------------------------//
// JitCompilationContext to manage specialization and compilation.
//----------------------------------------------------------------------------//

namespace {
// JitCompilationContext manages parsing, specialization and compilation of a
// single compiled module. It owns the MLIR context where the module is created,
// and handlers to capture all diagnostics messages.
class JitCompilationContext {
 public:
  // Instantiates JIT compilation context from the serialized mlir source.
  static Expected<std::unique_ptr<JitCompilationContext>> Instantiate(
      const CompilationOptions& opts, string_view mlir_module);

  // Makes an executable from the JIT compilation context. This is the end of
  // life for the compilation context, it effectively converts the MLIR module
  // to the executable (function pointer) using LLVM JIT code generation.
  static Expected<Executable> Compile(
      std::unique_ptr<JitCompilationContext> ctx, string_view entrypoint);

  template <typename OriginalError>
  llvm::Error Error(OriginalError original_error) {
    return MakeStringError(original_error, ":\n", diagnostic_);
  }

  mlir::ModuleOp module() const {
    assert(module_ && "failed to parse the mlir module");
    return *module_;
  }

  // Specialize compiled module to the operands: update all unknown dimensions
  // with concrete values and sink small constants into the function body.
  // Returns error if operands are not compatible with compiled module
  // entrypoint signature.
  llvm::Error Specialize(ArrayRef<MemrefDesc> operands,
                         ArrayRef<OperandConstraint> constraints,
                         string_view entrypoint,
                         JitExecutable::Listener* listener);

  const CompilationOptions& options() const { return opts_; }

 private:
  JitCompilationContext(const CompilationOptions& opts,
                        string_view mlir_module);

  CompilationOptions opts_;
  std::unique_ptr<mlir::MLIRContext> context_;
  std::string diagnostic_;
  llvm::raw_string_ostream diagnostic_os_;
  llvm::SourceMgr source_mgr_;
  mlir::SourceMgrDiagnosticHandler handler_;
  mlir::OwningModuleRef module_;  // can be null if failed to parse the module
};
}  // namespace

// Creates a new MLIR Context and registers all the dialects that are expected
// in the compiled module.
static std::unique_ptr<mlir::MLIRContext> CreateMlirContext(
    const CompilationOptions& opts) {
  mlir::DialectRegistry registry;

  // Register MLIR dialects supported by the compiled kernels.
  registry.insert<mlir::AffineDialect, mlir::async::AsyncDialect,
                  mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::StandardOpsDialect,
                  mlir::math::MathDialect, mlir::vector::VectorDialect>();

  // Register MLIR dialects that can be translated to LLVM IR.
  mlir::registerArmNeonDialectTranslation(registry);
  mlir::registerAMXDialectTranslation(registry);
  mlir::registerArmSVEDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerX86VectorDialectTranslation(registry);

  // Register additional dialects provided via compilation options.
  if (opts.register_dialects) opts.register_dialects(registry);

  return std::make_unique<mlir::MLIRContext>(registry);
}

// TODO(b/182944250): `cpurt.corert.entrypoint` indirection must go away with a
// support of async function.
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

JitCompilationContext::JitCompilationContext(const CompilationOptions& opts,
                                             string_view mlir_module)
    : opts_(opts),
      context_(CreateMlirContext(opts_)),
      diagnostic_os_(diagnostic_),
      handler_(source_mgr_, context_.get(), diagnostic_os_) {
  source_mgr_.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(mlir_module, "cpurt.kernel"),
      llvm::SMLoc());
  module_ = mlir::parseSourceFile(source_mgr_, context_.get());

  // TODO(ezhulenev): Every instance of MLIRContext owns its own thread pool,
  // and it leads to OOM errors when there are many JIT compiled kernels. Remove
  // this when MLIR will switch to a central thread pool ownership.
  context_->disableMultithreading();
}

/*static*/ Expected<std::unique_ptr<JitCompilationContext>>
JitCompilationContext::Instantiate(const CompilationOptions& opts,
                                   string_view mlir_module) {
  std::unique_ptr<JitCompilationContext> context(
      new JitCompilationContext(opts, mlir_module));
  if (!context->module_)
    return context->Error("failed to parse the mlir source");
  return {std::move(context)};
}

/*static*/ Expected<Executable> JitCompilationContext::Compile(
    std::unique_ptr<JitCompilationContext> ctx, string_view entrypoint) {
  // Lower loaded module to dialects supported by the CPURT to LLVM pipeline.
  if (failed(LowerToCpurt(ctx->module(), ctx->options())))
    return ctx->Error("failed to lower module to CPURT dialects");

  // Verify entrypoint function signature.
  auto entry_func = ResolveEntrypointFunction(ctx->module(), entrypoint);
  if (auto err = entry_func.takeError()) return std::move(err);

  std::string entry_name = entry_func->getName().str();
  mlir::FunctionType entry_signature = entry_func->getType();
  auto results_memory_layout =
      Executable::VerifyEntrypointSignature(entry_signature);
  if (auto err = results_memory_layout.takeError()) return std::move(err);

  // Lower kernel IR from high level dialects to the MLIR LLVM Dialect.
  if (failed(LowerToLlvm(ctx->module(), ctx->options())))
    return ctx->Error("failed to lower module to LLVM");

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

  // Build MLIR execution engine.
  auto engine = mlir::ExecutionEngine::create(
      ctx->module(), /*llvmModuleBuilder=*/nullptr, transformer,
      ctx->options().jit_code_opt_level, libs);
  if (!engine) return ctx->Error(engine.takeError());

  // Register Async Runtime API intrinsics.
  (*engine)->registerSymbols(AsyncRuntimeApiSymbolMap);

  return Executable(std::move(ctx->context_), std::move(*engine),
                    entry_signature, entry_name,
                    std::move(*results_memory_layout));
}

// Return input `type` specialized to memref descriptor operand.
static llvm::Expected<mlir::Type> SpecializeType(mlir::Type type,
                                                 const MemrefDesc& operand) {
  if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
    if (auto err = VerifyMemrefOperand(memref, operand)) return std::move(err);
    return mlir::MemRefType::get(operand.sizes, memref.getElementType());
  }

  if (auto tensor = type.dyn_cast<mlir::RankedTensorType>()) {
    if (auto err = VerifyMemrefOperand(tensor, operand)) return std::move(err);
    return mlir::RankedTensorType::get(operand.sizes, tensor.getElementType());
  }

  if (auto tensor = type.dyn_cast<mlir::UnrankedTensorType>()) {
    if (auto err = VerifyMemrefOperand(tensor, operand)) return std::move(err);
    return mlir::RankedTensorType::get(operand.sizes, tensor.getElementType());
  }

  return MakeStringError("Unsupported input type: ", type);
}

llvm::Error JitCompilationContext::Specialize(
    ArrayRef<MemrefDesc> operands, ArrayRef<OperandConstraint> constraints,
    string_view entrypoint, JitExecutable::Listener* listener) {
  mlir::FuncOp func = module_->lookupSymbol<mlir::FuncOp>(entrypoint);
  if (!func) return MakeStringError("Entrypoint not found: ", entrypoint);

  unsigned num_inputs = func.getNumArguments();

  // Specialize all function inputs to the given operands.
  llvm::SmallVector<mlir::Type> specialized_inputs(num_inputs);
  for (unsigned i = 0; i < num_inputs; ++i) {
    auto specialized = SpecializeType(func.getType().getInput(i), operands[i]);
    if (auto err = specialized.takeError()) return err;
    specialized_inputs[i] = *specialized;
  }

  // Update function type to a new specialized one.
  auto specialized = mlir::FunctionType::get(
      func.getContext(), specialized_inputs, func.getType().getResults());
  func.setType(specialized);

  // Update function entry block arguments.
  mlir::Block& entry_block = func.getBlocks().front();

  // Forward original block arguments to arguments with specialized type.
  for (int i = 0; i < num_inputs; ++i) {
    mlir::BlockArgument arg = entry_block.addArgument(specialized_inputs[i]);
    entry_block.getArgument(i).replaceAllUsesWith(arg);
  }

  // Erase all the original block arguments.
  llvm::SmallVector<unsigned> erase_block_args(num_inputs);
  std::iota(erase_block_args.begin(), erase_block_args.end(), 0);
  entry_block.eraseArguments(erase_block_args);

  // Sink small constants into the function body.
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(&func.front());
  mlir::Location loc = func.getLoc();
  for (int i = 0; i < constraints.size(); ++i) {
    if (constraints[i] != OperandConstraint::kValue) continue;
    mlir::Type operand_type = func.getType().getInput(i);
    mlir::ShapedType shaped = operand_type.dyn_cast<mlir::ShapedType>();
    assert(SupportsValueSpecialization(shaped) &&
           "Non-sinkable operand was marked for sinking");
    mlir::DenseElementsAttr shape_attr =
        GetMemrefValues(&builder, shaped, operands[i]);
    if (!shape_attr)
      return MakeStringError("Cannot get values from operand type: ",
                             operand_type);
    mlir::Value cst = builder.create<mlir::ConstantOp>(loc, shaped, shape_attr);
    entry_block.getArgument(i).replaceAllUsesWith(cst);
    if (listener) listener->notifyValueSpecialized(i, operand_type, shape_attr);
  }

  if (listener) listener->notifyModuleSpecialized(specialized_inputs);
  return Error::success();
}

//----------------------------------------------------------------------------//
// JitExecutable implementation.
//----------------------------------------------------------------------------//

constexpr const char* const JitExecutable::kConstraint;

static raw_ostream& operator<<(raw_ostream& os,
                               const OperandConstraint& constraint) {
  auto str = [](OperandConstraint constraint) -> string_view {
    switch (constraint) {
      case OperandConstraint::kResolved:
        return "resolved";
      case OperandConstraint::kRank:
        return "rank";
      case OperandConstraint::kShape:
        return "shape";
      case OperandConstraint::kValue:
        return "value";
      default:
        llvm_unreachable("unknown operand constraint");
    }
  };

  os << str(constraint);
  return os;
}

static raw_ostream& operator<<(raw_ostream& os,
                               ArrayRef<OperandConstraint> constraints) {
  os << "[";
  llvm::interleaveComma(constraints, os);
  os << "]";
  return os;
}

// Returns kResolved if the constraint can be resolved at compile time.
// Returns kValue for value specialization if it can be resolved at run time.
// Returns an error when the constraint cannot be resolved.
Expected<OperandConstraint> ResolveOperandConstraint(
    OperandConstraint operand_constraint, mlir::Type operand_type) {
  // Operand must be a shaped type: memref or tensor.
  auto shaped = operand_type.dyn_cast<mlir::ShapedType>();
  if (!shaped)
    return MakeStringError("unsupported operand type: ", operand_type);

  // Resolve `rank` constraint if rank is known at compile time.
  if (operand_constraint == OperandConstraint::kRank && shaped.hasRank())
    return OperandConstraint::kResolved;

  // Resolve `shape` constraint if shape is known at compile time.
  if (operand_constraint == OperandConstraint::kShape &&
      shaped.hasStaticShape())
    return OperandConstraint::kResolved;

  // Leave the `value` constraint unmodified if the operand is sinkable.
  if (operand_constraint == OperandConstraint::kValue) {
    if (SupportsValueSpecialization(shaped)) return operand_constraint;
    return MakeStringError("Cannot sink operand type: ", operand_type);
  }

  return operand_constraint;
}

static Expected<OperandConstraint> ParseOperandConstraints(string_view str) {
  if (str == "rank") return OperandConstraint::kRank;
  if (str == "shape") return OperandConstraint::kShape;
  if (str == "value") return OperandConstraint::kValue;
  return MakeStringError("unknown operand constraint: ", str);
}

// Returns operands constraints inferred from the entrypoint signature.
static Expected<llvm::SmallVector<OperandConstraint>> GetOperandsConstraints(
    mlir::ModuleOp module, string_view entrypoint) {
  llvm::SmallVector<OperandConstraint> constraints;

  auto func = ResolveEntrypointFunction(module, entrypoint);
  if (auto err = func.takeError()) return std::move(err);

  auto parse = [](mlir::Attribute attr) -> Expected<OperandConstraint> {
    // If attribute is not defined it means that there is no operand constraint.
    if (!attr) return OperandConstraint::kResolved;

    // Otherwise try to parse constraint from the string attribute.
    auto str = attr.dyn_cast_or_null<mlir::StringAttr>();
    if (!str)
      return MakeStringError("unexpected ", JitExecutable::kConstraint,
                             " attribute");
    return ParseOperandConstraints(str.getValue());
  };

  for (int i = 0; i < func->getNumArguments(); ++i) {
    auto operand_type = func->getType().getInput(i);

    auto constraint = parse(func->getArgAttr(i, JitExecutable::kConstraint));
    if (auto err = constraint.takeError()) return std::move(err);

    auto resolved = ResolveOperandConstraint(*constraint, operand_type);
    if (auto err = resolved.takeError()) return std::move(err);

    constraints.push_back(*resolved);
  }

  return constraints;
}

// Returns true if any of the operands have an unresolved constraint.
static bool IsSpecializationOnly(ArrayRef<OperandConstraint> constraints) {
  return llvm::any_of(constraints, [](OperandConstraint constraint) {
    return constraint != OperandConstraint::kResolved;
  });
}

/*static*/ Expected<JitExecutable> JitExecutable::Instantiate(
    string_view mlir_module, string_view entrypoint,
    const CompilationOptions& compilation_opts, Listener* listener) {
  // Set up LLVM target for code generation.
  InitializeCompiler();

  // Try to instantiate compilation context from the mlir source.
  Expected<std::unique_ptr<JitCompilationContext>> ctx =
      JitCompilationContext::Instantiate(compilation_opts, mlir_module);
  if (auto err = ctx.takeError()) return std::move(err);

  // Get resolved operands constraints for the entrypoint function.
  auto constraints = GetOperandsConstraints((*ctx)->module(), entrypoint);
  if (auto err = constraints.takeError()) return std::move(err);

  // If the module must be specialized, return JitExecutable without a default
  // compiled executable.
  if (IsSpecializationOnly(*constraints)) {
    // If specialization is explicitly disabled return an error, because we will
    // never be able to compile an executable.
    if (compilation_opts.disable_specializations)
      return MakeStringError(
          "compilation options disabled specialization, however operands have "
          "unresolved constraints: ",
          *constraints);

    return JitExecutable(mlir_module, entrypoint, compilation_opts,
                         *constraints, {}, listener);
  }

  // Otherwise try to compile the default executable.
  Expected<Executable> executable =
      JitCompilationContext::Compile(std::move(*ctx), entrypoint);
  if (auto err = executable.takeError()) return std::move(err);

  return JitExecutable(mlir_module, entrypoint, compilation_opts, *constraints,
                       std::move(*executable), listener);
}

JitExecutable::JitExecutable(string_view mlir_module, string_view entrypoint,
                             CompilationOptions compilation_opts,
                             ArrayRef<OperandConstraint> constraints,
                             Optional<Executable> default_executable,
                             Listener* listener)
    : mlir_module_(mlir_module.str()),
      entrypoint_(entrypoint.str()),
      compilation_opts_(std::move(compilation_opts)),
      constraints_(constraints.begin(), constraints.end()),
      default_executable_(std::move(default_executable)),
      specializations_(std::make_unique<Specializations>()),
      listener_(listener) {}

const Executable* JitExecutable::DefaultExecutable() const {
  return default_executable_.hasValue() ? &*default_executable_ : nullptr;
}

bool JitExecutable::HasDefaultExecutable() const {
  return default_executable_.hasValue();
}

ArrayRef<OperandConstraint> JitExecutable::constraints() const {
  return constraints_;
}

// Hashes the given operands.
// Note: due to value specialization, the resulting hash might depend on the
// values (and not only on the types) of the operands.
static llvm::hash_code HashOperands(ArrayRef<MemrefDesc> operands,
                                    ArrayRef<OperandConstraint> constraints) {
  llvm::hash_code hash = llvm::hash_value(operands);

  // Mix values of arguments to be sunk into the hash.
  for (int i = 0; i < constraints.size(); ++i) {
    if (constraints[i] != OperandConstraint::kValue) continue;
    const MemrefDesc& operand = operands[i];
    const auto* data = static_cast<uint8_t*>(operand.data);
    size_t rank = operand.sizes.size();
    assert(rank == 0 || rank == 1);
    size_t num_values = rank == 0 ? 1 : operand.sizes[0];
    ssize_t len = num_values * GetHostSize(operand.dtype);
    hash = llvm::hash_combine(hash, llvm::hash_combine_range(data, data + len));
  }
  return hash;
}

// Implement `hash_value` to rely on the ADL lookup for the MemrefDesc type.
static llvm::hash_code hash_value(const MemrefDesc& memref) {
  // We currently do not support non-contiguous memrefs as operands, so we do
  // not need to hash memref strides.
  return llvm::hash_combine(
      memref.sizes.size(),
      llvm::hash_combine_range(memref.sizes.begin(), memref.sizes.end()));
}

// TODO(ezhulenev): Current implementation unnecessarily blocks if the
// specialization is not available, however it can fallback on default
// executable if it is available. Also the fast path should be free of mutex to
// find the pre-compiled specialization. Maybe use atomic pointers (multiple
// atomic pointers?) to keep the most commonly used specialization available
// without grabbing a mutex and doing lookup in the DenseMap.
//
// TODO(ezhulenev): The number of specializations should be bounded, ideally we
// should only keep N most common specializations, and for everything else
// fall back on the default executable. However what to do if default executable
// is not available, and the number of specializations is above N?
//
// TODO(ezhulenev): Currently we always specialize operands to the shape, even
// if operand constraint only requires rank specialization. Although it might be
// beneficial to know the shape to do broadcasts fusion, consider not doing that
// when it is not needed.
Expected<const Executable*> JitExecutable::GetExecutable(
    ArrayRef<MemrefDesc> operands) {
  // Do not try to compile specialized executable if it is explicitly disabled.
  if (compilation_opts_.disable_specializations) {
    if (!HasDefaultExecutable())
      return MakeStringError(
          "jit executable specialization is disabled, but the default "
          "executable is not available");

    return &*default_executable_;
  }

  llvm::hash_code hash = HashOperands(operands, constraints_);

  // Convert ExecutableOrError to the function result.
  auto convert = [](ExecutableOrError& value) -> Expected<const Executable*> {
    // Only error or executable must be available.
    bool is_error = static_cast<bool>(value.error);
    assert(is_error != value.executable.hasValue());

    if (is_error)
      return MakeStringError("Compilation of specialized function failed: ",
                             StrCat(value.error));

    return value.executable.getPointer();
  };

  // Reduce the scope of the lock to ensure that compilation happens without
  // holding the lock.
  {
    tfrt::mutex_lock lock(specializations_->mu);
    auto it = specializations_->executables.find(hash);
    if (it != specializations_->executables.end())
      return convert(it->getSecond());
  }

  // Try to instantiate compilation context from the mlir source.
  Expected<std::unique_ptr<JitCompilationContext>> ctx =
      JitCompilationContext::Instantiate(compilation_opts_, mlir_module_);
  if (auto err = ctx.takeError()) {
    assert(false && "parsing mlir module must always succeed at this point");
    return std::move(err);
  }

  // Specialize executable to the concrete operands.
  if (auto err =
          (*ctx)->Specialize(operands, constraints_, entrypoint_, listener_))
    return MakeStringError("Failed to specialize executable: ", err);

  // Compile the specialized executable.
  Expected<Executable> executable =
      JitCompilationContext::Compile(std::move(*ctx), entrypoint_);

  // Update the specialized executables cache with an error or the value.
  tfrt::mutex_lock lock(specializations_->mu);

  // Concurrent thread updated the cache before us.
  auto it = specializations_->executables.find(hash);
  if (it != specializations_->executables.end())
    return convert(it->getSecond());

  // Update the cache with a compilation error.
  if (auto err = executable.takeError()) {
    auto emplaced =
        specializations_->executables.try_emplace(hash, std::move(err));
    assert(emplaced.second && "error must be successfully emplaced");
    return convert(emplaced.first->getSecond());
  }

  // Or update the cache with a compiled executable.
  auto emplaced =
      specializations_->executables.try_emplace(hash, std::move(*executable));
  assert(emplaced.second && "executable must be successfully emplaced");
  return convert(emplaced.first->getSecond());
}

//----------------------------------------------------------------------------//
// JitExecutableCache implementation.
//----------------------------------------------------------------------------//

AsyncValueRef<JitExecutable> JitExecutableCache::Find(intptr_t key) const {
  tfrt::mutex_lock lock(mu_);
  auto it = cache_.find(key);
  if (it != cache_.end()) return JitExecutableCache::MakeRef(it->getSecond());
  return AsyncValueRef<JitExecutable>();
}

AsyncValueRef<JitExecutable> JitExecutableCache::Insert(
    intptr_t key, JitExecutable jit_executable) {
  tfrt::mutex_lock lock(mu_);
  auto it = cache_.find(key);
  if (it != cache_.end()) return JitExecutableCache::MakeRef(it->getSecond());

  auto emplaced = cache_.try_emplace(
      key, std::make_unique<CachedExecutable>(std::move(jit_executable)));
  return JitExecutableCache::MakeRef(emplaced.first->getSecond());
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
