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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ArmSVE/ArmSVEToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_buffer.h"
#include "tfrt/jitrt/async_runtime.h"
#include "tfrt/jitrt/async_runtime_api.h"
#include "tfrt/jitrt/jitrt_pipeline.h"
#include "tfrt/jitrt/runtime.h"
#include "tfrt/jitrt/support.h"
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

raw_ostream& operator<<(raw_ostream& os, const MemrefDesc& desc) {
  auto print_arr = [&](string_view name, ArrayRef<Index> arr) {
    os << " " << name << ": [";
    if (!arr.empty()) {
      os << arr[0];
      for (int i = 1; i < arr.size(); ++i) os << ", " << arr[i];
    }
    os << "]";
  };

  os << "MemrefDesc: dtype: " << desc.dtype << " offset: " << desc.offset;
  print_arr("sizes", desc.sizes);
  print_arr("strides", desc.strides);

  return os;
}

raw_ostream& operator<<(raw_ostream& os, const Type& type) {
  auto print_arr = [&](ArrayRef<Index> arr) {
    if (!arr.empty()) {
      os << arr[0];
      for (int i = 1; i < arr.size(); ++i) os << "x" << arr[i];
    }
  };

  if (isa<AsyncTokenType>(&type)) {
    os << "!async.token";

  } else if (auto* value = dyn_cast<AsyncValueType>(&type)) {
    os << "!async.value<";
    os << value->value_type();
    os << ">";

  } else if (auto* tensor = dyn_cast<RankedTensorType>(&type)) {
    os << "tensor<";
    print_arr(tensor->sizes());
    os << "x" << tensor->element_type();
    os << ">";

  } else if (auto* tensor = dyn_cast<UnrankedTensorType>(&type)) {
    os << "tensor<";
    os << "*x" << tensor->element_type();
    os << ">";

  } else if (auto* memref = dyn_cast<MemrefType>(&type)) {
    os << "memref<";
    print_arr(memref->sizes());
    os << "x" << memref->element_type();
    os << ">";

  } else if (auto* memref = dyn_cast<UnrankedMemrefType>(&type)) {
    os << "memref<";
    os << "*x" << memref->element_type();
    os << ">";

  } else if (auto* kernel_context = dyn_cast<KernelContextOperandType>(&type)) {
    os << "!rt.kernel_context";

  } else {
    assert(false && "pretty printing is not implemented");
    os << "<unknown type>";
  }

  return os;
}

//----------------------------------------------------------------------------//
// Compiled function signature types conversion from the MLIR types.
//----------------------------------------------------------------------------//

AsyncTokenType::AsyncTokenType() : Type(TypeKind::kAsyncToken) {}

AsyncValueType::AsyncValueType(std::unique_ptr<Type> value_type)
    : Type(TypeKind::kAsyncValue), value_type_(std::move(value_type)) {}

RankedTensorType::RankedTensorType(ArrayRef<Index> sizes, DType element_type)
    : Type(TypeKind::kRankedTensor),
      sizes_(sizes.begin(), sizes.end()),
      element_type_(element_type) {}

ArrayRef<Index> RankedTensorType::sizes() const { return sizes_; }

unsigned RankedTensorType::rank() const { return sizes_.size(); }

DType RankedTensorType::element_type() const { return element_type_; }

UnrankedTensorType::UnrankedTensorType(DType element_type)
    : Type(TypeKind::kUnrankedTensor), element_type_(element_type) {}

DType UnrankedTensorType::element_type() const { return element_type_; }

MemrefType::MemrefType(ArrayRef<Index> sizes, DType element_type)
    : Type(TypeKind::kMemref),
      sizes_(sizes.begin(), sizes.end()),
      element_type_(element_type) {}

ArrayRef<Index> MemrefType::sizes() const { return sizes_; }

unsigned MemrefType::rank() const { return sizes_.size(); }

DType MemrefType::element_type() const { return element_type_; }

UnrankedMemrefType::UnrankedMemrefType(DType element_type)
    : Type(TypeKind::kUnrankedMemref), element_type_(element_type) {}

DType UnrankedMemrefType::element_type() const { return element_type_; }

KernelContextOperandType::KernelContextOperandType()
    : Type(TypeKind::kKernelContext) {}

FunctionType::FunctionType(llvm::SmallVector<std::unique_ptr<Type>> operands,
                           llvm::SmallVector<std::unique_ptr<Type>> results)
    : operands_(std::move(operands)), results_(std::move(results)) {}

const Type* FunctionType::operand(unsigned index) const {
  return operands_[index].get();
}
const Type* FunctionType::result(unsigned index) const {
  return results_[index].get();
}

unsigned FunctionType::num_operands() const { return operands_.size(); }
unsigned FunctionType::num_results() const { return results_.size(); }

static Expected<DType> ConvertElementType(mlir::Type type) {
  if (type.isF32()) return DType::F32;
  if (type.isUnsignedInteger(8)) return DType::UI8;
  if (type.isUnsignedInteger(32)) return DType::UI32;
  if (type.isUnsignedInteger(64)) return DType::UI64;
  if (type.isInteger(1)) return DType::I1;
  if (type.isInteger(8)) return DType::I8;
  if (type.isInteger(32)) return DType::I32;
  if (type.isInteger(64)) return DType::I64;

  return MakeStringError("unsupported element type: ", type);
}

static Expected<std::unique_ptr<Type>> ConvertType(mlir::Type type) {
  // mlir::async::TokenType -> tfrt::jitrt::AsyncTokenType
  if (type.isa<mlir::async::TokenType>())
    return std::make_unique<AsyncTokenType>();

  // mlir::async::ValueType -> tfrt::jitrt::AsyncValueType
  if (auto value = type.dyn_cast<mlir::async::ValueType>()) {
    if (!value.getValueType().isa<mlir::MemRefType>())
      return MakeStringError("async value can only hold memref type");

    auto value_type = ConvertType(value.getValueType());
    if (auto err = value_type.takeError()) return std::move(err);

    return std::make_unique<AsyncValueType>(std::move(*value_type));
  }

  // mlir::RankedTensorType -> tfrt::jitrt::RankedTensorType
  if (auto tensor = type.dyn_cast<mlir::RankedTensorType>()) {
    auto element_type = ConvertElementType(tensor.getElementType());
    if (auto err = element_type.takeError()) return std::move(err);
    return std::make_unique<RankedTensorType>(tensor.getShape(), *element_type);
  }

  // mlir::UnrankedTensorType -> tfrt::jitrt::UnrankedTensorType
  if (auto tensor = type.dyn_cast<mlir::UnrankedTensorType>()) {
    auto element_type = ConvertElementType(tensor.getElementType());
    if (auto err = element_type.takeError()) return std::move(err);
    return std::make_unique<UnrankedTensorType>(*element_type);
  }

  // mlir::MemrefType -> tfrt::jitrt::MemrefType
  if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
    auto element_type = ConvertElementType(memref.getElementType());
    if (auto err = element_type.takeError()) return std::move(err);
    return std::make_unique<MemrefType>(memref.getShape(), *element_type);
  }

  // mlir::UnrankedMemrefType -> tfrt::jitrt::UnrankedMemrefType
  if (auto memref = type.dyn_cast<mlir::UnrankedMemRefType>()) {
    auto element_type = ConvertElementType(memref.getElementType());
    if (auto err = element_type.takeError()) return std::move(err);
    return std::make_unique<UnrankedMemrefType>(*element_type);
  }

  // KernelContextType -> KernelContextOperandType (both in tfrt::jitrt).
  if (auto ctx = type.dyn_cast<KernelContextType>())
    return std::make_unique<KernelContextOperandType>();

  return MakeStringError("unsupported type: ", type);
}

/*static*/ Expected<FunctionType> FunctionType::Convert(
    mlir::FunctionType type) {
  assert(type && "function type must be not null");

  llvm::SmallVector<std::unique_ptr<Type>> operands;
  llvm::SmallVector<std::unique_ptr<Type>> results;

  operands.reserve(type.getNumInputs());
  results.reserve(type.getNumResults());

  auto error = [](string_view kind, unsigned i, mlir::Type type, Error err) {
    return MakeStringError("can't convert ", kind, " #", i, " type ", type,
                           " to the runtime type: ", err);
  };

  for (unsigned i = 0; i < type.getNumInputs(); ++i) {
    Expected<std::unique_ptr<Type>> converted = ConvertType(type.getInput(i));
    if (auto err = converted.takeError())
      return error("input", i, type.getInput(i), std::move(err));
    operands.push_back(std::move(*converted));
  }

  for (unsigned i = 0; i < type.getNumResults(); ++i) {
    Expected<std::unique_ptr<Type>> converted = ConvertType(type.getResult(i));
    if (auto err = converted.takeError())
      return error("result", i, type.getResult(i), std::move(err));
    results.push_back(std::move(*converted));
  }

  return FunctionType(std::move(operands), std::move(results));
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

static bool areCompatibleTypes(DType type1, DType type2) {
  auto compatible = [&](DType fromType, DType toType) {
    return (type1 == fromType && type2 == toType) ||
           (type1 == toType && type2 == fromType);
  };
  // I1 and I8 types are compatible since they both are 1-byte size at runtime.
  if (compatible(DType::I1, DType::I8)) return true;

  // Signed and unsigned integers of the same size are compatible in memory.
  if (compatible(DType::I8, DType::UI8) ||
      compatible(DType::I16, DType::UI16) ||
      compatible(DType::I32, DType::UI32) ||
      compatible(DType::I64, DType::UI64))
    return true;

  return type1 == type2;
}

// -------------------------------------------------------------------------- //
// Verify that signature operands types are matching runtime operands types.
// -------------------------------------------------------------------------- //

static Error VerifyMemrefOperand(unsigned index, DType element_type,
                                 Optional<ArrayRef<Index>> sizes,
                                 const MemrefDesc& memref) {
  // Format memref operand and expected type for user-friendly error messages.
  auto format_operands = [&]() -> std::string {
    std::string err;
    llvm::raw_string_ostream os(err);

    auto dim = [](Index d) -> std::string {
      return d == MemrefType::kDynamicSize ? "?" : std::to_string(d);
    };

    auto print_shaped = [&](Optional<ArrayRef<Index>> dims, DType dtype) {
      if (!dims.hasValue()) {
        os << "[*x" << dtype << "]";
        return;
      }

      if (dims->empty()) {
        os << "[" << dtype << "]";
        return;
      }

      os << "[" << dim((*dims)[0]);
      for (int i = 1; i < dims->size(); ++i) os << "x" << dim((*dims)[i]);
      os << "x" << dtype << "]";
    };

    os << "got ";
    print_shaped({memref.sizes}, memref.dtype);
    os << " vs expected ";
    print_shaped(sizes, element_type);

    return err;
  };

  // Check that memref data type is compatible with the operand element type.
  if (!areCompatibleTypes(element_type, memref.dtype)) {
    return MakeStringError(
        "operand #", index,
        " type is not compatible with the expected element type: ",
        memref.dtype, " vs ", element_type, " (", format_operands(), ")");
  }

  // Skip sizes verification if they are not available.
  if (!sizes.hasValue()) return Error::success();

  // Check that memref rank is the same as operand rank.
  if (memref.sizes.size() != sizes->size())
    return MakeStringError(
        "operand #", index,
        " rank does not match expected input rank: ", memref.sizes.size(),
        " vs ", sizes->size(), " (", format_operands(), ")");

  // Check that all statically known dimensions matches the memref dimensions.
  for (const auto& pair : llvm::enumerate(llvm::zip(memref.sizes, *sizes))) {
    Index operand_dim = std::get<0>(pair.value());
    Index expected_dim = std::get<1>(pair.value());

    bool is_dynamic_dim = mlir::ShapedType::isDynamic(expected_dim);

    if (operand_dim != expected_dim && !is_dynamic_dim)
      return MakeStringError(
          "operand #", index, " dimension #", pair.index(),
          " does not match expected input dimension: ", operand_dim, " vs ",
          expected_dim, " (", format_operands(), ")");
  }

  return Error::success();
}

static Error VerifyMemrefOperand(unsigned index, const RankedTensorType& type,
                                 const MemrefDesc& memref) {
  return VerifyMemrefOperand(index, type.element_type(), type.sizes(), memref);
}

static Error VerifyMemrefOperand(unsigned index, const MemrefType& type,
                                 const MemrefDesc& memref) {
  return VerifyMemrefOperand(index, type.element_type(), type.sizes(), memref);
}

static Error VerifyMemrefOperand(unsigned index, mlir::ShapedType type,
                                 const MemrefDesc& memref) {
  auto element_type = ConvertElementType(type.getElementType());
  if (auto err = element_type.takeError()) return err;

  // We do not support unranked memrefs at runtime, however we need to verify
  // operand types when we do compiled kernel specialization to shape.
  return VerifyMemrefOperand(
      index, *element_type,
      type.hasRank() ? Optional<ArrayRef<Index>>{type.getShape()} : llvm::None,
      memref);
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
static void AddMemrefArgument(const MemrefDesc& memref,
                              llvm::SmallVectorImpl<void*>* args) {
  assert(memref.sizes.size() == memref.strides.size());

  size_t size = args->size();
  args->resize(size + GetArgsCount(memref));

  auto* storage = &(*args)[size];
  auto add_arg = [&](const void* p) {
    *storage = const_cast<void*>(p);
    ++storage;
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
  if (operands.size() != runtime_signature_.num_operands() - 1)
    return MakeStringError(
        "number of operands doesn't match the function signature: ",
        operands.size(), " vs ", runtime_signature_.num_operands() - 1);

  // Verify that all operands passed at runtime are compatible with compiled
  // function signature.
  auto kctx = dyn_cast<KernelContextOperandType>(runtime_signature_.operand(0));
  if (!kctx) {
    return MakeStringError(
        "expected KernelContext in first argument of "
        "signature, got: ",
        runtime_signature_.operand(0));
  }

  // We use 0-based index for operands, because the kernel context operand is an
  // internal implementation detail, and in case of an error users should get
  // back operand index corresponding to the user provided signature.
  for (unsigned i = 0; i < operands.size(); ++i) {
    if (auto* memref =
            dyn_cast<MemrefType>(runtime_signature_.operand(1 + i))) {
      if (auto err = VerifyMemrefOperand(i, *memref, operands[i])) return err;
    } else {
      return MakeStringError("expected memref operand at #", i,
                             ", got: ", *runtime_signature_.operand(i));
    }
  }

  size_t n_args_elems = 1 + GetArgsCount(operands);
  call_frame->args.reserve(n_args_elems);

  // Add a placeholder for the kernel context as the first argument.
  call_frame->args.push_back(nullptr);

  // Pack all Memref operands as pointers to the call frame arguments.
  for (const MemrefDesc& desc : operands)
    AddMemrefArgument(desc, &call_frame->args);

  // Allocate storage for results.
  call_frame->results.resize_for_overwrite(results_memory_layout_.size);

  assert(call_frame->args.size() == n_args_elems &&
         "reserved number of args must match the actual number");

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

void EmitErrors(RemainingResults results, Error error,
                const ExecutionContext& exec_ctx) {
  auto async_error = EmitErrorAsync(exec_ctx, std::move(error));
  for (int i = 0; i < results.size(); ++i) results[i] = async_error;
}

void EmitErrors(RemainingResults results, DecodedDiagnostic error,
                const ExecutionContext& exec_ctx) {
  return EmitErrors(results, MakeStringError(error), exec_ctx);
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
    return EmitErrors(results, std::move(err), exec_ctx);

  Execute(call_frame, exec_ctx, opts);

  // Convert compiled function return values into results.
  if (auto err = ReturnResults(results, exec_ctx, &call_frame)) return err;

  return Error::success();
}

void Executable::Execute(CallFrame& call_frame,
                         const ExecutionContext& exec_ctx,
                         const ExecuteOpts& opts) const {
  // Set the AsyncRuntime to be used by all async tasks spawned by the compiled
  // kernel function.
  SetAsyncRuntime({exec_ctx.host(), opts.async_runtime_worker_threads});

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
                                const ExecutionContext& exec_ctx,
                                CallFrame* call_frame) const {
  // Forward error to all results.
  // TODO(ezhulenev): Forward the underlying error to all results once it will
  // be supported by the runtime API.
  if (call_frame->is_error) {
    results.EmitErrors(EmitErrorAsync(
        exec_ctx,
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

  if (!converted)
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
    const std::function<void(mlir::PassManager&)>& register_pipeline) {
  if (!register_pipeline) return mlir::success();

  mlir::PassManager pm(module.getContext());
  SetupPassDebugging(module.getContext(), pm);
  register_pipeline(pm);

  return pm.run(module);
}

// Runs the user-provided compilation pipeline to compile the module to LLVM.
static mlir::LogicalResult RunCompilationPipeline(
    mlir::ModuleOp module, const CompilationOptions& opts) {
  return RunPipeline(module, opts.register_compilation_pipeline);
}

// Runs the user-provided specialization pipeline.
static mlir::LogicalResult RunSpecializationPipeline(
    mlir::ModuleOp module, const CompilationOptions& opts) {
  return RunPipeline(module, opts.register_specialization_pipeline);
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
      std::unique_ptr<JitCompilationContext>,
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
  // Returns error if operands are not compatible with compiled module
  // entrypoint signature.
  llvm::Error Specialize(ArrayRef<MemrefDesc> operands,
                         ArrayRef<SymbolicShape> symbolic_shapes,
                         ArrayRef<OperandConstraint> constraints,
                         const JitExecutable::Listener* listener);

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
  mlir::OwningModuleRef module_;  // can be null if failed to parse the module
  mlir::FuncOp entrypoint_;       // can be null if failed to parse the module
  bool specialized_;
};
}  // namespace

// Creates a new MLIR Context and registers all the dialects that are expected
// in the compiled module.
static std::unique_ptr<mlir::MLIRContext> CreateMlirContext(
    const CompilationOptions& opts) {
  mlir::DialectRegistry registry;

  // TODO(b/210116436): Dialects and translation registration should be
  // controlled by the `opts.register_dialects` similar to passes.

  // Register MLIR dialects supported by the compiled kernels.
  registry.insert<mlir::AffineDialect, mlir::arith::ArithmeticDialect,
                  mlir::async::AsyncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::StandardOpsDialect,
                  mlir::tensor::TensorDialect, mlir::vector::VectorDialect,
                  RuntimeDialect>();

  // Register MLIR dialects that can be translated to LLVM IR.
  mlir::registerArmNeonDialectTranslation(registry);
  mlir::registerAMXDialectTranslation(registry);
  mlir::registerArmSVEDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerX86VectorDialectTranslation(registry);

  // Register other information needed for passes.
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);

  // Register additional dialects provided via compilation options.
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
  module_ = mlir::parseSourceFile(source_mgr_, context_.get());
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
    std::unique_ptr<JitCompilationContext> ctx,
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

  return Executable(ctx->name().str(), std::move(*engine), *kernel_fn,
                    std::move(*signature), std::move(*runtime_signature),
                    std::move(*results_memory_layout), specialization,
                    time_to_compile);
}

// Return input `type` specialized to memref operand and its symbolic shape.
static llvm::Expected<mlir::Type> SpecializeOperandType(
    unsigned index, mlir::Type type, const MemrefDesc& operand,
    const SymbolicShape& symbolic_shape) {
  // Replace all symbolic dimensions with dynamic dimension.
  auto shape = SymbolicShapesResolver::Normalize(symbolic_shape);

  if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
    if (auto err = VerifyMemrefOperand(index, memref, operand))
      return std::move(err);
    return mlir::MemRefType::get(shape, memref.getElementType());
  }

  if (auto tensor = type.dyn_cast<mlir::RankedTensorType>()) {
    if (auto err = VerifyMemrefOperand(index, tensor, operand))
      return std::move(err);
    return mlir::RankedTensorType::get(shape, tensor.getElementType());
  }

  if (auto tensor = type.dyn_cast<mlir::UnrankedTensorType>()) {
    if (auto err = VerifyMemrefOperand(index, tensor, operand))
      return std::move(err);
    return mlir::RankedTensorType::get(shape, tensor.getElementType());
  }

  return MakeStringError("Unsupported input type: ", type);
}

// Gets (copies) the values from `desc`, returning them in a DenseElementsAttr.
// If it cannot extract the values, returns an empty attribute.
static mlir::DenseElementsAttr GetMemrefValues(mlir::Builder& builder,
                                               mlir::TensorType operand_type,
                                               const MemrefDesc& desc) {
  size_t rank = desc.sizes.size();
  if (rank != 0 && rank != 1) return {};

  llvm::SmallVector<mlir::Attribute> attributes;
  size_t num_values = rank == 0 ? 1 : desc.sizes[0];
  switch (desc.dtype) {
    case DType::I32: {
      const auto* data = static_cast<TypeForDTypeKind<DType::I32>*>(desc.data);
      for (int i = 0; i < num_values; ++i) {
        attributes.push_back(builder.getI32IntegerAttr(data[i]));
      }
    } break;
    case DType::I64: {
      const auto* data = static_cast<TypeForDTypeKind<DType::I64>*>(desc.data);
      for (int i = 0; i < num_values; ++i) {
        attributes.push_back(builder.getI64IntegerAttr(data[i]));
      }
    } break;
    default:
      return {};
  }

  // Update operand type to a ranked tensor type with statically known shape.
  auto element_type = operand_type.getElementType();
  auto ranked_tensor = mlir::RankedTensorType::get(desc.sizes, element_type);

  return mlir::DenseElementsAttr::get(ranked_tensor, attributes);
}

llvm::Error JitCompilationContext::Specialize(
    ArrayRef<MemrefDesc> operands, ArrayRef<SymbolicShape> symbolic_shapes,
    ArrayRef<OperandConstraint> constraints,
    const JitExecutable::Listener* listener) {
  assert(!specialized_ && "can specialize executable only once");
  specialized_ = true;

  mlir::FuncOp func = entrypoint();
  unsigned num_inputs = func.getNumArguments();

  mlir::MLIRContext* ctx = func.getContext();

  // Specialize all function inputs to the given operands.
  llvm::SmallVector<mlir::Type> specialized_inputs(num_inputs);
  for (unsigned i = 0; i < num_inputs; ++i) {
    auto specialized = SpecializeOperandType(i, func.getType().getInput(i),
                                             operands[i], symbolic_shapes[i]);
    if (auto err = specialized.takeError()) return err;
    specialized_inputs[i] = *specialized;
  }

  // Update function type to a new specialized one.
  auto specialized = mlir::FunctionType::get(ctx, specialized_inputs,
                                             func.getType().getResults());
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

  // Add symbolic shapes as arguments attributes.
  for (unsigned i = 0; i < num_inputs; ++i) {
    const SymbolicShape& shape = symbolic_shapes[i];
    int64_t rank = shape.size();

    // Skip statically known shapes.
    if (llvm::all_of(shape, [](int64_t dim) { return dim >= 0; })) continue;

    // Symbolic shape attribute stored as 1d tensor attribute.
    auto i64 = mlir::IntegerType::get(ctx, 64);
    auto tensor = mlir::RankedTensorType::get({rank}, i64);

    // Create i64 attributes from the symbolic shape values.
    llvm::SmallVector<mlir::Attribute> values(rank);
    for (unsigned d = 0; d < rank; ++d)
      values[d] = mlir::IntegerAttr::get(i64, shape[d]);

    func.setArgAttr(i, "jitrt.symbolic_shape",
                    mlir::DenseElementsAttr::get(tensor, values));
  }

  // Sink small constants into the function body.
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(&func.front());
  mlir::Location loc = func.getLoc();

  for (int i = 0; i < constraints.size(); ++i) {
    if (constraints[i] != OperandConstraint::kValue) continue;

    // We only support sinking of Tensor operands into the function body.
    mlir::Type input_ty = func.getType().getInput(i);
    mlir::TensorType tensor_ty = input_ty.dyn_cast<mlir::TensorType>();
    if (!tensor_ty || !SupportsValueSpecialization(tensor_ty)) {
      return MakeStringError("non-sinkable operand was marked for sinking: ",
                             input_ty);
    }

    // Get the operand value from the runtime memref operand.
    mlir::DenseElementsAttr value =
        GetMemrefValues(builder, tensor_ty, operands[i]);
    if (!value) {
      return MakeStringError("cannot get value from operand type: ", input_ty);
    }

    auto cst =
        builder.create<mlir::arith::ConstantOp>(loc, value.getType(), value);
    entry_block.getArgument(i).replaceAllUsesWith(cst);

    if (listener) listener->notifyValueSpecialized(i, value.getType(), value);
  }

  if (listener) {
    llvm::SmallVector<mlir::DictionaryAttr> specialized_attrs;
    func.getAllArgAttrs(specialized_attrs);
    listener->notifyModuleSpecialized(specialized_inputs, specialized_attrs);
  }

  // Run the user-provided specialization pipeline to take advantage of the
  // specialized operands and sunk constants.
  if (failed(RunSpecializationPipeline(*module_, opts_)))
    return Error("failed to run specialization pipeline");

  return Error::success();
}

//----------------------------------------------------------------------------//
// Resolving JitExecutable OperandConstraint.
//----------------------------------------------------------------------------//

using Specialization = CompilationOptions::Specialization;

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
    mlir::FuncOp func) {
  llvm::SmallVector<OperandConstraint> constraints;

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

  for (int i = 0; i < func.getNumArguments(); ++i) {
    auto operand_type = func.getType().getInput(i);

    auto constraint = parse(func.getArgAttr(i, JitExecutable::kConstraint));
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

//----------------------------------------------------------------------------//
// SymbolicShapesResolver implementation.
//----------------------------------------------------------------------------//

SymbolicShapesResolver::SymbolicShapesResolver(
    const FunctionType& signature, ArrayRef<OperandConstraint> constraints)
    : constraints_(constraints.begin(), constraints.end()) {
  for (unsigned i = 0; i < signature.num_operands(); ++i) {
    auto* type = signature.operand(i);

    // For unranked operands we do not know any static shape information.
    if (isa<UnrankedTensorType, UnrankedMemrefType>(type)) {
      operands_sizes_.emplace_back();
      continue;
    }

    auto emplace_sizes = [&](ArrayRef<Index> sizes) {
      operands_sizes_.emplace_back(llvm::to_vector(sizes));

      // Keep track of all statically known dimension sizes.
      for (Index size : sizes) {
        if (size != MemrefType::kDynamicSize) seen_static_sizes_.insert(size);
      }
    };

    // Copy memref dimensions sizes from the signature type.
    if (auto* memref = dyn_cast<MemrefType>(type)) {
      emplace_sizes(memref->sizes());
      continue;
    }

    // Copy tensor dimensions sizes from the signature type.
    if (auto* tensor = dyn_cast<RankedTensorType>(type)) {
      emplace_sizes(tensor->sizes());
      continue;
    }

    assert(false && "unsupported operand type");
  }

  // When resolving symbolic shapes we should visit operands starting from the
  // more constrained ones, because they can change the static signature of the
  // function, and this information should be propagated to operands with
  // dynamic shapes (e.g. all seen static sizes should be materialized in the
  // function signature).
  iteration_order_.resize(signature.num_operands());
  std::iota(iteration_order_.begin(), iteration_order_.end(), 0);

  // Make the sort stable so that dynamic shapes are computed deterministically.
  llvm::sort(iteration_order_, [&](size_t a, size_t b) {
    unsigned ca = static_cast<unsigned>(constraints[a]);
    unsigned cb = static_cast<unsigned>(constraints[b]);
    if (ca > cb) return true;
    return ca < cb ? false : a < b;
  });
}

mlir::FailureOr<llvm::SmallVector<SymbolicShape>>
SymbolicShapesResolver::Resolve(ArrayRef<MemrefDesc> operands) {
  // The number of operands must match the function signature.
  assert(operands.size() == operands_sizes_.size());

  // Mapping from the runtime dimension size to the symbolic dimension.
  llvm::SmallDenseMap<int64_t, int64_t, 16> size_to_symbolic_dim;

  // Resolved symbolic shapes.
  llvm::SmallVector<SymbolicShape> symbolic_shapes;
  symbolic_shapes.resize(operands.size());

  int64_t sym_dim = -2;  // the next symbolic dimension id

  for (size_t i : iteration_order_) {
    bool has_static_sizes = operands_sizes_[i].hasValue();
    ArrayRef<int64_t> runtime_sizes = operands[i].sizes;

    // Check that statically known rank matches the runtime rank.
    if (has_static_sizes && operands_sizes_[i]->size() != runtime_sizes.size())
      return mlir::failure();

    // For shape constrained operands use runtime shape.
    if (constraints_[i] == OperandConstraint::kShape) {
      symbolic_shapes[i].assign(runtime_sizes.begin(), runtime_sizes.end());

      // Add all runtime dimensions to the `size_to_symbolic_dim` to materialize
      // all dynamic dimensions of the same size as static dimensions.
      for (int64_t d : runtime_sizes) size_to_symbolic_dim.try_emplace(d, d);

      continue;
    }

    // Initialize symbolic shape with a statically known shape of the operand if
    // it is available, otherwise initialize it with a fully dynamic shape with
    // rank matching the runtime rank.
    if (has_static_sizes) {
      ArrayRef<int64_t> static_sizes = *operands_sizes_[i];
      assert(runtime_sizes.size() == static_sizes.size());
      symbolic_shapes[i].assign(static_sizes.begin(), static_sizes.end());
    } else {
      size_t rank = runtime_sizes.size();
      symbolic_shapes[i].resize(rank, MemrefType::kDynamicSize);
    }

    MutableArrayRef<int64_t> symbolic_sizes = symbolic_shapes[i];

    for (unsigned d = 0; d < runtime_sizes.size(); ++d) {
      int64_t symbolic_dim = symbolic_sizes[d];
      int64_t runtime_dim = runtime_sizes[d];

      // Skip statically known dimensions.
      if (symbolic_dim >= 0) {
        // Check that statically known dimension agrees with runtime dimension.
        if (symbolic_dim != runtime_dim) return mlir::failure();
        continue;
      }

      // Update unknown dimension to a static dimension.
      if (runtime_dim == 1 || seen_static_sizes_.contains(runtime_dim)) {
        symbolic_sizes[d] = runtime_dim;
        continue;
      }

      // Try to assign a symbolic dimension to the runtime dimension.
      auto emplaced = size_to_symbolic_dim.try_emplace(runtime_dim, sym_dim);
      symbolic_sizes[d] = emplaced.first->second;

      // Update the symbolic dimension if we assigned the previous value to the
      // runtime dimension size.
      if (emplaced.second) --sym_dim;
    }
  }

  return symbolic_shapes;
}

/*static*/ llvm::SmallVector<int64_t> SymbolicShapesResolver::Normalize(
    const SymbolicShape& shape) {
  auto normalize = llvm::map_range(shape, [](int64_t dim) {
    return std::max(dim, mlir::ShapedType::kDynamicSize);
  });
  return {normalize.begin(), normalize.end()};
}

//----------------------------------------------------------------------------//
// JitExecutable implementation.
//----------------------------------------------------------------------------//

/*static*/ void JitExecutable::DefaultCompilationTaskRunner(
    size_t, ArrayRef<OperandConstraint>, ArrayRef<MemrefDesc>,
    TaskFunction task, const ExecutionContext& exec_ctx) {
  EnqueueWork(exec_ctx, std::move(task));
}

/*static*/ Expected<JitExecutable> JitExecutable::Instantiate(
    string_view mlir_module, string_view entrypoint,
    CompilationOptions compilation_opts, CompilationTaskRunner runner) {
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
    return JitExecutable(mlir_module, entrypoint, std::move(compilation_opts),
                         std::move(*constraints), std::move(*signature),
                         /*default_executable=*/llvm::None, std::move(runner));

  // Otherwise try to compile the default executable.
  Expected<Executable> executable =
      JitCompilationContext::Compile(std::move(*ctx));
  if (auto err = executable.takeError()) return std::move(err);

  return JitExecutable(mlir_module, entrypoint, std::move(compilation_opts),
                       std::move(*constraints), std::move(*signature),
                       std::move(*executable), std::move(runner));
}

JitExecutable::JitExecutable(string_view mlir_module, string_view entrypoint,
                             CompilationOptions compilation_opts,
                             ArrayRef<OperandConstraint> constraints,
                             FunctionType signature,
                             Optional<Executable> default_executable,
                             CompilationTaskRunner runner)
    : mlir_module_(mlir_module.str()),
      entrypoint_(entrypoint.str()),
      compilation_opts_(std::move(compilation_opts)),
      constraints_(constraints.begin(), constraints.end()),
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

// Hashes the given operands.
// Note: due to value specialization, the resulting hash might depend on the
// values (and not only on the types) of the operands.
static llvm::hash_code HashOperands(ArrayRef<MemrefDesc> operands,
                                    ArrayRef<SymbolicShape> symbolic_shapes,
                                    ArrayRef<OperandConstraint> constraints) {
  llvm::hash_code hash(0);

  // Compute hash based on the symbolic shapes of the operands.
  for (const SymbolicShape& shape : symbolic_shapes) {
    hash = llvm::hash_combine(
        hash, shape.size(),
        llvm::hash_combine_range(shape.begin(), shape.end()));
  }

  // Mix values of arguments to be sunk into the hash.
  for (int i = 0; i < constraints.size(); ++i) {
    if (constraints[i] != OperandConstraint::kValue) continue;
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
    ArrayRef<MemrefDesc> operands, const ExecutionContext& exec_ctx,
    const Listener* listener) {
  // Do not try to compile specialized executable if it is explicitly disabled.
  if (compilation_opts_.specialization == Specialization::kDisabled)
    return DefaultExecutable();

  // Resolve symbolic shapes based on the static and runtime information.
  mlir::FailureOr<llvm::SmallVector<SymbolicShape>> symbolic_shapes =
      symbolic_shapes_resolver_.Resolve(operands);

  // If we failed to resolve the symbolic shapes, then we need to verify all the
  // operands to find the mismatch and report it to the user.
  if (mlir::failed(symbolic_shapes)) {
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

  // We rely on the hash code to find the specialized executable. In case of
  // a collision (practically impossible) incompatible operands will be rejected
  // by the executable operands verification.
  llvm::hash_code hash = HashOperands(operands, *symbolic_shapes, constraints_);

  // Maybe return Executable from the cache.
  if (auto cached = specializations_->Find(hash)) {
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
  if (auto err = (*ctx)->Specialize(operands, *symbolic_shapes, constraints_,
                                    listener)) {
    return MakeStringError("failed to specialize executable: ", err);
  }

  // Allocate a placeholder for the compiled specialization only after we are
  // ready to dispatch the compilation task.
  Specializations::Entry entry = specializations_->Allocate(hash);

  // We lost the race; some other invocation will do the compilation.
  if (!entry.allocated) return entry.ptr;

  // Get the specialization id from the size of the specializations cache.
  size_t specialization = entry.size - 1;

  // Construct the task that will do the specialized executable compilation.
  auto compile = TaskFunction([ctx = std::move(*ctx), ref = entry.ptr.CopyRef(),
                               specialization]() mutable {
    Expected<Executable> executable =
        JitCompilationContext::Compile(std::move(ctx), specialization);

    // Set the allocated entry async value state to error or concrete.
    if (auto err = executable.takeError()) {
      ref.SetError(std::move(err));
    } else {
      ref.emplace(std::move(*executable));
    }
  });

  // Offload specialization compilation to the user provided runner.
  runner_(specialization, constraints_, operands, std::move(compile), exec_ctx);

  // Use the default executable while we are compiling a specialized version if
  // this is not explicitly disabled by the compilation options.
  if (compilation_opts_.specialization == Specialization::kAlways)
    return entry.ptr;
  else
    return has_default_executable_ ? DefaultExecutable() : entry.ptr;
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
