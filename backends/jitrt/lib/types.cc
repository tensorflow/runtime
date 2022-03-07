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

//===- types.cc - ---------------------------------------------------------===//
// Types supported at the JitRt function boundary.
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/types.h"

#include <memory>
#include <string>
#include <utility>

#include "llvm/Support/Compiler.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "tfrt/jitrt/opdefs/rt_ops.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace jitrt {

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

//----------------------------------------------------------------------------//
// Compiled function signature types conversion from the MLIR types.
//----------------------------------------------------------------------------//

Expected<DType> ConvertElementType(mlir::Type type) {
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

Expected<std::unique_ptr<Type>> ConvertType(mlir::Type type) {
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

// -------------------------------------------------------------------------- //
// Verify that operands types are matching runtime values.
// -------------------------------------------------------------------------- //

static bool AreCompatibleTypes(DType type1, DType type2) {
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

Error VerifyMemrefOperand(unsigned index, DType element_type,
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
  if (LLVM_UNLIKELY(!AreCompatibleTypes(element_type, memref.dtype))) {
    return MakeStringError(
        "operand #", index,
        " type is not compatible with the expected element type: ",
        memref.dtype, " vs ", element_type, " (", format_operands(), ")");
  }

  // Skip sizes verification if they are not available.
  if (!sizes.hasValue()) return Error::success();

  // Check that memref rank is the same as operand rank.
  if (LLVM_UNLIKELY(memref.sizes.size() != sizes->size()))
    return MakeStringError(
        "operand #", index,
        " rank does not match expected input rank: ", memref.sizes.size(),
        " vs ", sizes->size(), " (", format_operands(), ")");

  // Check that all statically known dimensions matches the memref dimensions.
  for (const auto& pair : llvm::enumerate(llvm::zip(memref.sizes, *sizes))) {
    Index operand_dim = std::get<0>(pair.value());
    Index expected_dim = std::get<1>(pair.value());

    bool is_dynamic_dim = mlir::ShapedType::isDynamic(expected_dim);

    if (LLVM_UNLIKELY(operand_dim != expected_dim && !is_dynamic_dim))
      return MakeStringError(
          "operand #", index, " dimension #", pair.index(),
          " does not match expected input dimension: ", operand_dim, " vs ",
          expected_dim, " (", format_operands(), ")");
  }

  return Error::success();
}

Error VerifyMemrefOperand(unsigned index, const RankedTensorType& type,
                          const MemrefDesc& memref) {
  return VerifyMemrefOperand(index, type.element_type(), type.sizes(), memref);
}

Error VerifyMemrefOperand(unsigned index, const MemrefType& type,
                          const MemrefDesc& memref) {
  return VerifyMemrefOperand(index, type.element_type(), type.sizes(), memref);
}

Error VerifyMemrefOperand(unsigned index, mlir::ShapedType type,
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

}  // namespace jitrt
}  // namespace tfrt
