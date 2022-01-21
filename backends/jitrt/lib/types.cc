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
#include <utility>

#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "tfrt/jitrt/opdefs/rt_ops.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace jitrt {

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

}  // namespace jitrt
}  // namespace tfrt
