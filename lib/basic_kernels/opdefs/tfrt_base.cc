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

// This file implements tfrt dialect.

#include "tfrt/basic_kernels/opdefs/tfrt_base.h"

#include "mlir/Transforms/InliningUtils.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"
#include "tfrt/basic_kernels/opdefs/types.h"

namespace tfrt {
namespace compiler {

namespace {

struct TFRTInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool would_be_cloned) const final {
    // Currently only allow inlining callables called by tfrt.call op.
    return llvm::isa<CallOp>(call);
  }

  bool isLegalToInline(Operation *op, Region *dest, bool would_be_cloned,
                       BlockAndValueMapping &) const final {
    // All TFRT dialect ops can be inlined.
    return true;
  }

  void handleTerminator(
      mlir::Operation *op,
      llvm::ArrayRef<mlir::Value> values_to_replace) const final {
    // Handle the given inlined terminator by replacing it with a new operation
    // as necessary. Required when the region has only one block.
    auto return_op = llvm::dyn_cast<ReturnOp>(op);
    if (!return_op) return;

    for (auto iter : llvm::zip(values_to_replace, return_op.operands())) {
      auto original_value = std::get<0>(iter);
      auto new_value = std::get<1>(iter);
      original_value.replaceAllUsesWith(new_value);
    }
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// TFRTDialect Dialect
//===----------------------------------------------------------------------===//

TFRTDialect::TFRTDialect(mlir::MLIRContext *context)
    : mlir::Dialect(/*name=*/"tfrt", context, TypeID::get<TFRTDialect>()) {
  allowUnknownTypes();

  // TODO(b/160693129): Eventually specify all of the operations.
  allowUnknownOperations();

  addTypes<compiler::ChainType, compiler::StringType, compiler::TensorTypeType,
           compiler::DeviceType>();

  addInterfaces<TFRTInlinerInterface>();

  addOperations<
#define GET_OP_LIST
#include "tfrt/basic_kernels/opdefs/basic_kernels_opdefs.cpp.inc"
      >();
}

mlir::Type TFRTDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef spec = parser.getFullSymbolSpec();

  if (spec == "chain") return compiler::ChainType::get(getContext());
  if (spec == "string") return compiler::StringType::get(getContext());
  if (spec == "tensor_type") return compiler::TensorTypeType::get(getContext());
  if (spec == "device") return compiler::DeviceType::get(getContext());
  if (auto type = mlir::Dialect::parseType(parser)) return type;

  mlir::Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  mlir::emitError(loc) << "unknown tfrt type " << spec;
  return {};
}

void TFRTDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &printer) const {
  if (type.isa<compiler::ChainType>()) {
    printer << "chain";
  } else if (type.isa<compiler::StringType>()) {
    printer << "string";
  } else if (type.isa<compiler::TensorTypeType>()) {
    printer << "tensor_type";
  } else if (type.isa<compiler::DeviceType>()) {
    printer << "device";
  } else {
    llvm_unreachable("unknown tfrt type");
  }
}

}  // namespace compiler
}  // namespace tfrt
