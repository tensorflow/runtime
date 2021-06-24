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

// This file implements MLIR operation functions for the data library.

#include "tfrt/data/opdefs/data_ops.h"

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/data/opdefs/types.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// DataDialect Dialect
//===----------------------------------------------------------------------===//

DataDialect::DataDialect(MLIRContext *context)
    : Dialect(/*name=*/getDialectNamespace(), context,
              TypeID::get<DataDialect>()) {
  context->getOrLoadDialect<TFRTDialect>();

  allowUnknownTypes();
  allowUnknownOperations();
  addTypes<DatasetType, IteratorType>();

  addOperations<
#define GET_OP_LIST
#include "tfrt/data/opdefs/data_ops_opdefs.cpp.inc"
      >();
}

mlir::Type DataDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (spec == "dataset") return DatasetType::get(getContext());
  if (spec == "iterator") return IteratorType::get(getContext());

  if (auto type = mlir::Dialect::parseType(parser)) return type;

  mlir::Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  mlir::emitError(loc) << "unknown data type " << spec;
  return {};
}

void DataDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &printer) const {
  if (type.isa<DatasetType>()) {
    printer << "dataset";
    return;
  }

  if (type.isa<IteratorType>()) {
    printer << "iterator";
    return;
  }
  llvm_unreachable("unknown data type");
}

namespace {

static Type GetIteratorType(Builder *builder) {
  return builder->getType<IteratorType>();
}

static Type GetChainType(Builder *builder) {
  return builder->getType<compiler::ChainType>();
}

}  // namespace

//===----------------------------------------------------------------------===//
// IteratorGetNextOp
//===----------------------------------------------------------------------===//

static ParseResult parseIteratorGetNextOp(OpAsmParser &parser,
                                          OperationState &result) {
  auto &builder = parser.getBuilder();
  auto iterator_type = GetIteratorType(&builder);
  auto chain_type = GetChainType(&builder);

  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<Type, 4> operand_types;
  SmallVector<Type, 4> result_types;
  operand_types.push_back(iterator_type);
  operand_types.push_back(chain_type);
  auto loc = parser.getNameLoc();

  if (parser.resolveOperands(operands, operand_types, loc, result.operands) ||
      parser.parseColonTypeList(result_types) ||
      parser.addTypeToList(chain_type, result.types) ||
      parser.addTypesToList(result_types, result.types))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// EnumerateIteratorOp
//===----------------------------------------------------------------------===//

static ParseResult parseEnumerateIteratorOp(OpAsmParser &parser,
                                            OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<Type, 4> types;
  // The first operand is the iterator.
  types.push_back(GetIteratorType(&parser.getBuilder()));
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseColonTypeList(types) ||
      parser.resolveOperands(operands, types, loc, result.operands))
    return failure();

  // The results have the same types as the operands besides the first
  // operand (the iterator).
  result.addTypes({types.begin() + 1, types.end()});
  return success();
}

// Verify that the signature of the functino matches the operands and results.
static LogicalResult verify(EnumerateIteratorOp op) {
  auto module = op->getParentOfType<ModuleOp>();
  auto function = module.lookupSymbol<FuncOp>(op.function());
  if (!function) {
    return op.emitOpError("function refers to an undefined function: ")
           << op.function();
  }

  auto function_type = function.getType();
  auto results_size = op.getResultTypes().size();

  if (function_type.getNumResults() != results_size) {
    return op.emitError(llvm::formatv(
        "requires the number of function results to be equal to the number of "
        "op results. Found {0} and {1}, respectively",
        function_type.getNumResults(), results_size));
  }

  if (function_type.getNumInputs() <= results_size) {
    // TODO(rachelim): Validate that the number of function inputs ==
    // number of function outputs + number of iterator components.
    // Currently, the number of iterator components is unknown.
    return op.emitError(
        llvm::formatv("requires the number of function inputs to be greater "
                      "than the number of function results. Namely, it should "
                      "have N more inputs, where N is the number of components "
                      "of the iterator. Found {0} and {1}, respectively",
                      function_type.getNumInputs(), results_size));
  }

  // Collect all the type lists for the op so that different pairs of type lists
  // can be compared for the compatibility. The op result types, function result
  // types, and final function input types, should all match.
  constexpr int kNumTypeLists = 3;
  const std::array<std::pair<std::string, TypeRange>, kNumTypeLists>
      type_lists = {{
          {"op results", op.getResultTypes()},
          {"function results", function_type.getResults()},
          {"final function inputs",
           function_type.getInputs().take_back(results_size)},
      }};
  for (int i = 0; i < kNumTypeLists; ++i) {
    for (int j = i + 1; j < kNumTypeLists; ++j) {
      auto &a = type_lists[i];
      auto &b = type_lists[j];

      for (int idx = 0; idx < results_size; ++idx) {
        auto a_type = a.second[idx];
        auto b_type = b.second[idx];

        if (a_type != b_type) {
          return op.emitError(llvm::formatv(
              "{0} type {1} is incompatible with {2} type {3} at index {4}",
              a.first, a_type, b.first, b_type, idx));
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RangeDatasetOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(RangeDatasetOp op) {
  // Only integer or float types are supported.
  if (!op.element_type().isIntOrFloat()) return failure();
  return success();
}

}  // namespace data
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/data/opdefs/data_ops_opdefs.cpp.inc"
