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

//===- test_kernels.cc ----------------------------------------------------===//
//
// This file implements MLIR operation functions for the test_kernels library.
//
//===----------------------------------------------------------------------===//

#include "tfrt/test_kernels/opdefs/test_kernels.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"

namespace tfrt {
namespace test {

//===----------------------------------------------------------------------===//
// TestDialect Dialect
//===----------------------------------------------------------------------===//

TestDialect::TestDialect(MLIRContext *context)
    : Dialect(/*name*/ "tfrt_test", context) {
  allowUnknownTypes();
  allowUnknownOperations();

  addOperations<
#define GET_OP_LIST
#include "tfrt/test_kernels/opdefs/test_kernels_opdefs.cpp.inc"
      >();
}

// Verify that the specified region contains a tfrt.return operation with the
// specified type list and emit an error if not.
template <typename ResultTypeContainer>
static LogicalResult checkTFRTReturn(Operation *op, Region *region,
                                     ResultTypeContainer result_types) {
  assert(std::distance(region->begin(), region->end()) == 1 &&
         "verifier should already check region size");
  auto *block = &region->front();

  if (block->empty() || block->back().getName().getStringRef() != "tfrt.return")
    return op->emitOpError("expected tfrt.return in body");

  if (!std::equal(block->back().getOperandTypes().begin(),
                  block->back().getOperandTypes().end(), result_types.begin(),
                  result_types.end()))
    return block->back().emitOpError()
           << "operand types don't match '" << op->getName() << "' result";

  return success();
}

//===----------------------------------------------------------------------===//
// DoAsyncOp
//===----------------------------------------------------------------------===//

static ParseResult parseDoAsyncOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperandList(operands)) return failure();

  if (succeeded(parser.parseOptionalKeyword("attributes"))) {
    if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  }

  FunctionType types;
  llvm::SMLoc type_loc = parser.getCurrentLocation();
  if (parser.parseColonType(types) ||
      parser.addTypesToList(types.getResults(), result.types))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  return failure(parser.resolveOperands(operands, types.getInputs(), type_loc,
                                        result.operands) ||
                 parser.parseRegion(*body, operands, types.getInputs(),
                                    /*enableNameShadowing=*/true));
}

static void print(OpAsmPrinter &p, DoAsyncOp op) {
  p << "tfrt_test.do.async ";
  p.printOperands(op.getOperands());
  if (!op.getAttrs().empty()) {
    p << " attributes ";
    p.printOptionalAttrDict(op.getAttrs());
  }
  p << " : (";
  interleaveComma(op.getOperandTypes(), p);
  p << ") -> (";
  interleaveComma(op.getResultTypes(), p);
  p << ") ";

  // Reuse the argument names provided to the op for the bbarg names within
  // the region.
  p.shadowRegionArgs(op.region(), op.getOperands());
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
}

static LogicalResult verify(DoAsyncOp op) {
  return checkTFRTReturn(op, &op.region(), op.getResultTypes());
}

//===----------------------------------------------------------------------===//
// BenchmarkOp
//===----------------------------------------------------------------------===//

// Parse the BenchmarkOp in the following format
// tfrt_test.benchmark "add.i32"(%c : i32, %d : f32)
//       max_count = 100, duration_secs = 1 {
// ...
// }

static ParseResult parseBenchmarkOp(OpAsmParser &parser,
                                    OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseAttribute(nameAttr, "name", result.attributes))
    return failure();

  // Parse the operands, e.g. (%c : i32, %d : f32)
  if (parser.parseLParen()) return failure();

  SmallVector<OpAsmParser::OperandType, 4> operands;
  SmallVector<Type, 4> types;
  llvm::SMLoc type_loc = parser.getCurrentLocation();

  if (parser.parseOptionalRParen()) {
    // Parse non-empty operands
    do {
      // Parse %c : i32,
      OpAsmParser::OperandType operand;
      Type type;

      if (parser.parseOperand(operand) || parser.parseColonType(type))
        return failure();

      operands.push_back(operand);
      types.push_back(type);

    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen()) return failure();
  }

  if (parser.resolveOperands(operands, types, type_loc, result.operands))
    return failure();

  auto chainType = ChainType::get(result.getContext());
  if (parser.addTypeToList(chainType, result.types)) return failure();

  auto parseIntegerKeywordAttr = [&]() -> ParseResult {
    StringRef attr;
    Attribute resultAttr;

    return failure(parser.parseKeyword(&attr) || parser.parseEqual() ||
                   parser.parseAttribute(resultAttr,
                                         parser.getBuilder().getIntegerType(32),
                                         attr, result.attributes));
  };

  // Parse the keyword attribute, e.g. max_count = 100, duration_secs = 1
  do {
    if (parseIntegerKeywordAttr()) return failure();
  } while (succeeded(parser.parseOptionalComma()));

  auto setDefaultAttrIfUnset = [&](const char *attr_name, int value) {
    bool found = llvm::any_of(result.attributes,
                              [attr_name](const NamedAttribute &attr) {
                                return attr.first == attr_name;
                              });
    if (!found) {
      IntegerAttr default_val = parser.getBuilder().getI32IntegerAttr(value);
      result.addAttribute(attr_name, default_val);
    }
  };

  // Set the default attribute num_warmup_runs to 1 if unset
  setDefaultAttrIfUnset("num_warmup_runs", 1);

  Region *target = result.addRegion();
  return parser.parseRegion(*target, operands, types,
                            /*enableNameShadowing=*/true);
}

// Print the BenchmarkOp in the following format
// tfrt_test.benchmark "add.i32"(%c : i32, %d : f32)
//       max_count = 100, duration_secs = 1 {
// ...
// }
static void print(OpAsmPrinter &p, BenchmarkOp op) {
  p << "tfrt_test.benchmark ";

  // Print the name attribute, e.g "add.i32"
  auto name_attr = op.getAttr("name");
  p << name_attr;

  // Print the operands and types, e.g. (%c : i32, %d : f32)
  p << '(';
  llvm::interleaveComma(llvm::zip(op.getOperands(), op.getOperandTypes()), p,
                        [&](const auto &it) {
                          p << std::get<0>(it) << " : " << std::get<1>(it);
                        });
  p << ") ";

  bool need_comma = false;

  // Print the attributes, e.g. max_count = 100, duration_secs = 1
  for (auto &name_attr : op.getAttrs()) {
    auto id = name_attr.first;
    if (id == "name") continue;

    if (need_comma) p << ", ";

    auto attr = name_attr.second;

    p << id << " = ";
    if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
      int_attr.getValue().print(p.getStream(), /*isSigned=*/false);
    } else {
      op.emitOpError("Unexpected attribute");
    }

    need_comma = true;
  }

  p << ' ';

  // Print the region
  // Reuse the argument names provided to the op for the bbarg names within
  // the region.
  p.shadowRegionArgs(op.region(), op.getOperands());
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
}

static LogicalResult verify(BenchmarkOp op) {
  // Verify that the target benchmark region has exactly one return value.
  auto &region = op.region();
  auto &last_op = region.front().back();
  if (last_op.getName().getStringRef() != "tfrt.return") {
    return op.emitOpError("missing return statement");
  }
  if (last_op.getNumOperands() != 1) {
    return op.emitOpError(
        "incorrect number of return values. One return value is expected");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SyncBenchmarkOp
//===----------------------------------------------------------------------===//

// Parse the SyncBenchmarkOp in the following format
//
// tfrt_test.sync_benchmark @fibonacci.i32()
//       duration_secs = 1, max_count = 100, num_warmup_runs = 10

static ParseResult parseSyncBenchmarkOp(OpAsmParser &parser,
                                        OperationState &result) {
  SymbolRefAttr targetFnAttr;
  if (parser.parseAttribute(targetFnAttr, "target_fn", result.attributes))
    return failure();

  // Parse the operands, e.g. (%c : i32, %d : f32)
  if (parser.parseLParen()) return failure();

  SmallVector<OpAsmParser::OperandType, 4> operands;
  SmallVector<Type, 4> types;
  llvm::SMLoc type_loc = parser.getCurrentLocation();

  if (parser.parseOptionalRParen()) {
    // Parse non-empty operands
    do {
      // Parse %c : i32,
      OpAsmParser::OperandType operand;
      Type type;

      if (parser.parseOperand(operand) || parser.parseColonType(type))
        return failure();

      operands.push_back(operand);
      types.push_back(type);

    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen()) return failure();
  }

  if (parser.resolveOperands(operands, types, type_loc, result.operands))
    return failure();

  auto parseIntegerKeywordAttr = [&]() -> ParseResult {
    StringRef attr;
    Attribute resultAttr;

    return failure(parser.parseKeyword(&attr) || parser.parseEqual() ||
                   parser.parseAttribute(resultAttr,
                                         parser.getBuilder().getIntegerType(32),
                                         attr, result.attributes));
  };

  // Parse the keyword attribute, e.g. max_count = 100, duration_secs = 1
  do {
    if (parseIntegerKeywordAttr()) return failure();
  } while (succeeded(parser.parseOptionalComma()));

  auto setDefaultAttrIfUnset = [&](const char *attr_name, int value) {
    bool found = llvm::any_of(result.attributes,
                              [attr_name](const NamedAttribute &attr) {
                                return attr.first == attr_name;
                              });
    if (!found) {
      IntegerAttr default_val = parser.getBuilder().getI32IntegerAttr(value);
      result.addAttribute(attr_name, default_val);
    }
  };

  // Set the default attribute num_warmup_runs to 1 if unset
  setDefaultAttrIfUnset("num_warmup_runs", 1);

  return success();
}

// Print the SyncBenchmarkOp in the following format
// tfrt_test.sync_benchmark @fibonacci.i32()
//       max_count = 100, duration_secs = 1
static void print(OpAsmPrinter &p, SyncBenchmarkOp op) {
  p << "tfrt_test.sync_benchmark ";

  // Print the target benchmark function
  p << op.getAttr("target_fn");

  // Print the operands and types, e.g. (%c : i32, %d : f32)
  p << '(';
  llvm::interleaveComma(llvm::zip(op.getOperands(), op.getOperandTypes()), p,
                        [&](const auto &it) {
                          p << std::get<0>(it) << " : " << std::get<1>(it);
                        });
  p << ") ";

  bool need_comma = false;

  // Print the attributes, e.g. max_count = 100, duration_secs = 1
  for (auto &name_attr : op.getAttrs()) {
    auto id = name_attr.first;
    if (id == "target_fn") continue;

    if (need_comma) p << ", ";

    auto attr = name_attr.second;

    p << id << " = ";
    if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
      int_attr.getValue().print(p.getStream(), /*isSigned=*/false);
    } else {
      op.emitOpError("Unexpected attribute");
    }

    need_comma = true;
  }
}

static LogicalResult verify(SyncBenchmarkOp op) {
  auto fnAttr = op.getAttrOfType<FlatSymbolRefAttr>("target_fn");
  if (!fnAttr)
    return op.emitOpError("requires a 'target_fn' symbol reference attribute");

  auto fn =
      op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
  if (!fn)
    return op.emitOpError() << "'" << fnAttr.getValue()
                            << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getType();
  if (fnType.getNumInputs() != op.getNumOperands())
    return op.emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (op.getOperand(i).getType() != fnType.getInput(i))
      return op.emitOpError("operand type mismatch");

  if (fnType.getNumResults() != 0)
    return op.emitOpError("Target benchmark function must return zero value.");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/test_kernels/opdefs/test_kernels_opdefs.cpp.inc"

}  // namespace test
}  // namespace tfrt
