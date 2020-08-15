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

//===- basic_kernels.cc ---------------------------------------------------===//
//
// This file implements MLIR operation functions for the basic_kernels library.
//
//===----------------------------------------------------------------------===//

#include "tfrt/basic_kernels/opdefs/basic_kernels.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "tfrt/basic_kernels/opdefs/types.h"

namespace tfrt {

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static ParseResult parseCallOp(OpAsmParser &parser, OperationState &result) {
  SymbolRefAttr calleeAttr;
  FunctionType calleeType;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  result.addAttribute("bef.nonstrict", parser.getBuilder().getUnitAttr());
  auto calleeLoc = parser.getNameLoc();
  if (parser.parseAttribute(calleeAttr, "callee", result.attributes) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(calleeType) ||
      parser.addTypesToList(calleeType.getResults(), result.types) ||
      parser.resolveOperands(operands, calleeType.getInputs(), calleeLoc,
                             result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, CallOp op) {
  p << "tfrt.call " << op.getAttr("callee") << '(';
  p.printOperands(op.getOperands());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{"callee", "bef.nonstrict"});
  p << " : ";
  p.printType(op.getCalleeType());
}

static LogicalResult verify(CallOp op) {
  // Check that the callee attribute was specified.
  auto fnAttr = op.getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return op.emitOpError("requires a 'callee' symbol reference attribute");
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

  if (fnType.getNumResults() != op.getNumResults())
    return op.emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (op.getResult(i).getType() != fnType.getResult(i))
      return op.emitOpError("result type mismatch");

  return success();
}

mlir::FunctionType CallOp::getCalleeType() {
  SmallVector<Type, 4> resultTypes(getResultTypes().begin(),
                                   getResultTypes().end());
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(argTypes, resultTypes, getContext());
}

//===----------------------------------------------------------------------===//
// Constant*Op
//===----------------------------------------------------------------------===//

static void printConstant(OpAsmPrinter &p, mlir::Operation *op) {
  p << op->getName() << " ";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});

  if (op->getAttrs().size() > 1) p << ' ';
  Attribute attr = op->getAttr("value");
  if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
    bool is_signed = int_attr.getType().isIndex() ||
                     int_attr.getType().getIntOrFloatBitWidth() != 1;
    int_attr.getValue().print(p.getStream(), is_signed);
  } else if (auto float_attr = attr.dyn_cast<FloatAttr>()) {
    p << float_attr.getValue().convertToFloat();
  } else {
    op->emitOpError("unknown attribute type");
  }
}

static ParseResult parseConstantOp(Type attrType, OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, attrType, "value", result.attributes) ||
      parser.addTypeToList(attrType, result.types))
    return failure();

  return success();
}

/// The constant op requires an attribute, and furthermore requires that it
/// matches the return type.
static LogicalResult verifyBoolConstant(Operation *op) {
  auto value = op->getAttr("value");
  if (!value) return op->emitOpError("requires a 'value' attribute");

  auto bool_attr = value.dyn_cast<BoolAttr>();
  if (!bool_attr)
    return op->emitOpError("unsupported 'value' attribute: ") << value;

  return success();
}

static LogicalResult verifyIntegerConstant(Operation *op,
                                           int expected_bitwidth) {
  auto value = op->getAttr("value");
  if (!value) return op->emitOpError("requires a 'value' attribute");

  auto type = op->getResult(0).getType();

  auto intAttr = value.dyn_cast<IntegerAttr>();
  if (!intAttr)
    return op->emitOpError("unsupported 'value' attribute: ") << value;

  // If the type has a known bitwidth we verify that the value can be
  // represented with the given bitwidth.
  auto bitwidth = type.cast<IntegerType>().getWidth();
  auto intVal = intAttr.getValue();
  if (!intVal.isSignedIntN(bitwidth) && !intVal.isIntN(bitwidth))
    return op->emitOpError(
        "requires 'value' to be an integer within the "
        "range of the integer result type");

  if (bitwidth != expected_bitwidth)
    return op->emitOpError("attribute bitwidth doesn't match op name");
  return success();
}

static LogicalResult verifyFloatConstant(Operation *op) {
  auto value = op->getAttr("value");
  if (!value) return op->emitOpError("requires a 'value' attribute");

  auto type = op->getResult(0).getType();

  auto floatAttr = value.dyn_cast<FloatAttr>();
  if (!floatAttr)
    return op->emitOpError("unsupported 'value' attribute: ") << value;

  if (floatAttr.getType() != type)
    return op->emitOpError("attribute 'value' does not match type") << value;
  return success();
}

static ParseResult parseConstantI1Op(OpAsmParser &parser,
                                     OperationState &result) {
  return parseConstantOp(IntegerType::get(1, result.getContext()), parser,
                         result);
}

static void print(OpAsmPrinter &p, ConstantI1Op op) { printConstant(p, op); }

static LogicalResult verify(ConstantI1Op op) {
  return verifyIntegerConstant(op, 1);
}

static ParseResult parseConstantI32Op(OpAsmParser &parser,
                                      OperationState &result) {
  return parseConstantOp(IntegerType::get(32, result.getContext()), parser,
                         result);
}

static void print(OpAsmPrinter &p, ConstantI32Op op) { printConstant(p, op); }

static LogicalResult verify(ConstantI32Op op) {
  return verifyIntegerConstant(op, 32);
}

static ParseResult parseConstantI64Op(OpAsmParser &parser,
                                      OperationState &result) {
  return parseConstantOp(IntegerType::get(64, result.getContext()), parser,
                         result);
}

static void print(OpAsmPrinter &p, ConstantI64Op op) { printConstant(p, op); }

static LogicalResult verify(ConstantI64Op op) {
  return verifyIntegerConstant(op, 64);
}

static ParseResult parseConstantF32Op(OpAsmParser &parser,
                                      OperationState &result) {
  return parseConstantOp(FloatType::getF32(result.getContext()), parser,
                         result);
}

static void print(OpAsmPrinter &p, ConstantF32Op op) { printConstant(p, op); }

static LogicalResult verify(ConstantF32Op op) {
  return verifyFloatConstant(op);
}

static ParseResult parseConstantF64Op(OpAsmParser &parser,
                                      OperationState &result) {
  return parseConstantOp(FloatType::getF64(result.getContext()), parser,
                         result);
}

static void print(OpAsmPrinter &p, ConstantF64Op op) { printConstant(p, op); }

static LogicalResult verify(ConstantF64Op op) {
  return verifyFloatConstant(op);
}

// Verify that the specified region contains a tfrt.return operation with the
// specified type list and emit an error if not.
template <typename ResultTypeContainer>
static LogicalResult checkTFRTReturn(Operation *op, Region *region,
                                     ResultTypeContainer result_types) {
  assert(std::distance(region->begin(), region->end()) == 1 &&
         "verifier should already check region size");
  auto *block = &region->front();

  if (block->empty() || !isa<ReturnOp>(block->back()))
    return op->emitOpError("expected tfrt.return in body");

  auto returnOp = cast<ReturnOp>(block->back());
  if (!std::equal(returnOp.getOperandTypes().begin(),
                  returnOp.getOperandTypes().end(), result_types.begin(),
                  result_types.end()))
    return returnOp.emitOpError()
           << "operand types don't match '" << op->getName() << "' result";

  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(IfOp op) {
  // Verify that the operands match the bb arguments.  The ODS verifier already
  // checked the first argument to be present and i1.
  auto *then_block = &op.then_region().front();
  if (op.getNumOperands() - 1 != then_block->getNumArguments())
    return op.emitOpError("incorrect number of arguments to 'then' block");
  auto *else_block = &op.else_region().front();
  if (op.getNumOperands() - 1 != else_block->getNumArguments())
    return op.emitOpError("incorrect number of arguments to 'else' block");

  for (unsigned i = 0, e = op.getNumOperands() - 1; i != e; ++i)
    if (op.getOperand(i + 1).getType() !=
            then_block->getArgument(i).getType() ||
        op.getOperand(i + 1).getType() != else_block->getArgument(i).getType())
      return op.emitOpError("operand/argument type mismatch");

  if (failed(checkTFRTReturn(op, &op.then_region(), op.getResultTypes())))
    return failure();

  return checkTFRTReturn(op, &op.else_region(), op.getResultTypes());
}

ParseResult parseIfOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperandList(operands)) return failure();

  result.addAttribute("bef.nonstrict", parser.getBuilder().getUnitAttr());
  if (succeeded(parser.parseOptionalKeyword("attributes"))) {
    if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  }

  FunctionType types;
  llvm::SMLoc type_loc = parser.getCurrentLocation();
  if (parser.parseColonType(types) ||
      parser.addTypesToList(types.getResults(), result.types))
    return failure();

  if (operands.empty())
    return parser.emitError(parser.getCurrentLocation(), "expected condition");

  auto body_operands = llvm::makeArrayRef(operands).drop_front();
  auto body_types = types.getInputs();
  auto i1_type = IntegerType::get(1, result.getContext());
  if (parser.resolveOperand(operands[0], i1_type, result.operands) ||
      parser.resolveOperands(body_operands, types.getInputs(), type_loc,
                             result.operands))
    return failure();

  // Parse the body region.
  Region *then_region = result.addRegion();
  if (parser.parseRegion(*then_region, body_operands, body_types,
                         /*enableNameShadowing=*/true))
    return failure();

  Region *else_region = result.addRegion();
  if (succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*else_region, body_operands, body_types,
                           /*enableNameShadowing=*/true))
      return failure();
  } else {
    // While the else region is syntactically optional, it is structurally
    // required in the IR and by the op kernel implementation.  Fill in the
    // default implementation.
    if (!types.getResults().empty())
      return parser.emitError(
          parser.getCurrentLocation(),
          "expected 'else' in 'tfrt.if' with result values");

    mlir::OpBuilder builder(result.getContext());
    auto *block = builder.createBlock(else_region);
    block->addArguments(body_types);
    builder.create<ReturnOp>(result.location);
  }

  return success();
}

void print(OpAsmPrinter &p, IfOp op) {
  p << "tfrt.if ";
  p.printOperands(op.getOperands());
  if (!op.getAttrs().empty()) {
    p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"bef.nonstrict"});
  }
  p << " : (";
  interleaveComma(llvm::drop_begin(op.getOperandTypes(), 1), p);
  p << ") -> (";
  interleaveComma(op.getResultTypes(), p);
  p << ") ";

  // Reuse the argument names provided to the op for the bbarg names within
  // the region.
  auto arg_name_values = llvm::drop_begin(op.getOperands(), 1);
  p.shadowRegionArgs(op.then_region(), arg_name_values);
  p.printRegion(op.then_region(), /*printEntryBlockArgs=*/false);
  p << " else ";
  p.shadowRegionArgs(op.else_region(), arg_name_values);
  p.printRegion(op.else_region(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// CondOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(CondOp op) {
  // Check that the true/false function attributes are specified.
  auto trueFnAttr = op.getAttrOfType<FlatSymbolRefAttr>("a_true_fn");
  if (!trueFnAttr)
    return op.emitOpError("requires a 'a_true_fn' symbol reference attribute");

  auto falseFnAttr = op.getAttrOfType<FlatSymbolRefAttr>("b_false_fn");
  if (!falseFnAttr)
    return op.emitOpError("requires a 'a_false_fn' symbol reference attribute");

  auto trueFn = op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      trueFnAttr.getValue());
  if (!trueFn)
    return op.emitOpError() << "'" << trueFnAttr.getValue()
                            << "' does not reference a valid function";

  auto falseFn = op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      falseFnAttr.getValue());
  if (!falseFn)
    return op.emitOpError() << "'" << falseFnAttr.getValue()
                            << "' does not reference a valid function";

  // Verify that the operand and result types match the true/false function.
  auto trueFnType = trueFn.getType();
  if (trueFnType.getNumInputs() != op.getNumOperands() - 1)
    return op.emitOpError("incorrect number of operands for true function");

  auto falseFnType = falseFn.getType();
  if (falseFnType.getNumInputs() != op.getNumOperands() - 1)
    return op.emitOpError("incorrect number of operands for false function");

  for (unsigned i = 0, e = trueFnType.getNumInputs(); i != e; ++i) {
    if (op.getOperand(i + 1).getType() != trueFnType.getInput(i))
      return op.emitOpError("operand type mismatch for true function");

    if (op.getOperand(i + 1).getType() != falseFnType.getInput(i))
      return op.emitOpError("operand type mismatch for false function");
  }

  if (trueFnType.getNumResults() != op.getNumResults())
    return op.emitOpError("incorrect number of results for true function");

  if (falseFnType.getNumResults() != op.getNumResults())
    return op.emitOpError("incorrect number of results for false function");

  for (unsigned i = 0, e = trueFnType.getNumResults(); i != e; ++i) {
    if (op.getResult(i).getType() != trueFnType.getResult(i))
      return op.emitOpError("result type mismatch for true function");

    if (op.getResult(i).getType() != falseFnType.getResult(i))
      return op.emitOpError("result type mismatch for false function");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RepeatI32Op
//===----------------------------------------------------------------------===//

static ParseResult parseRepeatI32Op(OpAsmParser &parser,
                                    OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperandList(operands)) return failure();

  result.addAttribute("bef.nonstrict", parser.getBuilder().getUnitAttr());
  if (succeeded(parser.parseOptionalKeyword("attributes"))) {
    if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  }

  SmallVector<Type, 4> types;
  llvm::SMLoc type_loc = parser.getCurrentLocation();
  if (parser.parseOptionalColonTypeList(types) ||
      parser.addTypesToList(types, result.types))
    return failure();

  if (operands.empty())
    return parser.emitError(parser.getCurrentLocation(), "expected trip count");

  auto loop_operands = llvm::makeArrayRef(operands).drop_front();
  auto i32_type = IntegerType::get(32, result.getContext());

  if (parser.resolveOperand(operands[0], i32_type, result.operands) ||
      parser.resolveOperands(loop_operands, types, type_loc, result.operands))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, loop_operands, types,
                            /*enableNameShadowing=*/true);
}

static void print(OpAsmPrinter &p, RepeatI32Op op) {
  p << "tfrt.repeat.i32 ";
  p.printOperands(op.getOperands());
  if (!op.getAttrs().empty()) {
    p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"bef.nonstrict"});
  }
  if (op.getNumOperands() > 1) {
    p << " : ";
    interleaveComma(llvm::drop_begin(op.getOperandTypes(), 1), p);
  }

  // Reuse the argument names provided to the op for the bbarg names within
  // the region.
  SmallVector<Value, 4> arg_name_values(llvm::drop_begin(op.getOperands(), 1));
  p.shadowRegionArgs(op.region(), arg_name_values);
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
}

static LogicalResult verify(RepeatI32Op op) {
  // Verify that the operand and result types match.
  if (op.getNumResults() != op.getNumOperands() - 1)
    return op.emitOpError("incorrect number of operands");

  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
    if (op.getOperand(i + 1).getType() != op.getResult(i).getType())
      return op.emitOpError("operand/result type mismatch");

  return checkTFRTReturn(op, &op.region(), op.getResultTypes());
}

//===----------------------------------------------------------------------===//
// ParallelForI32Op
//===----------------------------------------------------------------------===//

// Parse tfrt.parallel_for.i32 operation.
//
// Expected format:
//
//   %ch = tfrt.parallel_for.i32 %start to %end fixed %block_size,
//                              %loop_arg0 : !my.type {
//     ... parallel block compute function ...
//     tfrt.return ... : !tfrt.chain
//   }
static ParseResult parseParallelForI32Op(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::OperandType start;
  OpAsmParser::OperandType end;
  OpAsmParser::OperandType block_size;

  // Parse parallel for bounds: %start to %end fixed %block_size
  if (parser.parseOperand(start) || parser.parseKeyword("to") ||
      parser.parseOperand(end) || parser.parseKeyword("fixed") ||
      parser.parseOperand(block_size)) {
    return failure();
  }

  // Parse additional parallel for operands.
  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperandList(operands)) return failure();
  }

  // Parse types for additional operands.
  SmallVector<Type, 4> types;
  llvm::SMLoc type_loc = parser.getCurrentLocation();
  if (parser.parseOptionalColonTypeList(types)) return failure();

  // Resolve parsed parallel for bounds operands ...
  auto i32_type = IntegerType::get(32, result.getContext());
  if (parser.resolveOperand(start, i32_type, result.operands) ||
      parser.resolveOperand(end, i32_type, result.operands) ||
      parser.resolveOperand(block_size, i32_type, result.operands)) {
    return failure();
  }

  // ... and additional body operands.
  parser.resolveOperands(operands, types, type_loc, result.operands);

  // Parallel for returns chain when all parallel blocks are completed.
  auto chain_type = ChainType::get(result.getContext());
  parser.addTypesToList(chain_type, result.types);

  // Parallel for body operands and types.
  SmallVector<OpAsmParser::OperandType, 6> body_operands = {start, end};
  for (auto &operand : operands) body_operands.push_back(operand);
  SmallVector<Type, 6> body_operands_types = {i32_type, i32_type};
  for (auto &type : types) body_operands_types.push_back(type);

  Region *body = result.addRegion();
  return parser.parseRegion(*body, body_operands, body_operands_types,
                            /*enableNameShadowing=*/true);
}

static void print(OpAsmPrinter &p, ParallelForI32Op op) {
  p << "tfrt.parallel_for.i32 ";

  p.printOperand(op.getOperand(0));
  p << " to ";
  p.printOperand(op.getOperand(1));
  p << " fixed ";
  p.printOperand(op.getOperand(2));

  if (op.getNumOperands() > 3) {
    p << ", ";
    p.printOperands(llvm::drop_begin(op.getOperands(), 3));
    p << " : ";
    interleaveComma(llvm::drop_begin(op.getOperandTypes(), 3), p);
  }

  // Reuse the argument names provided to the op for the bbarg names within
  // the region (except block_size argument).
  SmallVector<Value, 4> arg_name_values(op.getOperands());
  arg_name_values.erase(arg_name_values.begin() + 2);

  p.shadowRegionArgs(op.region(), arg_name_values);
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
}

static LogicalResult verify(ParallelForI32Op op) {
  auto *block = &op.getRegion().front();
  if (block->empty() || !isa<ReturnOp>(block->back()))
    return op.emitOpError("expected tfrt.return in body");

  // Synchronous parallel region can have a return op without operands.
  auto return_op = cast<ReturnOp>(block->back());
  if (return_op.getNumOperands() == 0) return success();

  // Otherwise parallel region must return a chain (same result type as
  // tfrt.parallel_for itself).
  return checkTFRTReturn(op, &op.region(), op.getResultTypes());
}

//===----------------------------------------------------------------------===//
// ParallelCallI32Op
//===----------------------------------------------------------------------===//

// Parse tfrt.parallel_call.i32 operation.
//
// Expected format:
//
//   %ch = tfrt.parallel_call.i32 %start to %end fixed %block_size
//         @callee(%loop_arg0) : !my.type
static ParseResult parseParallelCallI32Op(OpAsmParser &parser,
                                          OperationState &result) {
  OpAsmParser::OperandType start;
  OpAsmParser::OperandType end;
  OpAsmParser::OperandType block_size;

  // Parse parallel for bounds: %start to %end fixed %block_size
  if (parser.parseOperand(start) || parser.parseKeyword("to") ||
      parser.parseOperand(end) || parser.parseKeyword("fixed") ||
      parser.parseOperand(block_size)) {
    return failure();
  }

  // Parse callee attribute.
  SymbolRefAttr callee_attr;
  if (parser.parseAttribute(callee_attr, "callee", result.attributes))
    return failure();

  // Parse additional parallel call operands.
  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse types for additional operands.
  SmallVector<Type, 4> types;
  llvm::SMLoc type_loc = parser.getCurrentLocation();
  if (parser.parseOptionalColonTypeList(types)) return failure();

  // Resolve parsed parallel call bounds operands ...
  auto i32_type = IntegerType::get(32, result.getContext());
  if (parser.resolveOperand(start, i32_type, result.operands) ||
      parser.resolveOperand(end, i32_type, result.operands) ||
      parser.resolveOperand(block_size, i32_type, result.operands)) {
    return failure();
  }

  // ... and additional body operands.
  parser.resolveOperands(operands, types, type_loc, result.operands);

  // Parallel for returns chain when all parallel blocks are completed.
  auto chain_type = ChainType::get(result.getContext());
  parser.addTypesToList(chain_type, result.types);

  return success();
}

static void print(OpAsmPrinter &p, ParallelCallI32Op op) {
  p << "tfrt.parallel_call.i32 ";

  p.printOperand(op.getOperand(0));
  p << " to ";
  p.printOperand(op.getOperand(1));
  p << " fixed ";
  p.printOperand(op.getOperand(2));
  p << " ";

  p << op.getAttr("callee");
  p << '(';
  p.printOperands(llvm::drop_begin(op.getOperands(), 3));
  p << ')';

  if (op.getNumOperands() > 3) {
    p << " : ";
    interleaveComma(llvm::drop_begin(op.getOperandTypes(), 3), p);
  }
}

static LogicalResult verify(ParallelCallI32Op op) {
  // Check that the callee attribute was specified.
  auto fnAttr = op.getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return op.emitOpError("requires a 'callee' symbol reference attribute");
  auto fn =
      op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
  if (!fn)
    return op.emitOpError() << "'" << fnAttr.getValue()
                            << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getType();

  // Callee must take start and end indices followed by parallel call operands.
  if (fnType.getNumInputs() != op.getNumOperands() - 1)
    return op.emitOpError("incorrect number of callee operands");

  auto i32_type = IntegerType::get(32, op.getContext());
  for (unsigned i = 0; i != 2; ++i) {
    if (fnType.getInput(i) != i32_type)
      return op.emitOpError("callee must take stard and end indices first");
  }

  for (unsigned i = 2, e = fnType.getNumInputs(); i != e; ++i) {
    if (op.getOperand(i + 1).getType() != fnType.getInput(i))
      return op.emitOpError("operand type mismatch");
  }

  // Callee must have empty results for synchronous body function, or a single
  // chain for an asynchronous body function.
  if (fnType.getNumResults() > 1)
    return op.emitOpError("invalid callee result type");

  if (fnType.getNumResults() == 1) {
    auto chain_type = ChainType::get(op.getContext());

    if (fnType.getResult(0) != chain_type)
      return op.emitOpError("async callee must return a chain");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

static void print(OpAsmPrinter &p, ReturnOp op) {
  p << "tfrt.return";
  if (op.getNumOperands() > 0) {
    p << ' ';
    p.printOperands(op.getOperands());
    p << " : ";
    interleaveComma(op.getOperandTypes(), p);
  }
}

static LogicalResult verify(ReturnOp op) {
  // The parent is often a 'func' but not always.
  auto function = dyn_cast<FuncOp>(op.getParentOp());

  // We allow tfrt.return in arbitrary control flow structures.
  if (!function) return success();

  // The operand number and types must match the function signature.
  auto results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError("has ")
           << op.getNumOperands()
           << " operands, but enclosing function returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (op.getOperand(i).getType() != results[i])
      return op.emitError()
             << "type of return operand " << i << " ("
             << op.getOperand(i).getType()
             << ") doesn't match function result type (" << results[i] << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/basic_kernels/opdefs/basic_kernels_opdefs.cpp.inc"

}  // namespace tfrt
