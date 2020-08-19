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

//===- core_runtime.cc ----------------------------------------------------===//
//
// This file implements MLIR operation functions for the core runtime library.
//
//===----------------------------------------------------------------------===//
#include "tfrt/core_runtime/opdefs/core_runtime.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/core_runtime/opdefs/attributes.h"
#include "tfrt/core_runtime/opdefs/types.h"

namespace tfrt {
namespace corert {

//===----------------------------------------------------------------------===//
// CoreRTDialect Dialect
//===----------------------------------------------------------------------===//

CoreRTDialect::CoreRTDialect(MLIRContext *context)
    : Dialect(/*name=*/"corert", context, TypeID::get<CoreRTDialect>()) {
  allowUnknownTypes();
  allowUnknownOperations();

  addAttributes<ShapeAttr>();

  addTypes<StringType, TensorHandleType, DeviceType>();

  addOperations<
#define GET_OP_LIST
#include "tfrt/core_runtime/opdefs/core_runtime_opdefs.cpp.inc"
      >();
}

namespace {

ShapeAttr ParseShapeAttr(MLIRContext *context, llvm::StringRef spec,
                         Location loc) {
  auto emit_error = [&, spec]() {
    mlir::emitError(loc, "unknown corert shape attribute: ") << spec;
  };

  if (!spec.consume_front("shape<")) return (emit_error(), nullptr);

  if (spec.consume_front("*>")) return ShapeAttr::get(context);

  SmallVector<int64_t, 4> shape;
  while (!spec.consume_front(">")) {
    int64_t dim;

    if (spec.consume_front("?"))
      dim = -1;
    else if (spec.consumeInteger(10, dim))
      return (emit_error(), nullptr);

    spec.consume_front("x");

    shape.push_back(dim);
  }

  return ShapeAttr::get(context, shape);
}

void PrintShapeAttr(ShapeAttr attr, mlir::DialectAsmPrinter &os) {  // NOLINT
  os << "shape";

  os << "<";
  if (attr.hasRank()) {
    auto print_dim = [&](int64_t dim) {
      if (dim > -1)
        os << dim;
      else
        os << "?";
    };
    llvm::interleave(attr.getShape(), os, print_dim, "x");
  } else {
    os << "*";
  }
  os << ">";
}

}  // namespace

mlir::Type CoreRTDialect::parseType(mlir::DialectAsmParser &parser) const {
  StringRef data;
  if (parser.parseKeyword(&data)) return Type();

  if (data == "string") return StringType::get(getContext());
  if (data == "device") return DeviceType::get(getContext());
  if (data == "tensorhandle") return TensorHandleType::get(getContext());

  // TODO(tfrt-devs): Every type should be properly defined. Remove
  // OpaqueType here once all types are defined in corerrt.
  return mlir::OpaqueType::get(mlir::Identifier::get("corert", getContext()),
                               data, getContext());
}

void CoreRTDialect::printType(mlir::Type type,
                              mlir::DialectAsmPrinter &os) const {
  if (type.isa<StringType>()) {
    os << "string";
    return;
  }

  if (type.isa<DeviceType>()) {
    os << "device";
    return;
  }

  if (type.isa<TensorHandleType>()) {
    os << "tensorhandle";
    return;
  }

  llvm_unreachable("unexpected corert type kind");
}

mlir::Attribute CoreRTDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                              mlir::Type type) const {
  auto spec = parser.getFullSymbolSpec();
  auto loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  if (spec.startswith("shape")) return ParseShapeAttr(getContext(), spec, loc);

  return (mlir::emitError(loc, "unknown corert attribute: ") << spec, nullptr);
}

void CoreRTDialect::printAttribute(mlir::Attribute attr,
                                   mlir::DialectAsmPrinter &os) const {
  if (auto shape_attr = attr.dyn_cast<ShapeAttr>())
    PrintShapeAttr(attr.cast<ShapeAttr>(), os);
  else
    llvm_unreachable("unexpected corert attribute kind");
}

Operation *CoreRTDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  if (auto dense_attr = value.dyn_cast<DenseElementsAttr>())
    return builder.create<ConstDenseTensorOp>(loc, type, dense_attr);

  return nullptr;
}

static Type GetDeviceType(Builder *builder) {
  return builder->getType<DeviceType>();
}

static Type GetChainType(Builder *builder) {
  return builder->getType<ChainType>();
}

static Type GetTensorHandleType(Builder *builder) {
  return builder->getType<TensorHandleType>();
}

template <typename OpTy>
LogicalResult VerifyExecuteOpImpl(OpTy op) {
  auto op_attr_array = op.op_attrs().getValue();
  for (auto op_attr : op_attr_array) {
    auto key_value = op_attr.template dyn_cast<ArrayAttr>();
    if (!key_value || key_value.getValue().size() != 2 ||
        !key_value.getValue()[0].template isa<StringAttr>())
      return op.emitOpError() << "each op_attr should be a key-value pair, "
                                 "where the key is a string";
  }
  return success();
}

void ExecuteOp::build(OpBuilder &builder, OperationState &state,
                      ArrayRef<Type> results, Value device, ValueRange operands,
                      ArrayRef<std::pair<StringRef, Attribute>> op_attrs,
                      StringRef op_name) {
  SmallVector<Attribute, 4> attrs;
  for (const auto &named_attr : op_attrs) {
    auto name = builder.getStringAttr(named_attr.first);
    SmallVector<Attribute, 2> key_value{name, named_attr.second};
    attrs.push_back(ArrayAttr::get(key_value, builder.getContext()));
  }
  auto attr = ArrayAttr::get(attrs, builder.getContext());
  build(builder, state, results, device, operands, attr, op_name);
}

static LogicalResult verify(ExecuteOp op) { return VerifyExecuteOpImpl(op); }
static LogicalResult verify(ExecuteOpSeq op) { return VerifyExecuteOpImpl(op); }

static ParseResult ParseExecuteOpImpl(OpAsmParser &parser,
                                      OperationState &result, int num_chains) {
  auto &builder = parser.getBuilder();
  auto device_type = GetDeviceType(&builder);
  auto chain_type = GetChainType(&builder);
  auto tensorhandle_type = GetTensorHandleType(&builder);

  StringAttr op_name;
  SmallVector<OpAsmParser::OperandType, 4> device_and_in_chains;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  NamedAttrList op_attrs;
  auto loc = parser.getNameLoc();
  if (parser.parseOperandList(device_and_in_chains,
                              /*requiredOperandCount=*/num_chains + 1,
                              OpAsmParser::Delimiter::Paren) ||
      parser.parseAttribute(op_name, "op_name", result.attributes) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(op_attrs))
    return failure();

  int64_t num_results = 0;
  if (succeeded(parser.parseOptionalColon())) {
    IntegerAttr attr;
    mlir::NamedAttrList attrs;
    if (failed(parser.parseAttribute(attr, "num_results", attrs)))
      return failure();
    num_results = attr.getValue().getSExtValue();
  }

  SmallVector<Type, 4> operand_types;
  operand_types.push_back(device_type);
  operand_types.append(num_chains, chain_type);
  if (parser.resolveOperands(device_and_in_chains, operand_types, loc,
                             result.operands) ||
      parser.resolveOperands(operands, tensorhandle_type, result.operands))
    return failure();

  result.types.append(num_chains, chain_type);
  result.types.append(num_results, tensorhandle_type);

  SmallVector<Attribute, 4> op_attr_array;
  for (const auto &key_value : op_attrs) {
    auto key = builder.getStringAttr(key_value.first.strref());
    auto value = key_value.second;
    op_attr_array.push_back(builder.getArrayAttr({key, value}));
  }

  result.attributes.push_back(
      builder.getNamedAttr("op_attrs", builder.getArrayAttr(op_attr_array)));

  return success();
}
static ParseResult parseExecuteOp(OpAsmParser &parser, OperationState &result) {
  return ParseExecuteOpImpl(parser, result, /*num_chains=*/0);
}
static ParseResult parseExecuteOpSeq(OpAsmParser &parser,
                                     OperationState &result) {
  // ExecuteOpSeq is nonstrict.
  result.addAttribute("bef.nonstrict", parser.getBuilder().getUnitAttr());
  return ParseExecuteOpImpl(parser, result, /*num_chains=*/1);
}

template <typename OpTy>
void PrintExecuteOpImpl(OpAsmPrinter &p, OpTy op) {
  auto op_attrs = op.op_attrs();
  if (!op_attrs.empty()) {
    auto print_key_value = [&](mlir::Attribute attr) {
      auto key_value = attr.cast<ArrayAttr>().getValue();
      auto key = key_value[0];
      auto value = key_value[1];

      p << key.cast<StringAttr>().getValue();
      p << " = ";
      p << value;
    };

    auto op_attr_array = op_attrs.getValue();
    p << " {";
    interleaveComma(op_attr_array, p, print_key_value);
    p << '}';
  }
  if (!op.results().empty()) p << " : " << op.results().size();
}
static void print(OpAsmPrinter &p, ExecuteOp op) {
  p << "corert.executeop(" << op.device() << ") " << op.getAttr("op_name")
    << '(' << op.operands() << ')';

  PrintExecuteOpImpl(p, op);
}
static void print(OpAsmPrinter &p, ExecuteOpSeq op) {
  p << "corert.executeop.seq(" << op.device() << ", " << op.in_op_chain()
    << ") " << op.getAttr("op_name") << '(' << op.operands() << ')';

  PrintExecuteOpImpl(p, op);
}

void ExecuteOp::getOpAttrs(
    SmallVectorImpl<std::pair<StringRef, Attribute>> *op_attrs) {
  assert(op_attrs);
  op_attrs->clear();
  auto op_attr_array = this->op_attrs().getValue();

  Builder builder(getContext());
  for (auto iter : op_attr_array) {
    auto key_value = iter.cast<ArrayAttr>().getValue();
    StringRef key = key_value[0].cast<StringAttr>().getValue();
    Attribute value = key_value[1];
    op_attrs->push_back({key, value});
  }
}

LogicalResult ExecuteOp::fold(ArrayRef<Attribute> operands,
                              SmallVectorImpl<OpFoldResult> &results) {
  if (op_name() == "tf.Const") {
    auto op_attr_array = op_attrs().getValue();
    assert(!op_attr_array.empty());
    for (auto attr : op_attr_array) {
      auto key_value = attr.cast<ArrayAttr>().getValue();
      assert(key_value.size() == 2);
      if (key_value[0].cast<StringAttr>().getValue() == "value") {
        results.push_back(key_value[1]);
        return success();
      }
    }
  }

  return failure();
}

OpFoldResult ConstDenseTensorOp::fold(ArrayRef<Attribute> operands) {
  return value();
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
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/core_runtime/opdefs/core_runtime_opdefs.cpp.inc"

}  // namespace corert
}  // namespace tfrt
