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

// This file implements MLIR operation functions for the core runtime library.
#include "tfrt/core_runtime/opdefs/core_runtime.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/core_runtime/opdefs/attributes.h"
#include "tfrt/core_runtime/opdefs/corert_utils.h"
#include "tfrt/core_runtime/opdefs/types.h"

namespace tfrt {
namespace corert {

namespace {

struct CoreRTInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *op, Region *dest, bool would_be_cloned,
                       BlockAndValueMapping &) const final {
    // All CoreRT dialect ops can be inlined.
    return true;
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// CoreRTDialect Dialect
//===----------------------------------------------------------------------===//

CoreRTDialect::CoreRTDialect(MLIRContext *context)
    : Dialect(/*name=*/"corert", context, TypeID::get<CoreRTDialect>()) {
  context->getOrLoadDialect<compiler::TFRTDialect>();

  allowUnknownTypes();
  allowUnknownOperations();

  addAttributes<ShapeAttr>();

  addTypes<StringType, TensorHandleType, OpHandlerType, ResourceType,
           VariantType, Quint8Type, Qint8Type, Qint16Type, Qint32Type,
           Quint16Type>();

  addInterfaces<CoreRTInlinerInterface>();

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
  if (data == "ophandler") return OpHandlerType::get(getContext());
  if (data == "tensorhandle") return TensorHandleType::get(getContext());
  if (data == "resource") return ResourceType::get(getContext());
  if (data == "variant") return VariantType::get(getContext());
  if (data == "quint8") return Quint8Type::get(getContext());
  if (data == "qint8") return Qint8Type::get(getContext());
  if (data == "qint16") return Qint16Type::get(getContext());
  if (data == "qint32") return Qint32Type::get(getContext());
  if (data == "quint16") return Quint16Type::get(getContext());

  // TODO(tfrt-devs): Every type should be properly defined. Remove
  // OpaqueType here once all types are defined in corerrt.
  return mlir::OpaqueType::get(mlir::Identifier::get("corert", getContext()),
                               data);
}

void CoreRTDialect::printType(mlir::Type type,
                              mlir::DialectAsmPrinter &os) const {
  if (type.isa<StringType>()) {
    os << "string";
    return;
  }

  if (type.isa<OpHandlerType>()) {
    os << "ophandler";
    return;
  }

  if (type.isa<TensorHandleType>()) {
    os << "tensorhandle";
    return;
  }

  if (type.isa<ResourceType>()) {
    os << "resource";
    return;
  }

  if (type.isa<VariantType>()) {
    os << "variant";
    return;
  }

  if (type.isa<Quint8Type>()) {
    os << "quint8";
    return;
  }

  if (type.isa<Quint16Type>()) {
    os << "quint16";
    return;
  }

  if (type.isa<Qint8Type>()) {
    os << "qint8";
    return;
  }

  if (type.isa<Qint16Type>()) {
    os << "qint16";
    return;
  }

  if (type.isa<Qint32Type>()) {
    os << "qint32";
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

void ExecuteOp::build(OpBuilder &builder, OperationState &state,
                      TypeRange results, Value op_handler, ValueRange operands,
                      ArrayRef<std::pair<StringRef, Attribute>> op_attrs,
                      ArrayRef<std::pair<StringRef, Attribute>> op_func_attrs,
                      StringRef op_name) {
  SmallVector<Attribute, 4> attrs;
  for (const auto &named_attr : op_attrs) {
    auto name = builder.getStringAttr(named_attr.first);
    SmallVector<Attribute, 2> key_value{name, named_attr.second};
    attrs.push_back(ArrayAttr::get(builder.getContext(), key_value));
  }

  SmallVector<Attribute, 4> func_attrs;
  for (const auto &named_attr : op_func_attrs) {
    auto name = builder.getStringAttr(named_attr.first);
    SmallVector<Attribute, 2> key_value{name, named_attr.second};
    func_attrs.push_back(ArrayAttr::get(builder.getContext(), key_value));
  }

  auto attr = ArrayAttr::get(builder.getContext(), attrs);
  auto func_attr = ArrayAttr::get(builder.getContext(), func_attrs);
  build(builder, state, results, op_handler, operands, attr, func_attr,
        op_name);
}

static LogicalResult verify(ExecuteOp op) { return VerifyExecuteOpImpl(op); }
static LogicalResult verify(ExecuteOpSeq op) { return VerifyExecuteOpImpl(op); }

static ParseResult parseExecuteOp(OpAsmParser &parser, OperationState &result) {
  return ParseExecuteOpImpl(parser, result, /*num_chains=*/0,
                            /*has_func_attr=*/true);
}
static ParseResult parseExecuteOpSeq(OpAsmParser &parser,
                                     OperationState &result) {
  return ParseExecuteOpImpl(parser, result, /*num_chains=*/1,
                            /*has_func_attr=*/true);
}
static void print(OpAsmPrinter &p, ExecuteOp op) {
  p << "corert.executeop(" << op.op_handler() << ") " << op->getAttr("op_name")
    << '(' << op.operands() << ')';

  PrintExecuteOpImpl(p, op);
  PrintExecuteOpFuncAttribute(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}
static void print(OpAsmPrinter &p, ExecuteOpSeq op) {
  p << "corert.executeop.seq(" << op.op_handler() << ", " << op.in_op_chain()
    << ") " << op->getAttr("op_name") << '(' << op.operands() << ')';

  PrintExecuteOpImpl(p, op);
  PrintExecuteOpFuncAttribute(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}

void ExecuteOp::getOpAttrs(
    SmallVectorImpl<std::pair<StringRef, Attribute>> *op_attrs) {
  assert(op_attrs);
  op_attrs->clear();
  ArrayRef<Attribute> op_attr_array = this->op_attrs().getValue();

  Builder builder(getContext());
  for (Attribute iter : op_attr_array) {
    ArrayRef<Attribute> key_value = iter.cast<ArrayAttr>().getValue();
    StringRef key = key_value[0].cast<StringAttr>().getValue();
    Attribute value = key_value[1];
    op_attrs->push_back({key, value});
  }
}

void ExecuteOp::getOpFuncAttrs(
    SmallVectorImpl<std::pair<StringRef, Attribute>> *op_func_attrs) {
  assert(op_func_attrs);
  op_func_attrs->clear();
  ArrayRef<Attribute> op_func_attr_array = this->op_func_attrs().getValue();

  Builder builder(getContext());
  for (Attribute iter : op_func_attr_array) {
    ArrayRef<Attribute> key_value = iter.cast<ArrayAttr>().getValue();
    StringRef key = key_value[0].cast<StringAttr>().getValue();
    Attribute value = key_value[1];
    op_func_attrs->push_back({key, value});
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

//===----------------------------------------------------------------------===//
// ConstDenseTensorOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstDenseTensorOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return value();
}

//===----------------------------------------------------------------------===//
// CoreRt_CondOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(CondOp op) {
  // Check that the true/false function attributes are specified.
  auto trueFnAttr = op->getAttrOfType<FlatSymbolRefAttr>("a_true_fn");
  if (!trueFnAttr)
    return op.emitOpError("requires a 'a_true_fn' symbol reference attribute");

  auto falseFnAttr = op->getAttrOfType<FlatSymbolRefAttr>("b_false_fn");
  if (!falseFnAttr)
    return op.emitOpError("requires a 'a_false_fn' symbol reference attribute");

  auto trueFn = op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      trueFnAttr.getValue());
  if (!trueFn)
    return op.emitOpError() << "'" << trueFnAttr.getValue()
                            << "' does not reference a valid function";

  auto falseFn = op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
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
// CoreRt_WhileOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(WhileOp op) {
  // Check that the cond and body function attributes are specified.
  auto condFnAttr = op->getAttrOfType<FlatSymbolRefAttr>("a_cond_fn");
  if (!condFnAttr)
    return op.emitOpError("requires a 'a_cond_fn' symbol reference attribute");

  auto bodyFnAttr = op->getAttrOfType<FlatSymbolRefAttr>("b_body_fn");
  if (!bodyFnAttr)
    return op.emitOpError("requires a 'b_body_fn' symbol reference attribute");

  auto condFn = op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      condFnAttr.getValue());
  if (!condFn)
    return op.emitOpError() << "'" << condFnAttr.getValue()
                            << "' does not reference a valid function";

  auto bodyFn = op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      bodyFnAttr.getValue());
  if (!bodyFn)
    return op.emitOpError() << "'" << bodyFnAttr.getValue()
                            << "' does not reference a valid function";

  // Verify the operand and result types of the cond and body functions.
  auto condFnType = condFn.getType();
  if (condFnType.getNumInputs() != op.getNumOperands())
    return op.emitOpError(
        llvm::formatv("incorrect number of operands for cond function: WhileOp "
                      "has {0} operands but cond function has {1} operands",
                      op.getNumOperands(), condFnType.getNumInputs()));

  auto bodyFnType = bodyFn.getType();
  if (bodyFnType.getNumInputs() != op.getNumOperands())
    return op.emitOpError(
        llvm::formatv("incorrect number of operands for body function: WhileOp "
                      "has {0} operands but body function has {1} operands",
                      op.getNumOperands(), bodyFnType.getNumInputs()));

  for (unsigned i = 0, e = condFnType.getNumInputs(); i != e; ++i) {
    if (op.getOperand(i).getType() != condFnType.getInput(i))
      return op.emitOpError("operand type mismatch for cond function");

    if (op.getOperand(i).getType() != bodyFnType.getInput(i))
      return op.emitOpError("operand type mismatch for body function");
  }

  if (bodyFnType.getNumResults() != op.getNumResults())
    return op.emitOpError(
        llvm::formatv("incorrect number of results for body function: WhileOp "
                      "has {0} results but body function has {1} results",
                      op.getNumResults(), bodyFnType.getNumResults()));

  for (unsigned i = 0, e = bodyFnType.getNumResults(); i != e; ++i) {
    if (op.getResult(i).getType() != bodyFnType.getResult(i))
      return op.emitOpError("result type mismatch for body function");
  }

  return success();
}

}  // namespace corert
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/core_runtime/opdefs/core_runtime_opdefs.cpp.inc"
