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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
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
  p << "(" << op.op_handler() << ") " << op->getAttr("op_name") << '('
    << op.operands() << ')';

  PrintExecuteOpImpl(p, op);
  PrintExecuteOpFuncAttribute(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}
static void print(OpAsmPrinter &p, ExecuteOpSeq op) {
  p << "(" << op.op_handler() << ", " << op.in_op_chain() << ") "
    << op->getAttr("op_name") << '(' << op.operands() << ')';

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

static LogicalResult VerifyFunctionAttribute(Operation *op, StringRef name,
                                             TypeRange inputTypes,
                                             TypeRange resultTypes) {
  auto attribute = op->getAttrOfType<FlatSymbolRefAttr>(name);
  if (!attribute) {
    return op->emitOpError()
           << "requires a '" << name << "' symbol reference attribute";
  }

  auto function = op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      attribute.getValue());
  if (!function) {
    return op->emitOpError() << "'" << attribute.getValue()
                             << "' does not reference a valid function";
  }

  auto type = function.getType();
  if (inputTypes != type.getInputs()) {
    return op->emitOpError()
           << "'" << attribute.getValue() << "' has mismatching operand types";
  }
  if (resultTypes != type.getResults()) {
    return op->emitOpError()
           << "'" << attribute.getValue() << "' has mismatching result types";
  }

  return success();
}

static LogicalResult verify(CondOp op) {
  auto operand_types = TypeRange(op.getOperandTypes()).drop_front();
  return success(succeeded(VerifyFunctionAttribute(
                     op, "a_true_fn", operand_types, op.getResultTypes())) &&
                 succeeded(VerifyFunctionAttribute(
                     op, "b_false_fn", operand_types, op.getResultTypes())));
}

//===----------------------------------------------------------------------===//
// CoreRt_WhileOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(WhileOp op) {
  return success(
      succeeded(VerifyFunctionAttribute(op, "a_cond_fn", op.getOperandTypes(),
                                        op.getResultTypes())) &&
      succeeded(VerifyFunctionAttribute(op, "b_body_fn", op.getOperandTypes(),
                                        op.getResultTypes())));
}

}  // namespace corert
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/core_runtime/opdefs/core_runtime_opdefs.cpp.inc"
