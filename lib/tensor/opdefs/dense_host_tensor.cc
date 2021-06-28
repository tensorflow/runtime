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

// This file implements MLIR operations for the dense host tensor dialect.

#include "tfrt/tensor/opdefs/dense_host_tensor.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/tensor/opdefs/host_tensor.h"
#include "tfrt/tensor/opdefs/tensor.h"

namespace tfrt {
namespace dht {

//===----------------------------------------------------------------------===//
// DenseHostTensor Dialect
//===----------------------------------------------------------------------===//

DenseHostTensorDialect::DenseHostTensorDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfrt_dht", context,
              TypeID::get<DenseHostTensorDialect>()) {
  context->getOrLoadDialect<compiler::TFRTDialect>();
  context->getOrLoadDialect<tfrt::t::TensorDialect>();
  context->getOrLoadDialect<tfrt::ht::HostTensorDialect>();

  allowUnknownTypes();
  allowUnknownOperations();
  addOperations<
#define GET_OP_LIST
#include "tfrt/tensor/opdefs/dense_host_tensor.cpp.inc"
      >();
}

static Type getChainType(mlir::MLIRContext *context) {
  return compiler::ChainType::get(context);
}

static Type getTensorType(mlir::MLIRContext *context) {
  return tfrt::t::TensorType::get(context);
}

//===----------------------------------------------------------------------===//
// CreateUnitializedTensorOp
//===----------------------------------------------------------------------===//

static ParseResult parseCreateUninitTensorOp(OpAsmParser &parser,
                                             OperationState &result) {
  // Shape is a list of i64.
  Type attrType = IntegerType::get(result.getContext(), 64);
  auto tensorType = getTensorType(result.getContext());

  Attribute valueAttr;
  return failure(
      parser.parseAttribute(valueAttr, attrType, "shape", result.attributes) ||
      parser.addTypeToList(tensorType, result.types));
}

template <typename CreateUninitTensorOp>
static void printCreateUninitTensorOp(OpAsmPrinter &p,
                                      CreateUninitTensorOp op) {
  p << CreateUninitTensorOp::getOperationName() << " " << op->getAttr("shape");
}

//===----------------------------------------------------------------------===//
// FillTensorOp
//===----------------------------------------------------------------------===//

static ParseResult parseFillTensorOp(OpAsmParser &parser,
                                     OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2))
    return failure();

  auto tensorType = getTensorType(result.getContext());
  auto chainType = getChainType(result.getContext());

  Attribute valueAttr;
  return failure(
      parser.resolveOperand(operands[0], tensorType, result.operands) ||
      parser.resolveOperand(operands[1], chainType, result.operands) ||
      parser.parseAttribute(valueAttr, "value", result.attributes) ||
      parser.addTypeToList(chainType, result.types));
}

template <typename FillTensorOp>
static void printFillTensorOp(OpAsmPrinter &p, FillTensorOp op) {
  p << FillTensorOp::getOperationName() << " ";
  p.printOperands(op.getOperands());
  p << " " << op->getAttr("value");
}

//===----------------------------------------------------------------------===//
// SetTensorOp
//===----------------------------------------------------------------------===//

static ParseResult parseSetTensorOp(OpAsmParser &parser,
                                    OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2))
    return failure();

  auto tensorType = getTensorType(result.getContext());
  auto chainType = getChainType(result.getContext());

  Attribute valueAttr;
  return failure(
      parser.resolveOperand(operands[0], tensorType, result.operands) ||
      parser.resolveOperand(operands[1], chainType, result.operands) ||
      parser.parseAttribute(valueAttr, "values", result.attributes) ||
      parser.addTypeToList(chainType, result.types));
}

template <typename SetTensorOp>
static void printSetTensorOp(OpAsmPrinter &p, SetTensorOp op) {
  p << SetTensorOp::getOperationName() << " ";
  p.printOperands(op.getOperands());
  p << " " << op->getAttr("values");
}
}  // namespace dht
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/tensor/opdefs/dense_host_tensor.cpp.inc"
