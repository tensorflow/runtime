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

// This file implements MLIR operations for the tpu_ops library.

#include "tfrt/tpu/opdefs/tpu_ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/core_runtime/opdefs/types.h"

namespace tfrt {
namespace tpu {

//===----------------------------------------------------------------------===//
// TpuRuntimeDialect Dialect
//===----------------------------------------------------------------------===//

TpuRuntimeDialect::TpuRuntimeDialect(MLIRContext *context)
    : Dialect(/*name*/ "tpurt", context, TypeID::get<TpuRuntimeDialect>()) {
  allowUnknownTypes();
  allowUnknownOperations();

  addOperations<
#define GET_OP_LIST
#include "tfrt/tpu/opdefs/tpu_ops_opdefs.cpp.inc"
      >();
}

static Type GetStringType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("hex"), "string");
}

static Type GetChainType(Builder *builder) {
  return builder->getType<ChainType>();
}

static Type GetTensorHandleType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("corert"), "tensor_handle");
}

static Type GetDenseTensorType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tpurt"), "dense_tensor");
}

static Type GetCompilationResultType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tpurt"), "compilation_result");
}

static Type GetLoadedProgramType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tpurt"), "loaded_program");
}

static Type GetXlaShapeType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tpurt"), "xla_shape");
}

static Type GetCoreLocationType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tpurt"), "core_location");
}

static Type GetSystemType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tpurt"), "system");
}

static Type GetResourceManagerType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tpurt"), "resource_manager");
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

// tpurt.compile
static ParseResult parseCompileOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto loc = parser.getNameLoc();
  auto compilation_result_type = GetCompilationResultType(&builder);
  SmallVector<OpAsmParser::OperandType, 4> operands;
  StringAttr metadata;
  StringAttr mlir_module;
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("metadata") || parser.parseEqual() ||
      parser.parseAttribute(metadata, "metadata", result.attributes) ||
      parser.parseKeyword("mlir_module") || parser.parseEqual() ||
      parser.parseAttribute(mlir_module, "mlir_module", result.attributes))
    return failure();

  SmallVector<mlir::Type, 4> operand_types;
  operand_types.push_back(builder.getType<tfrt::ChainType>());
  for (int i = 1; i < operands.size(); ++i) {
    operand_types.push_back(builder.getType<tfrt::corert::TensorHandleType>());
  }
  if (parser.resolveOperands(operands, operand_types, loc, result.operands))
    return failure();

  result.types.push_back(compilation_result_type);

  return success();
}

static void print(OpAsmPrinter &p, CompileOp op) {
  p << "tpurt.compile (" << op.getOperation()->getOperands() << ") "
    << "metadata = " << op->getAttr("metadata") << " "
    << "mlir_module = " << op->getAttr("mlir_module");
}

// tpurt.compile_v2
static ParseResult parseCompileV2Op(OpAsmParser &parser,
                                    OperationState &result) {
  auto &builder = parser.getBuilder();
  auto compilation_result_type = GetCompilationResultType(&builder);
  auto resource_manager_type = GetResourceManagerType(&builder);
  auto tensor_handle_type = GetTensorHandleType(&builder);

  OpAsmParser::OperandType resource_manager;
  SmallVector<OpAsmParser::OperandType, 4> dynamic_shapes;
  StringAttr metadata;
  StringAttr mlir_module;
  if (parser.parseOperand(resource_manager) ||
      parser.parseOperandList(dynamic_shapes, OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("metadata") || parser.parseEqual() ||
      parser.parseAttribute(metadata, "metadata", result.attributes) ||
      parser.parseKeyword("mlir_module") || parser.parseEqual() ||
      parser.parseAttribute(mlir_module, "mlir_module", result.attributes))
    return failure();

  if (parser.resolveOperand(resource_manager, resource_manager_type,
                            result.operands) ||
      parser.resolveOperands(dynamic_shapes, tensor_handle_type,
                             result.operands))
    return failure();

  result.types.push_back(compilation_result_type);

  return success();
}

static void print(OpAsmPrinter &p, CompileV2Op op) {
  p << "tpurt.compile_v2 " << op.resource_manager() << " ("
    << op.dynamic_shapes() << ") "
    << "metadata = " << op->getAttr("metadata") << " "
    << "mlir_module = " << op->getAttr("mlir_module");
}

// tpurt.execute
static ParseResult parseExecuteOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto core_location_type = GetCoreLocationType(&builder);
  auto dense_tensor_type = GetDenseTensorType(&builder);
  auto loaded_program_type = GetLoadedProgramType(&builder);

  OpAsmParser::OperandType loaded_program;
  OpAsmParser::OperandType core_location;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperand(loaded_program) || parser.parseComma() ||
      parser.parseOperand(core_location) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren))
    return failure();

  int64_t num_results = 0;
  if (succeeded(parser.parseOptionalColon())) {
    IntegerAttr attr;
    mlir::NamedAttrList attrs;
    if (failed(parser.parseAttribute(attr, "num_results", attrs)))
      return failure();
    num_results = attr.getValue().getSExtValue();
  }

  if (parser.resolveOperand(loaded_program, loaded_program_type,
                            result.operands) ||
      parser.resolveOperand(core_location, core_location_type,
                            result.operands) ||
      parser.resolveOperands(operands, dense_tensor_type, result.operands))
    return failure();

  result.types.append(num_results, dense_tensor_type);

  return success();
}

static void print(OpAsmPrinter &p, ExecuteOp op) {
  p << "tpurt.execute " << op.loaded_program() << ", " << op.core_location()
    << " (" << op.operands() << ") : " << op.getNumResults();
}

// tpurt.execute_v2
static ParseResult parseExecuteV2Op(OpAsmParser &parser,
                                    OperationState &result) {
  auto &builder = parser.getBuilder();
  auto system_type = GetSystemType(&builder);
  auto resource_manager_type = GetResourceManagerType(&builder);
  auto core_location_type = GetCoreLocationType(&builder);
  auto dense_tensor_type = GetDenseTensorType(&builder);
  auto loaded_program_type = GetLoadedProgramType(&builder);

  OpAsmParser::OperandType loaded_program;
  OpAsmParser::OperandType system;
  OpAsmParser::OperandType resource_manager;
  OpAsmParser::OperandType core_location;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperand(system) || parser.parseComma() ||
      parser.parseOperand(resource_manager) || parser.parseComma() ||
      parser.parseOperand(loaded_program) || parser.parseComma() ||
      parser.parseOperand(core_location) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren))
    return failure();

  int64_t num_results = 0;
  if (succeeded(parser.parseOptionalColon())) {
    IntegerAttr attr;
    mlir::NamedAttrList attrs;
    if (failed(parser.parseAttribute(attr, "num_results", attrs)))
      return failure();
    num_results = attr.getValue().getSExtValue();
  }

  if (parser.resolveOperand(system, system_type, result.operands) ||
      parser.resolveOperand(resource_manager, resource_manager_type,
                            result.operands) ||
      parser.resolveOperand(loaded_program, loaded_program_type,
                            result.operands) ||
      parser.resolveOperand(core_location, core_location_type,
                            result.operands) ||
      parser.resolveOperands(operands, dense_tensor_type, result.operands))
    return failure();

  result.types.append(num_results, dense_tensor_type);

  return success();
}

static void print(OpAsmPrinter &p, ExecuteV2Op op) {
  p << "tpurt.execute_v2 " << op.system() << ", " << op.resource_manager()
    << ", " << op.loaded_program() << ", " << op.core_location() << " ("
    << op.operands() << ") : " << op.getNumResults();
}

// tpurt.get_input_layout
static ParseResult parseGetInputLayoutOp(OpAsmParser &parser,
                                         OperationState &result) {
  auto &builder = parser.getBuilder();
  auto compilation_result_type = GetCompilationResultType(&builder);
  auto xla_shape_type = GetXlaShapeType(&builder);

  OpAsmParser::OperandType compilation_result;
  if (parser.parseOperand(compilation_result)) return failure();

  int64_t num_results = 0;
  if (succeeded(parser.parseOptionalColon())) {
    IntegerAttr attr;
    mlir::NamedAttrList attrs;
    if (failed(parser.parseAttribute(attr, "num_results", attrs)))
      return failure();
    num_results = attr.getValue().getSExtValue();
  }

  if (parser.resolveOperand(compilation_result, compilation_result_type,
                            result.operands))
    return failure();

  result.types.append(num_results, xla_shape_type);

  return success();
}

static void print(OpAsmPrinter &p, GetInputLayoutOp op) {
  p << "tpurt.get_input_layout " << op.compilation_result() << " : "
    << op.getNumResults();
}

}  // namespace tpu
}  // end namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/tpu/opdefs/tpu_ops_opdefs.cpp.inc"
