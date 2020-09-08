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

//===-_kernels.cc ---------------------------------------------------------===//
//
// This file implements MLIR operations for the distributed dialect.
//
//===----------------------------------------------------------------------===//

#include "tfrt/distributed_runtime/opdefs/kernels.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"

namespace tfrt {
namespace dist {

//===----------------------------------------------------------------------===//
// Distributed Dialect
//===----------------------------------------------------------------------===//

DistributedDialect::DistributedDialect(MLIRContext *context)
    : Dialect(/*name=*/"dist", context, TypeID::get<DistributedDialect>()) {
  allowUnknownTypes();
  allowUnknownOperations();
  addOperations<
#define GET_OP_LIST
#include "tfrt/distributed_runtime/opdefs/kernels_opdefs.cpp.inc"
      >();
}

static Type GetContextType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("dist"), "dist_context",
                         builder->getContext());
}

static Type GetChainType(Builder *builder) {
  return builder->getType<ChainType>();
}
static Type GetDistributedContextConfigurationType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("dist"),
                         "dist_context_configuration", builder->getContext());
}
static Type GetRemoteObjectIdType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("dist"), "remote_object_id",
                         builder->getContext());
}

static void print(OpAsmPrinter &p, RemoteExecuteOp op) {
  p << "dist.remote_execute(" << op.getAttr("hostid") << ") "
    << op.getAttr("program_name");
}

static void print(OpAsmPrinter &p, RemoteRegisterOp op) {
  p << "dist.remote_register(" << op.getAttr("hostid") << ") "
    << op.getAttr("program_name");
}

static ParseResult parseCreateConfigurations(OpAsmParser &parser,
                                             OperationState &result) {
  auto &builder = parser.getBuilder();

  int64_t num_results = 0;
  if (succeeded(parser.parseOptionalColon())) {
    IntegerAttr attr;
    mlir::NamedAttrList attrs;
    if (failed(parser.parseAttribute(attr, "num_results", attrs)))
      return failure();
    num_results = attr.getValue().getSExtValue();
  }
  auto configuration_type = GetDistributedContextConfigurationType(&builder);

  result.types.append(num_results, configuration_type);

  return success();
}

static ParseResult parseRemoteRegisterOp(OpAsmParser &parser,
                                         OperationState &result) {
  auto &builder = parser.getBuilder();
  StringAttr program_type;

  auto chain_type = GetChainType(&builder);

  SmallVector<OpAsmParser::OperandType, 4> chain_context_and_hostid;
  if (parser.parseOperandList(chain_context_and_hostid, 3,
                              OpAsmParser::Delimiter::Paren)) {
    return failure();
  }
  SmallVector<Type, 4> operand_types;
  operand_types.push_back(chain_type);
  operand_types.push_back(GetContextType(&builder));
  operand_types.push_back(builder.getI32Type());
  if (parser.resolveOperands(chain_context_and_hostid, operand_types,
                             parser.getNameLoc(), result.operands))
    return failure();

  if (parser.parseAttribute(program_type, "program", result.attributes)) {
    return failure();
  }
  if (parser.parseAttribute(program_type, "program_name", result.attributes)) {
    return failure();
  }

  result.types.append(1, chain_type);

  return success();
}

static ParseResult parseRemoteExecuteOp(OpAsmParser &parser,
                                        OperationState &result) {
  auto &builder = parser.getBuilder();
  StringAttr program_type;

  auto chain_type = GetChainType(&builder);

  SmallVector<OpAsmParser::OperandType, 4> chain_context_and_hostid;
  if (parser.parseOperandList(chain_context_and_hostid, 3,
                              OpAsmParser::Delimiter::Paren)) {
    return failure();
  }
  SmallVector<Type, 4> operand_types;
  operand_types.push_back(chain_type);
  operand_types.push_back(GetContextType(&builder));
  operand_types.push_back(builder.getI32Type());
  if (parser.resolveOperands(chain_context_and_hostid, operand_types,
                             parser.getNameLoc(), result.operands))
    return failure();

  if (parser.parseAttribute(program_type, "program_name", result.attributes)) {
    return failure();
  }

  auto remote_object_id_type = GetRemoteObjectIdType(&builder);

  SmallVector<OpAsmParser::OperandType, 4> inputs;
  if (parser.parseOperandList(inputs, -1, OpAsmParser::Delimiter::Paren)) {
    return failure();
  }
  if (parser.resolveOperands(inputs, remote_object_id_type, result.operands))
    return failure();

  result.types.append(1, chain_type);

  int64_t num_results = 0;
  if (succeeded(parser.parseOptionalColon())) {
    IntegerAttr attr;
    mlir::NamedAttrList attrs;
    if (failed(parser.parseAttribute(attr, "num_results", attrs)))
      return failure();
    num_results = attr.getValue().getSExtValue();
  }

  result.types.append(num_results, remote_object_id_type);
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/distributed_runtime/opdefs/kernels_opdefs.cpp.inc"

}  // namespace dist
}  // end namespace tfrt
