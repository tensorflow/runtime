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
#include "tfrt/core_runtime/opdefs/types.h"

namespace tfrt {
namespace dist {

//===----------------------------------------------------------------------===//
// Distributed Dialect
//===----------------------------------------------------------------------===//

DistributedDialect::DistributedDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfrt_dist", context,
              TypeID::get<DistributedDialect>()) {
  allowUnknownTypes();
  allowUnknownOperations();
  addOperations<
#define GET_OP_LIST
#include "tfrt/distributed_runtime/opdefs/kernels_opdefs.cpp.inc"
      >();
}

static Type GetContextType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tfrt_dist"), "dist_context",
                         builder->getContext());
}

static Type GetStringType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tfrt"), "string",
                         builder->getContext());
}
static Type GetRemoteExecuteSpecType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tfrt_dist"),
                         "remote_execute_spec", builder->getContext());
}

static Type GetChainType(Builder *builder) {
  return builder->getType<ChainType>();
}
static Type GetDistributedContextConfigurationType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tfrt_dist"),
                         "dist_context_configuration", builder->getContext());
}
static Type GetRemoteObjectIdType(Builder *builder) {
  return OpaqueType::get(builder->getIdentifier("tfrt_dist"),
                         "remote_object_id", builder->getContext());
}

static void print(OpAsmPrinter &p, CreateRemoteExecuteSpecOp op) {
  p << "tfrt_dist.create_remote_execute_spec(" << op.output_devices() << ")";
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

static ParseResult parseCreateRemoteExecuteSpecOp(OpAsmParser &parser,
                                                  OperationState &result) {
  auto &builder = parser.getBuilder();

  SmallVector<OpAsmParser::OperandType, 4> inputs;
  if (parser.parseOperandList(inputs, -1, OpAsmParser::Delimiter::Paren)) {
    return failure();
  }
  if (parser.resolveOperands(inputs, GetStringType(&builder), result.operands))
    return failure();

  result.types.append(1, GetRemoteExecuteSpecType(&builder));

  return success();
}

}  // namespace dist
}  // end namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/distributed_runtime/opdefs/kernels_opdefs.cpp.inc"
