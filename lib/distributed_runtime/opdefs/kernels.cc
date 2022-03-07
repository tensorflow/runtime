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

#include "tfrt/distributed_runtime/opdefs/kernels.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/core_runtime/opdefs/types.h"
#include "tfrt/distributed_runtime/opdefs/types.h"

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
  addTypes<DistributedContextType, DistributedContextConfigurationType,
           TaskHandleType, RemoteObjectIdType, RemoteExecuteSpecType,
           RemoteChainManagerType, PayloadType>();

  addOperations<
#define GET_OP_LIST
#include "tfrt/distributed_runtime/opdefs/kernels_opdefs.cpp.inc"
      >();
}

static Type GetDistributedContextConfigurationType(Builder *builder) {
  return builder->getType<tfrt::dist::DistributedContextConfigurationType>();
}

LogicalResult CreateConfigurations::inferReturnTypes(
    MLIRContext *ctx, Optional<Location> location, ValueRange operands,
    DictionaryAttr attr, RegionRange ranges,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  CreateConfigurationsAdaptor op(operands, attr, ranges);
  inferredReturnTypes.insert(
      inferredReturnTypes.begin(), op.n(),
      tfrt::dist::DistributedContextConfigurationType::get(ctx));
  return success();
}

mlir::Type DistributedDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (spec == "dist_context") return DistributedContextType::get(getContext());
  if (spec == "dist_context_configuration")
    return DistributedContextConfigurationType::get(getContext());
  if (spec == "task_handle") return TaskHandleType::get(getContext());
  if (spec == "remote_object_id") return RemoteObjectIdType::get(getContext());
  if (spec == "remote_execute_spec")
    return RemoteExecuteSpecType::get(getContext());
  if (spec == "remote_chain_manager")
    return RemoteChainManagerType::get(getContext());
  if (spec == "payload") return PayloadType::get(getContext());

  mlir::Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  mlir::emitError(loc) << "unknown tfrt_dist type " << spec;
  return {};
}

void DistributedDialect::printType(mlir::Type type,
                                   mlir::DialectAsmPrinter &printer) const {
  if (type.isa<DistributedContextType>()) {
    printer << "dist_context";
  } else if (type.isa<DistributedContextConfigurationType>()) {
    printer << "dist_context_configuration";
  } else if (type.isa<TaskHandleType>()) {
    printer << "task_handle";
  } else if (type.isa<RemoteObjectIdType>()) {
    printer << "remote_object_id";
  } else if (type.isa<RemoteExecuteSpecType>()) {
    printer << "remote_execute_spec";
  } else if (type.isa<RemoteChainManagerType>()) {
    printer << "remote_chain_manager";
  } else if (type.isa<PayloadType>()) {
    printer << "payload";
  } else {
    llvm_unreachable("unknown dist type");
  }
}

}  // namespace dist
}  // end namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/distributed_runtime/opdefs/kernels_opdefs.cpp.inc"
