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

// This file implements MLIR operation functions for 'corert_sync' dialect.
#include "tfrt/core_runtime/opdefs/sync/core_runtime.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tfrt/core_runtime/opdefs/core_runtime.h"
#include "tfrt/core_runtime/opdefs/corert_utils.h"
#include "tfrt/core_runtime/opdefs/types.h"

namespace tfrt {
namespace corert_sync {

//===----------------------------------------------------------------------===//
// CoreRTSyncDialect Dialect
//===----------------------------------------------------------------------===//

CoreRTSyncDialect::CoreRTSyncDialect(MLIRContext *context)
    : Dialect(/*name=*/"corert_sync", context,
              TypeID::get<CoreRTSyncDialect>()) {
  allowUnknownTypes();
  allowUnknownOperations();

  context->getOrLoadDialect<corert::CoreRTDialect>();

  addOperations<
#define GET_OP_LIST
#include "tfrt/core_runtime/opdefs/sync/core_runtime_opdefs.cpp.inc"
      >();
}

Operation *CoreRTSyncDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  if (auto dense_attr = value.dyn_cast<DenseElementsAttr>())
    return builder.create<ConstDenseTensorOp>(loc, type, dense_attr);

  return nullptr;
}

void ExecuteOp::build(OpBuilder &builder, OperationState &state,
                      TypeRange results, Value op_handler, ValueRange operands,
                      ArrayRef<std::pair<StringRef, Attribute>> op_attrs,
                      StringRef op_name) {
  SmallVector<Attribute, 4> attrs;
  for (const auto &named_attr : op_attrs) {
    auto name = builder.getStringAttr(named_attr.first);
    SmallVector<Attribute, 2> key_value{name, named_attr.second};
    attrs.push_back(ArrayAttr::get(builder.getContext(), key_value));
  }
  auto attr = ArrayAttr::get(builder.getContext(), attrs);
  build(builder, state, results, op_handler, operands, attr, op_name);
}

LogicalResult ExecuteOp::verify() {
  ExecuteOp op = *this;
  return corert::VerifyExecuteOpImpl(op);
}

ParseResult ExecuteOp::parse(OpAsmParser &parser, OperationState &result) {
  return corert::ParseExecuteOpImpl(parser, result, /*num_chains=*/0,
                                    /*has_func_attr=*/false);
}

void ExecuteOp::print(OpAsmPrinter &p) {
  p << "corert_sync.executeop(" << op_handler() << ") "
    << (*this)->getAttr("op_name") << '(' << operands() << ')';

  corert::PrintExecuteOpImpl(p, *this);
  if (!results().empty()) p << " : " << results().size();
}

void ExecuteOp::getOpAttrs(
    llvm::SmallVectorImpl<std::pair<StringRef, Attribute>> *op_attrs) {
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
                              llvm::SmallVectorImpl<OpFoldResult> &results) {
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

}  // namespace corert_sync
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/core_runtime/opdefs/sync/core_runtime_opdefs.cpp.inc"
