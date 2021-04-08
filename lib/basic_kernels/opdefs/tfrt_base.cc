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

// This file implements tfrt dialect.

#include "tfrt/basic_kernels/opdefs/tfrt_base.h"

#include "tfrt/basic_kernels/opdefs/basic_kernels.h"
#include "tfrt/basic_kernels/opdefs/types.h"

namespace tfrt {

//===----------------------------------------------------------------------===//
// TFRTDialect Dialect
//===----------------------------------------------------------------------===//

TFRTDialect::TFRTDialect(mlir::MLIRContext *context)
    : mlir::Dialect(/*name=*/"tfrt", context, TypeID::get<TFRTDialect>()) {
  allowUnknownTypes();

  // TODO(b/160693129): Eventually specify all of the operations.
  allowUnknownOperations();

  addTypes<ChainType, StringType, TensorTypeType, DeviceType>();

  addOperations<
#define GET_OP_LIST
#include "tfrt/basic_kernels/opdefs/basic_kernels_opdefs.cpp.inc"
      >();
}

mlir::Type TFRTDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef spec = parser.getFullSymbolSpec();

  if (spec == "chain") return ChainType::get(getContext());
  if (spec == "string") return StringType::get(getContext());
  if (spec == "tensor_type") return TensorTypeType::get(getContext());
  if (spec == "device") return DeviceType::get(getContext());
  if (auto type = mlir::Dialect::parseType(parser)) return type;

  mlir::Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  mlir::emitError(loc) << "unknown tfrt type " << spec;
  return {};
}

void TFRTDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &printer) const {
  if (type.isa<ChainType>()) {
    printer << "chain";
  } else if (type.isa<StringType>()) {
    printer << "string";
  } else if (type.isa<TensorTypeType>()) {
    printer << "tensor_type";
  } else if (type.isa<DeviceType>()) {
    printer << "device";
  } else {
    llvm_unreachable("unknown tfrt type");
  }
}

}  // namespace tfrt
