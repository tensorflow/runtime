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

// This file implements MLIR operation functions for the tensor dialect.

#include "tfrt/tensor/opdefs/tensor.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

namespace tfrt {
namespace tfrt_tensor {

//===----------------------------------------------------------------------===//
// TensorShape Dialect
//===----------------------------------------------------------------------===//

TensorDialect::TensorDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfrt_tensor", context, TypeID::get<TensorDialect>()) {
  allowUnknownTypes();
  addTypes<TensorType>();
  addOperations<
#define GET_OP_LIST
#include "tfrt/tensor/opdefs/tensor.cpp.inc"
      >();
}

/// Parse a type registered to this dialect.
Type TensorDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword)) return Type();

  if (keyword == "tensor") return TensorType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown tensor type: ") << keyword;
  return Type();
}

/// Print a type registered to this dialect.
void TensorDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (mlir::isa<TensorType>(type)) {
    os << "tensor";
    return;
  }

  llvm_unreachable("unexpected 'tensor' type kind");
}

}  // namespace tfrt_tensor
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/tensor/opdefs/tensor.cpp.inc"
