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

// This file implements MLIR operation functions for the host tensor dialect.

#include "tfrt/tensor/opdefs/host_tensor.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

namespace tfrt {
namespace ht {

//===----------------------------------------------------------------------===//
// HostTensor Dialect
//===----------------------------------------------------------------------===//

HostTensorDialect::HostTensorDialect(MLIRContext *context)
    : Dialect(/*name=*/"ht", context, TypeID::get<HostTensorDialect>()) {
  allowUnknownTypes();
  addTypes<HostBufferType>();
  addOperations<
#define GET_OP_LIST
#include "tfrt/tensor/opdefs/host_tensor.cpp.inc"
      >();
}

/// Parse a type registered to this dialect.
Type HostTensorDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword)) return Type();

  if (keyword == "host_buffer") return HostBufferType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown type: ") << keyword;
  return Type();
}

/// Print a type registered to this dialect.
void HostTensorDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (type.isa<HostBufferType>()) {
    os << "host_buffer";
    return;
  }

  llvm_unreachable("unexpected type kind");
}

}  // namespace ht
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/tensor/opdefs/host_tensor.cpp.inc"
