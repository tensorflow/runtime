/*
 * Copyright 2021 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MLIR operations for the RT dialect.

#include "tfrt/jitrt/opdefs/rt_ops.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"

//===----------------------------------------------------------------------===//
// RT Dialect
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/opdefs/rt_dialect.cpp.inc"

namespace tfrt {
namespace jitrt {

void RuntimeDialect::initialize() {
  allowUnknownTypes();

  addOperations<
#define GET_OP_LIST
#include "tfrt/jitrt/opdefs/rt_ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "tfrt/jitrt/opdefs/rt_types.cpp.inc"
      >();
}

}  // namespace jitrt
}  // end namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/jitrt/opdefs/rt_ops.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tfrt/jitrt/opdefs/rt_types.cpp.inc"
