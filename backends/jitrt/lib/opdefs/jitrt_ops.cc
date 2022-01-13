/*
 * Copyright 2020 The TensorFlow Runtime Authors
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

// This file implements MLIR operations for the TFRT CPU Runtime dialect.

#include "tfrt/jitrt/opdefs/jitrt_ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/core_runtime/opdefs/types.h"
#include "tfrt/tensor/opdefs/tensor.h"
#include "tfrt/tensor/opdefs/tensor_shape.h"

//===----------------------------------------------------------------------===//
// JitRuntimeDialect Dialect
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/opdefs/jitrt_ops_dialect.cpp.inc"

namespace tfrt {
namespace jitrt {

void JitRuntimeDialect::initialize() {
  allowUnknownTypes();

  addOperations<
#define GET_OP_LIST
#include "tfrt/jitrt/opdefs/jitrt_ops.cpp.inc"
      >();
}

}  // namespace jitrt
}  // end namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/jitrt/opdefs/jitrt_ops.cpp.inc"
