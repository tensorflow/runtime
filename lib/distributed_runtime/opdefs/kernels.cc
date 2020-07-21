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
    : Dialect(/*name=*/"dist", context) {
  allowUnknownTypes();
  allowUnknownOperations();
  addOperations<
#define GET_OP_LIST
#include "tfrt/distributed_runtime/opdefs/kernels_opdefs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/distributed_runtime/opdefs/kernels_opdefs.cpp.inc"

}  // namespace dist
}  // end namespace tfrt
