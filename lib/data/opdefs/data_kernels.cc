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

//===- data_kernels.cc ----------------------------------------------------===//
//
// This file implements MLIR operation functions for the data library.
//
//===----------------------------------------------------------------------===//

#include "tfrt/data/opdefs/data_kernels.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// DataDialect Dialect
//===----------------------------------------------------------------------===//

DataDialect::DataDialect(MLIRContext *context)
    : Dialect(/*name*/ "data", context) {
  allowUnknownTypes();
  allowUnknownOperations();

  addOperations<
#define GET_OP_LIST
#include "tfrt/data/opdefs/data_kernels_opdefs.cpp.inc"
      >();
}

// Verify that the specified region contains a hex.return operation with the
// specified type list and emit an error if not.
template <typename ResultTypeContainer>
static LogicalResult checkHexReturn(Operation *op, Region *region,
                                    ResultTypeContainer result_types) {
  assert(std::distance(region->begin(), region->end()) == 1 &&
         "verifier should already check region size");
  auto *block = &region->front();

  if (block->empty() || block->back().getName().getStringRef() != "hex.return")
    return op->emitOpError("expected hex.return in body");

  if (!std::equal(block->back().getOperandTypes().begin(),
                  block->back().getOperandTypes().end(), result_types.begin(),
                  result_types.end()))
    return block->back().emitOpError()
           << "operand types don't match '" << op->getName() << "' result";

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/data/opdefs/data_kernels_opdefs.cpp.inc"

}  // namespace data
}  // namespace tfrt
