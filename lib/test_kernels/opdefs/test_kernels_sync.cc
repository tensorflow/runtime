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

// This file implements MLIR operation functions for the test_kernels library.

#include "tfrt/test_kernels/opdefs/test_kernels_sync.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"
#include "tfrt/basic_kernels/opdefs/types.h"

namespace tfrt {
namespace test_sync {

//===----------------------------------------------------------------------===//
// TestSyncDialect Dialect
//===----------------------------------------------------------------------===//

TestSyncDialect::TestSyncDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfrt_test_sync", context,
              TypeID::get<TestSyncDialect>()) {
  context->getOrLoadDialect<compiler::TFRTDialect>();

  allowUnknownTypes();
  allowUnknownOperations();

  addOperations<
#define GET_OP_LIST
#include "tfrt/test_kernels/opdefs/test_kernels_sync_opdefs.cpp.inc"
      >();
}

}  // namespace test_sync
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/test_kernels/opdefs/test_kernels_sync_opdefs.cpp.inc"
