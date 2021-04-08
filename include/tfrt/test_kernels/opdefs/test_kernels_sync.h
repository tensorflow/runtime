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

// MLIR op definitions for sync test kernels.
// This file declares the 'tfrt_test_sync' dialect as well as the operators in
// the test_kernels_sync library.

#ifndef TFRT_TEST_KERNELS_OPDEFS_TEST_KERNELS_SYNC_H_
#define TFRT_TEST_KERNELS_OPDEFS_TEST_KERNELS_SYNC_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tfrt/tensor/opdefs/tensor.h"

namespace tfrt {
namespace test_sync {

// Dialect for synchronous test operations.
class TestSyncDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tfrt_test_sync"; }
  explicit TestSyncDialect(MLIRContext* context);
};

}  // namespace test_sync
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/test_kernels/opdefs/test_kernels_sync.h.inc"

#endif  // TFRT_TEST_KERNELS_OPDEFS_TEST_KERNELS_SYNC_H_
