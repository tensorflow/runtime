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

// op definitions for core runtime kernels
//
// This file declares the 'corert_sync' dialect as well as the operators that
// make up the core_runtime kernel library.

#ifndef TFRT_CORE_RUNTIME_OPDEFS_SYNC_CORE_RUNTIME_H_
#define TFRT_CORE_RUNTIME_OPDEFS_SYNC_CORE_RUNTIME_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tfrt/core_runtime/opdefs/traits.h"

namespace tfrt {
namespace corert_sync {

using namespace mlir;

// Dialect for corert_sync operations.
class CoreRTSyncDialect : public Dialect {
 public:
  explicit CoreRTSyncDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "corert_sync"; }

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 mlir::Location loc) override;
};

}  // namespace corert_sync
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/core_runtime/opdefs/sync/core_runtime_opdefs.h.inc"

#endif  // TFRT_CORE_RUNTIME_OPDEFS_SYNC_CORE_RUNTIME_H_
