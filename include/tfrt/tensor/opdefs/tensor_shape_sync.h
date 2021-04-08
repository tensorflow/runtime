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

// MLIR op definitions for ts dialect
//
// This file declares the 'ts_sync' dialect.

#ifndef TFRT_TENSOR_OPDEFS_TENSOR_SHAPE_SYNC_H_
#define TFRT_TENSOR_OPDEFS_TENSOR_SHAPE_SYNC_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tfrt/tensor/opdefs/tensor_shape.h"

namespace tfrt {
namespace ts_sync {

// Dialect for sync tensor shape operations.
class TensorShapeSyncDialect : public mlir::Dialect {
 public:
  static mlir::StringRef getDialectNamespace() { return "ts_sync"; }
  explicit TensorShapeSyncDialect(mlir::MLIRContext *context);
};

}  // namespace ts_sync
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/tensor/opdefs/tensor_shape_sync.h.inc"

#endif  // TFRT_TENSOR_OPDEFS_TENSOR_SHAPE_SYNC_H_
