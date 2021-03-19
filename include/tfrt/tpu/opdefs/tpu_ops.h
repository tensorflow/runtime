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

//===- tpu_ops.h - MLIR op definitions for tpu_ops library ------*- C++ -*-===//
//
// This file declares the 'tpu' dialect as well as the operators that make up
// the tpu_ops library.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_TPU_OPDEFS_TPU_OPS_H_
#define TFRT_TPU_OPDEFS_TPU_OPS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "tfrt/tensor/opdefs/tensor.h"

using namespace mlir;

namespace tfrt {
namespace tpu {

// Dialect for TPU runtime operations.
class TpuRuntimeDialect : public Dialect {
 public:
  explicit TpuRuntimeDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "tpurt"; }
};

}  // namespace tpu
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/tpu/opdefs/tpu_ops.h.inc"

#endif  // TFRT_TPU_OPDEFS_TPU_OPS_H_
