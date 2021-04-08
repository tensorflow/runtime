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

// CPU Runtime Operations.

#ifndef TFRT_BACKENDS_CPU_JIT_CPURT_OPS_H_
#define TFRT_BACKENDS_CPU_JIT_CPURT_OPS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace tfrt {
namespace cpu {
namespace jit {

// Dialect for CPU runtime operations.
class CpuRuntimeDialect : public mlir::Dialect {
 public:
  explicit CpuRuntimeDialect(mlir::MLIRContext* context);
  static llvm::StringRef getDialectNamespace() { return "cpurt"; }
};

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/cpu/jit/opdefs/cpurt_ops.h.inc"

#endif  // TFRT_BACKENDS_CPU_JIT_CPURT_OPS_H_
