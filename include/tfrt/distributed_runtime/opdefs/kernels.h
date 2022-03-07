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

// MLIR op defs for dist dialect
//
// This file declares the 'dist' dialect.

#ifndef TFRT_DIST_OPDEFS_DIST_H_
#define TFRT_DIST_OPDEFS_DIST_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tfrt/tensor/opdefs/tensor.h"

using namespace mlir;

namespace tfrt {
namespace dist {

// Dialect for distributed operations.
class DistributedDialect : public Dialect {
 public:
  explicit DistributedDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tfrt_dist"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};

}  // namespace dist
}  // end namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/distributed_runtime/opdefs/kernels.h.inc"

#endif  // TFRT_DIST_OPDEFS_DIST_H_
