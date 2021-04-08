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

// MLIR opdefs for tfrt dialect
//
// This file declares the 'tfrt' dialect.
#ifndef TFRT_BASIC_KERNELS_OPDEFS_TFRT_BASE_H_
#define TFRT_BASIC_KERNELS_OPDEFS_TFRT_BASE_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

namespace tfrt {

// Dialect for basic operations.
class TFRTDialect : public mlir::Dialect {
 public:
  explicit TFRTDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "tfrt"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};

}  // namespace tfrt

#endif  // TFRT_BASIC_KERNELS_OPDEFS_TFRT_BASE_H_
