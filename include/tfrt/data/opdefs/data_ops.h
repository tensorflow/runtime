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

//===- data_ops.h -----------------------------------------------*- C++ -*-===//
//
// MLIR opdefs for data library.
//
// This file declares the 'data' dialect as well as the operators that make up
// the data library.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DATA_OPDEFS_DATA_OPS_H_
#define TFRT_DATA_OPDEFS_DATA_OPS_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"

using namespace mlir;

namespace tfrt {
namespace data {

// Dialect for data operations.
class DataDialect : public Dialect {
 public:
  explicit DataDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tfrt_data"; }
};

#define GET_OP_CLASSES
#include "tfrt/data/opdefs/data_ops.h.inc"

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_DATA_OPDEFS_DATA_OPS_H_
