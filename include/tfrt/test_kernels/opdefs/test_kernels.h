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

// MLIR op definitions for test_kernels
//
// This file declares the 'test' dialect as well as the operators in the
// test_kernels library.

#ifndef TFRT_TEST_KERNELS_OPDEFS_TEST_KERNELS_H_
#define TFRT_TEST_KERNELS_OPDEFS_TEST_KERNELS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tfrt/compiler/opdefs/tfrt_op_interfaces.h"
#include "tfrt/compiler/opdefs/tfrt_traits.h"
#include "tfrt/core_runtime/opdefs/traits.h"
#include "tfrt/tensor/opdefs/tensor.h"

using namespace mlir;

namespace tfrt {
namespace test {

// Dialect for test operations.
class TestDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tfrt_test"; }
  explicit TestDialect(MLIRContext* context);
};

}  // namespace test
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/test_kernels/opdefs/test_kernels.h.inc"

#endif  // TFRT_TEST_KERNELS_OPDEFS_TEST_KERNELS_H_
