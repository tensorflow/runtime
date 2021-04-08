/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

// This file declares traits for the 'tfrt' dialect.

#ifndef TFRT_BASIC_KERNELS_OPDEFS_TFRT_TRAITS_H_
#define TFRT_BASIC_KERNELS_OPDEFS_TFRT_TRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace tfrt {
namespace compiler_internal {

mlir::LogicalResult VerifyCostAttr(mlir::Operation* op,
                                   llvm::StringRef attr_name);

}
}  // namespace tfrt

namespace mlir {
namespace OpTrait {
namespace tfrt {

// Defines a trait for TFRT operations that have an integer attribute specifying
// its estimated cost. This trait class groups the relevant functionalities for
// costs of operations, so that MLIR operations directly use these methods
// without re-implement for each op, and the framework can access costs of
// operations in a centralized way.
template <typename ConcreteType>
class CostTrait : public mlir::OpTrait::TraitBase<ConcreteType, CostTrait> {
 public:
  static llvm::StringRef GetCostAttr() { return "_tfrt_cost"; }

  static LogicalResult verifyTrait(Operation* op) {
    return ::tfrt::compiler_internal::VerifyCostAttr(op, GetCostAttr());
  }
};

}  // namespace tfrt
}  // namespace OpTrait
}  // namespace mlir

#endif  // TFRT_BASIC_KERNELS_OPDEFS_TFRT_TRAITS_H_
