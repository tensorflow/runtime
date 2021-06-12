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

#ifndef TFRT_COMPILER_OPDEFS_TFRT_TRAITS_H_
#define TFRT_COMPILER_OPDEFS_TFRT_TRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace tfrt {
namespace compiler {
namespace internal {

mlir::LogicalResult VerifyCostAttr(mlir::Operation* op, llvm::StringRef attr);

}  // namespace internal
}  // namespace compiler
}  // namespace tfrt

namespace mlir {
namespace OpTrait {
namespace tfrt {

//===----------------------------------------------------------------------===//
// TFRT Traits implementing the Cost Function Interface (tfrt_op_interfaces.td).
//===----------------------------------------------------------------------===//

// The cost of executing an operation is fixed and independent of the
// operation operands or attributes (e.g. simple metadata manipulation).
template <int value>
class FixedCostTrait {
 public:
  template <typename ConcreteType>
  class Impl : TraitBase<ConcreteType, Impl> {
   public:
    int64_t cost() { return value; }
  };
};

// The cost of executing operation specified by a special attribute attached to
// the operation. If operation cost can't be inferred locally just from the
// operation itself, then the cost can be computed using a pass that analyses
// larger scope (e.g. whole function analysis) and attaches this information
// to relevant operations using the attribute.
//
// Also TFRT operation (kernel) can loose some of the static information
// available at earlier compilation stages (e.g. all tensors become `!t.tensor`
// and have no dimension information), and in this case conversion passes can
// use an attribute to pass cost information to the runtime.
template <typename ConcreteType>
class AttrCostTrait : public TraitBase<ConcreteType, AttrCostTrait> {
 public:
  static llvm::StringRef attr_name() { return "_tfrt_cost"; }

  static LogicalResult verifyTrait(Operation* op) {
    return ::tfrt::compiler::internal::VerifyCostAttr(op, attr_name());
  }

  int64_t cost() {
    Operation* op = AttrCostTrait::getOperation();
    return op->getAttrOfType<mlir::IntegerAttr>(attr_name()).getInt();
  }
};

}  // namespace tfrt
}  // namespace OpTrait
}  // namespace mlir

#endif  // TFRT_COMPILER_OPDEFS_TFRT_TRAITS_H_
