/*
 * Copyright 2022 The TensorFlow Runtime Authors
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

#ifndef TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_CONSTRAINTS_H_
#define TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_CONSTRAINTS_H_

#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace jitrt {

// Constraints on the function operands can be specified with the function
// argument attributes.
//
// Example:
//
//   func @compute(
//     // Rank of the `%arg` must be known at compile time.
//     %arg: tensor<*xf32> { jitrt.constraint = "rank" }
//   ) -> tensor<?xf32> { ... }
//
// TODO(b/187114012): Add attribute verifier to jitrt dialect.
constexpr const char* kOperandConstraintAttrName = "jitrt.constraint";

// Constraint on what operand information must be available at compile time in
// order to successfully compile the executable:
//
//   `rank`  : operand must have statically known rank.
//   `shape` : operand must have statically known shape.
//   `value` : operand must have statically known value, and such operands
//             replaced with constants inside the compiled function body and
//             and all value constrained argument uses replaced with the sunk
//             constant value.
//
enum class OperandConstraint {
  // Constraint was resolved based on the static information in the function
  // signature type or it was never specified by the operand attribute.
  kResolved = 0,
  kRank = 1,
  kShape = 2,
  kValue = 3
};

raw_ostream& operator<<(raw_ostream& os, const OperandConstraint&);
raw_ostream& operator<<(raw_ostream& os, ArrayRef<OperandConstraint>);

// Converts operand constraint string to the corresponding enum class.
Expected<OperandConstraint> ParseOperandConstraint(string_view str);

// Returns operands constraints inferred from the entrypoint signature.
Expected<llvm::SmallVector<OperandConstraint>> GetOperandsConstraints(
    mlir::FuncOp func);

// Resolves operand constraint based on the operand type, if constraint is fully
// satisfied by the type, returns `kResolved`.
Expected<OperandConstraint> ResolveOperandConstraint(
    OperandConstraint operand_constraint, mlir::Type operand_type);

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_CONSTRAINTS_H_
