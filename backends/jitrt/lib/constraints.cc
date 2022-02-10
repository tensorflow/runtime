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

//===- constraints.cc - ---------------------------------------------------===//
// JitRt constraints on the compiled function operands.
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/constraints.h"

#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "tfrt/jitrt/support.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace jitrt {

raw_ostream& operator<<(raw_ostream& os, const OperandConstraint& constraint) {
  auto str = [](OperandConstraint constraint) -> string_view {
    switch (constraint) {
      case OperandConstraint::kResolved:
        return "resolved";
      case OperandConstraint::kRank:
        return "rank";
      case OperandConstraint::kShape:
        return "shape";
      case OperandConstraint::kValue:
        return "value";
      default:
        llvm_unreachable("unknown operand constraint");
    }
  };

  os << str(constraint);
  return os;
}

raw_ostream& operator<<(raw_ostream& os,
                        ArrayRef<OperandConstraint> constraints) {
  os << "[";
  llvm::interleaveComma(constraints, os);
  os << "]";
  return os;
}

Expected<OperandConstraint> ParseOperandConstraint(string_view str) {
  if (str == "rank") return OperandConstraint::kRank;
  if (str == "shape") return OperandConstraint::kShape;
  if (str == "value") return OperandConstraint::kValue;
  return MakeStringError("unknown operand constraint: ", str);
}

Expected<llvm::SmallVector<OperandConstraint>> GetOperandsConstraints(
    mlir::FuncOp func) {
  llvm::SmallVector<OperandConstraint> constraints;
  constraints.reserve(func.getNumArguments());

  auto parse = [](mlir::Attribute attr) -> Expected<OperandConstraint> {
    // If attribute is not defined it means that there is no operand constraint.
    if (!attr) return OperandConstraint::kResolved;

    // Otherwise try to parse constraint from the string attribute.
    auto str = attr.dyn_cast_or_null<mlir::StringAttr>();
    if (!str)
      return MakeStringError("unexpected ", kOperandConstraintAttrName,
                             " attribute");
    return ParseOperandConstraint(str.getValue());
  };

  for (int i = 0; i < func.getNumArguments(); ++i) {
    auto operand_type = func.getType().getInput(i);

    auto constraint = parse(func.getArgAttr(i, kOperandConstraintAttrName));
    if (auto err = constraint.takeError()) return std::move(err);

    auto resolved = ResolveOperandConstraint(*constraint, operand_type);
    if (auto err = resolved.takeError()) return std::move(err);

    constraints.push_back(*resolved);
  }

  return constraints;
}

Expected<OperandConstraint> ResolveOperandConstraint(
    OperandConstraint operand_constraint, mlir::Type operand_type) {
  // Operand must be a shaped type: memref or tensor.
  auto shaped = operand_type.dyn_cast<mlir::ShapedType>();
  if (!shaped)
    return MakeStringError("unsupported operand type: ", operand_type);

  // Resolve `rank` constraint if rank is known at compile time.
  if (operand_constraint == OperandConstraint::kRank && shaped.hasRank())
    return OperandConstraint::kResolved;

  // Resolve `shape` constraint if shape is known at compile time.
  if (operand_constraint == OperandConstraint::kShape &&
      shaped.hasStaticShape())
    return OperandConstraint::kResolved;

  // Leave the `value` constraint unmodified if the operand is sinkable.
  if (operand_constraint == OperandConstraint::kValue) {
    if (SupportsValueSpecialization(shaped)) return operand_constraint;
    return MakeStringError("cannot sink operand type: ", operand_type);
  }

  return operand_constraint;
}

}  // namespace jitrt
}  // namespace tfrt
