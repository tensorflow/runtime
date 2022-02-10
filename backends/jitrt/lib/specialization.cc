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

//===- specialization.cc - ------------------------------------------------===//
// Specializing compiled modules to operands shapes or values.
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/specialization.h"

#include <memory>
#include <numeric>
#include <utility>

#include "llvm/Support/Error.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "tfrt/jitrt/support.h"
#include "tfrt/jitrt/symbolic_shape.h"
#include "tfrt/jitrt/types.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace jitrt {

using SymbolicShape = SymbolicShapesResolver::SymbolicShape;

// Return input `type` specialized to memref operand and its symbolic shape.
static llvm::Expected<mlir::Type> SpecializeOperandType(
    unsigned index, mlir::Type type, const MemrefDesc& operand,
    const SymbolicShape& symbolic_shape) {
  // Replace all symbolic dimensions with dynamic dimension.
  auto shape = SymbolicShapesResolver::Normalize(symbolic_shape);

  if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
    if (auto err = VerifyMemrefOperand(index, memref, operand))
      return std::move(err);
    return mlir::MemRefType::get(shape, memref.getElementType());
  }

  if (auto tensor = type.dyn_cast<mlir::RankedTensorType>()) {
    if (auto err = VerifyMemrefOperand(index, tensor, operand))
      return std::move(err);
    return mlir::RankedTensorType::get(shape, tensor.getElementType());
  }

  if (auto tensor = type.dyn_cast<mlir::UnrankedTensorType>()) {
    if (auto err = VerifyMemrefOperand(index, tensor, operand))
      return std::move(err);
    return mlir::RankedTensorType::get(shape, tensor.getElementType());
  }

  return MakeStringError("Unsupported input type: ", type);
}

// Gets (copies) the values from `desc`, returning them in a DenseElementsAttr.
// If it cannot extract the values, returns an empty attribute.
static mlir::DenseElementsAttr GetMemrefValues(mlir::Builder& builder,
                                               mlir::TensorType operand_type,
                                               const MemrefDesc& desc) {
  size_t rank = desc.sizes.size();
  if (rank != 0 && rank != 1) return {};

  llvm::SmallVector<mlir::Attribute> attributes;
  size_t num_values = rank == 0 ? 1 : desc.sizes[0];
  switch (desc.dtype) {
    case DType::I32: {
      const auto* data = static_cast<TypeForDTypeKind<DType::I32>*>(desc.data);
      for (int i = 0; i < num_values; ++i) {
        attributes.push_back(builder.getI32IntegerAttr(data[i]));
      }
    } break;
    case DType::I64: {
      const auto* data = static_cast<TypeForDTypeKind<DType::I64>*>(desc.data);
      for (int i = 0; i < num_values; ++i) {
        attributes.push_back(builder.getI64IntegerAttr(data[i]));
      }
    } break;
    default:
      return {};
  }

  // Update operand type to a ranked tensor type with statically known shape.
  auto element_type = operand_type.getElementType();
  auto ranked_tensor = mlir::RankedTensorType::get(desc.sizes, element_type);

  return mlir::DenseElementsAttr::get(ranked_tensor, attributes);
}

Error SpecializeFunction(mlir::FuncOp func, ArrayRef<MemrefDesc> operands,
                         ArrayRef<SymbolicShape> symbolic_shapes,
                         ArrayRef<OperandConstraint> constraints,
                         const SpecializationListener* listener) {
  mlir::MLIRContext* ctx = func.getContext();

  unsigned num_inputs = func.getNumArguments();

  // Specialize all function inputs to the given operands.
  llvm::SmallVector<mlir::Type> specialized_inputs(num_inputs);
  for (unsigned i = 0; i < num_inputs; ++i) {
    auto specialized = SpecializeOperandType(i, func.getType().getInput(i),
                                             operands[i], symbolic_shapes[i]);
    if (auto err = specialized.takeError()) return err;
    specialized_inputs[i] = *specialized;
  }

  // Update function type to a new specialized one.
  auto specialized = mlir::FunctionType::get(ctx, specialized_inputs,
                                             func.getType().getResults());
  func.setType(specialized);

  // Update function entry block arguments.
  mlir::Block& entry_block = func.getBlocks().front();
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(&entry_block);
  mlir::Location loc = func.getLoc();

  // Forward original block arguments to arguments with specialized type. We
  // need to insert casts to ensure the users still get the correct type and
  // avoid illegal IR. This can be optimized away by the user-provided
  // specialization pipeline, e.g., in Tensorflow these casts will be optimized
  // away by the shape inference pass.
  for (int i = 0; i < num_inputs; ++i) {
    mlir::Value new_arg = entry_block.addArgument(specialized_inputs[i], loc);
    mlir::Value old_arg = entry_block.getArgument(i);
    if (new_arg.getType() != old_arg.getType()) {
      new_arg =
          builder.create<mlir::tensor::CastOp>(loc, old_arg.getType(), new_arg);
    }
    old_arg.replaceAllUsesWith(new_arg);
  }

  // Erase all the original block arguments.
  llvm::SmallVector<unsigned> erase_block_args(num_inputs);
  std::iota(erase_block_args.begin(), erase_block_args.end(), 0);
  entry_block.eraseArguments(erase_block_args);

  // Add symbolic shapes as arguments attributes.
  for (unsigned i = 0; i < num_inputs; ++i) {
    const SymbolicShape& shape = symbolic_shapes[i];
    int64_t rank = shape.size();

    // Skip statically known shapes.
    if (llvm::all_of(shape, [](int64_t dim) { return dim >= 0; })) continue;

    // Symbolic shape attribute stored as 1d tensor attribute.
    auto i64 = mlir::IntegerType::get(ctx, 64);
    auto tensor = mlir::RankedTensorType::get({rank}, i64);

    // Create i64 attributes from the symbolic shape values.
    llvm::SmallVector<mlir::Attribute> values(rank);
    for (unsigned d = 0; d < rank; ++d)
      values[d] = mlir::IntegerAttr::get(i64, shape[d]);

    func.setArgAttr(i, "jitrt.symbolic_shape",
                    mlir::DenseElementsAttr::get(tensor, values));
  }

  // Sink small constants into the function body.
  builder.setInsertionPointToStart(&func.getBody().front());
  for (int i = 0; i < constraints.size(); ++i) {
    if (constraints[i] != OperandConstraint::kValue) continue;

    // We only support sinking of Tensor operands into the function body.
    mlir::Type input_ty = func.getType().getInput(i);
    mlir::TensorType tensor_ty = input_ty.dyn_cast<mlir::TensorType>();
    if (!tensor_ty || !SupportsValueSpecialization(tensor_ty)) {
      return MakeStringError("non-sinkable operand was marked for sinking: ",
                             input_ty);
    }

    // Get the operand value from the runtime memref operand.
    mlir::DenseElementsAttr value =
        GetMemrefValues(builder, tensor_ty, operands[i]);
    if (!value) {
      return MakeStringError("cannot get value from operand type: ", input_ty);
    }

    auto cst =
        builder.create<mlir::arith::ConstantOp>(loc, value.getType(), value);
    entry_block.getArgument(i).replaceAllUsesWith(cst);

    if (listener) listener->notifyValueSpecialized(i, value.getType(), value);
  }

  if (listener) {
    llvm::SmallVector<mlir::DictionaryAttr> specialized_attrs;
    func.getAllArgAttrs(specialized_attrs);
    listener->notifyModuleSpecialized(specialized_inputs, specialized_attrs);
  }

  return Error::success();
}

}  // namespace jitrt
}  // namespace tfrt
