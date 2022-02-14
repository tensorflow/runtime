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

//===- symbolic_shape.cc - ------------------------------------------------===//
// Resolving symbolic shapes for JitRt shape specialization.
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/symbolic_shape.h"

#include <algorithm>
#include <numeric>

#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Compiler.h"

namespace tfrt {
namespace jitrt {

using SymbolicShape = SymbolicShapesResolver::SymbolicShape;

//----------------------------------------------------------------------------//
// SymbolicShapesResolver implementation.
//----------------------------------------------------------------------------//

SymbolicShapesResolver::SymbolicShapesResolver(
    const FunctionType& signature, ArrayRef<OperandConstraint> constraints)
    : constraints_(constraints.begin(), constraints.end()) {
  for (unsigned i = 0; i < signature.num_operands(); ++i) {
    auto* type = signature.operand(i);

    // For unranked operands we do not know any static shape information.
    if (isa<UnrankedTensorType, UnrankedMemrefType>(type)) {
      operands_sizes_.emplace_back();
      continue;
    }

    auto emplace_sizes = [&](ArrayRef<int64_t> sizes) {
      operands_sizes_.emplace_back(llvm::to_vector(sizes));

      // Keep track of all statically known dimension sizes.
      for (int64_t size : sizes) {
        if (size != MemrefType::kDynamicSize) seen_static_sizes_.insert(size);
      }
    };

    // Copy memref dimensions sizes from the signature type.
    if (auto* memref = dyn_cast<MemrefType>(type)) {
      emplace_sizes(memref->sizes());
      continue;
    }

    // Copy tensor dimensions sizes from the signature type.
    if (auto* tensor = dyn_cast<RankedTensorType>(type)) {
      emplace_sizes(tensor->sizes());
      continue;
    }

    assert(false && "unsupported operand type");
  }

  // When resolving symbolic shapes we should visit operands starting from the
  // more constrained ones, because they can change the static signature of the
  // function, and this information should be propagated to operands with
  // dynamic shapes (e.g. all seen static sizes should be materialized in the
  // function signature).
  iteration_order_.resize(signature.num_operands());
  std::iota(iteration_order_.begin(), iteration_order_.end(), 0);

  // Make the sort stable so that dynamic shapes are computed deterministically.
  llvm::sort(iteration_order_, [&](size_t a, size_t b) {
    unsigned ca = static_cast<unsigned>(constraints[a]);
    unsigned cb = static_cast<unsigned>(constraints[b]);
    if (ca > cb) return true;
    return ca < cb ? false : a < b;
  });
}

mlir::FailureOr<llvm::SmallVector<SymbolicShape>>
SymbolicShapesResolver::Resolve(ArrayRef<MemrefDesc> operands) {
  // The number of operands must match the function signature.
  assert(operands.size() == operands_sizes_.size());

  // Mapping from the runtime dimension size to the symbolic dimension.
  llvm::SmallDenseMap<int64_t, int64_t, 16> size_to_symbolic_dim;

  // Resolved symbolic shapes.
  llvm::SmallVector<SymbolicShape> symbolic_shapes;
  symbolic_shapes.resize(operands.size());

  int64_t sym_dim = -2;  // the next symbolic dimension id

  for (size_t i : iteration_order_) {
    bool has_static_sizes = operands_sizes_[i].hasValue();
    ArrayRef<int64_t> runtime_sizes = operands[i].sizes;

    // Check that statically known rank matches the runtime rank.
    if (LLVM_UNLIKELY(has_static_sizes &&
                      operands_sizes_[i]->size() != runtime_sizes.size()))
      return mlir::failure();

    // For shape constrained operands use runtime shape.
    if (constraints_[i] == OperandConstraint::kShape) {
      symbolic_shapes[i].assign(runtime_sizes.begin(), runtime_sizes.end());

      // Add all runtime dimensions to the `size_to_symbolic_dim` to materialize
      // all dynamic dimensions of the same size as static dimensions.
      for (int64_t d : runtime_sizes) size_to_symbolic_dim.try_emplace(d, d);

      continue;
    }

    // Initialize symbolic shape with a statically known shape of the operand if
    // it is available, otherwise initialize it with a fully dynamic shape with
    // rank matching the runtime rank.
    if (has_static_sizes) {
      ArrayRef<int64_t> static_sizes = *operands_sizes_[i];
      assert(runtime_sizes.size() == static_sizes.size());
      symbolic_shapes[i].assign(static_sizes.begin(), static_sizes.end());
    } else {
      size_t rank = runtime_sizes.size();
      symbolic_shapes[i].resize(rank, MemrefType::kDynamicSize);
    }

    MutableArrayRef<int64_t> symbolic_sizes = symbolic_shapes[i];

    for (unsigned d = 0; d < runtime_sizes.size(); ++d) {
      int64_t symbolic_dim = symbolic_sizes[d];
      int64_t runtime_dim = runtime_sizes[d];

      // Skip statically known dimensions.
      if (symbolic_dim >= 0) {
        // Check that statically known dimension agrees with runtime dimension.
        if (LLVM_UNLIKELY(symbolic_dim != runtime_dim)) return mlir::failure();
        continue;
      }

      // Update unknown dimension to a static dimension.
      if (runtime_dim == 1 || seen_static_sizes_.contains(runtime_dim)) {
        symbolic_sizes[d] = runtime_dim;
        continue;
      }

      // Try to assign a symbolic dimension to the runtime dimension.
      auto emplaced = size_to_symbolic_dim.try_emplace(runtime_dim, sym_dim);
      symbolic_sizes[d] = emplaced.first->second;

      // Update the symbolic dimension if we assigned the previous value to the
      // runtime dimension size.
      if (emplaced.second) --sym_dim;
    }
  }

  return symbolic_shapes;
}

/*static*/ llvm::SmallVector<int64_t> SymbolicShapesResolver::Normalize(
    const SymbolicShape& shape) {
  auto normalize = llvm::map_range(shape, [](int64_t dim) {
    return std::max(dim, mlir::ShapedType::kDynamicSize);
  });
  return {normalize.begin(), normalize.end()};
}

static llvm::hash_code SymbolicShapeHash(const SymbolicShape& shape) {
  return llvm::hash_combine(
      shape.size(), llvm::hash_combine_range(shape.begin(), shape.end()));
}

/*static*/ llvm::hash_code SymbolicShapesResolver::Hash(
    ArrayRef<SymbolicShape> symbolic_shapes) {
  if (LLVM_UNLIKELY(symbolic_shapes.empty())) return llvm::hash_code(0);

  llvm::hash_code hash = SymbolicShapeHash(symbolic_shapes[0]);
  for (unsigned i = 1; i < symbolic_shapes.size(); ++i)
    hash = llvm::hash_combine(hash, SymbolicShapeHash(symbolic_shapes[i]));

  return hash;
}

}  // namespace jitrt
}  // namespace tfrt
