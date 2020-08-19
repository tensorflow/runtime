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

//===- attributes.h ---------------------------------------------*- C++ -*-===//
//
// This file declares attributes for the 'corert' dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_CORE_RUNTIME_OPDEFS_ATTRIBUTES_H_
#define TFRT_CORE_RUNTIME_OPDEFS_ATTRIBUTES_H_

#include "mlir/IR/Attributes.h"

namespace tfrt {
namespace corert {

namespace internal {

struct ShapeAttrStorage : public mlir::AttributeStorage {
  using KeyTy = std::pair<llvm::ArrayRef<int64_t>, unsigned>;

  explicit ShapeAttrStorage(llvm::ArrayRef<int64_t> shape,
                            unsigned unranked = 0)
      : shape(shape), unranked(unranked) {}

  bool operator==(const KeyTy& key) const {
    return key == KeyTy(shape, unranked);
  }

  // NOLINTNEXTLINE
  static ShapeAttrStorage* construct(mlir::AttributeStorageAllocator& allocator,
                                     const KeyTy& key) {
    return new (allocator.allocate<ShapeAttrStorage>())
        ShapeAttrStorage(allocator.copyInto(key.first), key.second);
  }

  llvm::ArrayRef<int64_t> shape;
  unsigned unranked = 0;
};

}  // namespace internal

class ShapeAttr : public mlir::Attribute::AttrBase<ShapeAttr, mlir::Attribute,
                                                   internal::ShapeAttrStorage> {
 public:
  using Base::Base;

  // Get or create an unranked shape attribute.
  static ShapeAttr get(mlir::MLIRContext* context) {
    return Base::get(context, llvm::ArrayRef<int64_t>(), /*unranked=*/1);
  }

  // Get or create a ranked shape attribute.
  static ShapeAttr get(mlir::MLIRContext* context,
                       llvm::ArrayRef<int64_t> shape) {
    return Base::get(context, shape, /*unranked=*/0);
  }

  bool hasRank() const { return !getImpl()->unranked; }

  int64_t getRank() const {
    assert(hasRank());
    return getImpl()->shape.size();
  }

  llvm::ArrayRef<int64_t> getShape() const {
    assert(hasRank());
    return getImpl()->shape;
  }
};

}  // namespace corert
}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_OPDEFS_ATTRIBUTES_H_
