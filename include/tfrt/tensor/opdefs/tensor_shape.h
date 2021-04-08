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

// MLIR op definitions for ts dialect
//
// This file declares the 'ts' dialect.

#ifndef TFRT_TENSOR_OPDEFS_TENSOR_SHAPE_H_
#define TFRT_TENSOR_OPDEFS_TENSOR_SHAPE_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace tfrt {
namespace ts {

/// The shape descriptor type represents rank and dimension sizes.
class ShapeType : public Type::TypeBase<ShapeType, Type, TypeStorage> {
 public:
  using Base::Base;
};

/// The partial shape descriptor type represents a static or a dynamic (unknown
/// rank/dim) shape.
class PartialShapeType
    : public Type::TypeBase<PartialShapeType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace ts
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/tensor/opdefs/tensor_shape.h.inc"
#include "tfrt/tensor/opdefs/tensor_shape_dialect.h.inc"

#endif  // TFRT_TENSOR_OPDEFS_TENSOR_SHAPE_H_
