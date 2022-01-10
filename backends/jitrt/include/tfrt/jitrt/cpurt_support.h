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

// Support library to be used by the CPURT library and related code,
// without the latter having to link the former (and its dependencies).

#ifndef TFRT_BACKENDS_CPU_JIT_CPURT_SUPPORT_H_
#define TFRT_BACKENDS_CPU_JIT_CPURT_SUPPORT_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace tfrt {
namespace cpu {
namespace jit {

// Returns true iff the shape can be sunk into the function body at run time
// via value specialization.
inline bool SupportsValueSpecialization(mlir::Type type) {
  mlir::ShapedType shaped = type.dyn_cast<mlir::ShapedType>();
  return shaped && (shaped.getRank() == 0 || shaped.getRank() == 1) &&
         (shaped.getElementType().isInteger(32) ||
          shaped.getElementType().isInteger(64));
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_JIT_CPURT_SUPPORT_H_
