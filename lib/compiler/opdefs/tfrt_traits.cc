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

#include "tfrt/compiler/opdefs/tfrt_traits.h"

namespace tfrt {
namespace compiler {
namespace internal {

mlir::LogicalResult VerifyCostAttr(mlir::Operation* op, llvm::StringRef attr) {
  auto cost_attr = op->getAttrOfType<mlir::IntegerAttr>(attr);

  if (!cost_attr) return op->emitOpError("cost attribute not found");

  if (cost_attr.getInt() <= 0)
    return op->emitOpError("cost value must be larger than 0");

  return mlir::success();
}

}  // namespace internal
}  // namespace compiler
}  // namespace tfrt
