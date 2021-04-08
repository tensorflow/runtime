// Copyright 2021 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements MLIR traits for the `tfrt` dialect.

#include "tfrt/basic_kernels/opdefs/tfrt_traits.h"

namespace tfrt {
namespace compiler_internal {

mlir::LogicalResult VerifyCostAttr(mlir::Operation* op,
                                   llvm::StringRef attr_name) {
  auto cost_attr = op->getAttrOfType<mlir::IntegerAttr>(attr_name);

  if (!cost_attr) return op->emitOpError("failed to find cost attribute");

  if (cost_attr.getInt() <= 0)
    return op->emitOpError("requires the cost attribute larger than 0");

  return mlir::success();
}

}  // namespace compiler_internal
}  // namespace tfrt
