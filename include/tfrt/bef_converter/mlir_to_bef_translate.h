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

// This file declares the translation function for MLIRToBEF.

#ifndef TFRT_BEF_CONVERTER_MLIR_TO_BEF_TRANSLATE_H_
#define TFRT_BEF_CONVERTER_MLIR_TO_BEF_TRANSLATE_H_

#include "tfrt/support/forward_decls.h"

namespace mlir {
class ModuleOp;
struct LogicalResult;
}  // namespace mlir

namespace tfrt {

mlir::LogicalResult MLIRToBEFTranslate(mlir::ModuleOp module,
                                       llvm::raw_ostream& output);

}

#endif  // TFRT_BEF_CONVERTER_MLIR_TO_BEF_TRANSLATE_H_
