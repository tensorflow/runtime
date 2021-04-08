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

// This declares a helper to register all TFRT dialects.

#ifndef TFRT_INIT_TFRT_DIALECTS_H_
#define TFRT_INIT_TFRT_DIALECTS_H_

#include "mlir/IR/Dialect.h"

namespace tfrt {

// Registers dialects that can be used in executed MLIR functions (functions
// with operations that will be translated to BEF kernel calls).
void RegisterTFRTDialects(mlir::DialectRegistry &registry);

// Registers dialects that can be used in compiled MLIR modules (blobs of IR
// embedded into the executed MLIR/BEF programs).
void RegisterTFRTCompiledDialects(mlir::DialectRegistry &registry);

}  // namespace tfrt

#endif  // TFRT_INIT_TFRT_DIALECTS_H_
