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

//===- init_tfrt_dialects.h -------------------------------------*- C++ -*-===//
//
// This declares a helper to register all TFRT dialects.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_INIT_TFRT_DIALECTS_H_
#define TFRT_INIT_TFRT_DIALECTS_H_

#include "mlir/IR/Dialect.h"

namespace tfrt {

void RegisterTFRTDialects(mlir::DialectRegistry &registry);

}  // namespace tfrt

#endif  // TFRT_INIT_TFRT_DIALECTS_H_
