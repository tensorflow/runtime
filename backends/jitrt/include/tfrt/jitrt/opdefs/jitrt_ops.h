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

// JIT Runtime Operations.

#ifndef TFRT_BACKENDS_JITRT_JITRT_OPS_H_
#define TFRT_BACKENDS_JITRT_JITRT_OPS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect for JIT runtime operations.
#include "tfrt/jitrt/opdefs/jitrt_ops_dialect.h.inc"

#define GET_OP_CLASSES
#include "tfrt/jitrt/opdefs/jitrt_ops.h.inc"

#endif  // TFRT_BACKENDS_JITRT_JITRT_OPS_H_
