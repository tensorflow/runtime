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

//===- basic_kernels.h - MLIR opdefs for basic_kernels library --*- C++ -*-===//
//
// This file declares the 'tfrt' dialect as well as the operators that make up
// the basic_kernels library.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BASIC_OPS_OPDEFS_BASIC_OPS_H_
#define TFRT_BASIC_OPS_OPDEFS_BASIC_OPS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "tfrt/basic_kernels/opdefs/basic_kernels.h.inc"

#endif  // TFRT_BASIC_OPS_OPDEFS_BASIC_OPS_H_
