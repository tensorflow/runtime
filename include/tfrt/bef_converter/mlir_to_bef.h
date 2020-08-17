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

//===- mlir_to_bef.h --------------------------------------------*- C++ -*-===//
//
// This file declares the interface to the MLIRToBEF library.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BEF_CONVERTER_MLIR_TO_BEF_H_
#define TFRT_BEF_CONVERTER_MLIR_TO_BEF_H_

#include <cstdint>
#include <vector>

#include "tfrt/support/aligned_buffer.h"

namespace mlir {
class ModuleOp;
}

namespace tfrt {

using BEFBuffer = AlignedBuffer<8>;

// This function converts the specified MLIR module containing a host executor
// compatible program to the BinaryExecutableFormat (BEF) format, which is the
// low level format that the executor takes.
//
// On error, this emits the error message through the MLIR error handler, and
// returns an empty AlignedBuffer.
BEFBuffer ConvertMLIRToBEF(mlir::ModuleOp module,
                           bool disable_optional_sections);

}  // namespace tfrt

#endif  // TFRT_BEF_CONVERTER_MLIR_TO_BEF_H_
