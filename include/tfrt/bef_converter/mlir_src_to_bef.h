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

// This file declares a utility function to convert MLIR source code to BEF.

#ifndef TFRT_BEF_CONVERTER_MLIR_SRC_TO_BEF_H_
#define TFRT_BEF_CONVERTER_MLIR_SRC_TO_BEF_H_

#include "tfrt/bef/bef_buffer.h"
#include "tfrt/support/forward_decls.h"

namespace mlir {
class MLIRContext;
}
namespace tfrt {

// This function converts the specified MLIR source containing a host executor
// compatible program to the BinaryExecutableFormat (BEF) format, which is the
// low level format that the executor takes. The client can create an
// MLIRContext and add custom dialects if needed. Otherwise, the function will
// use an MLIRContext and configure it with the default TFRT dialects for the
// conversion.
//
// On error, this emits the error message through the MLIR error handler, and
// returns an empty AlignedBuffer.
BefBuffer ConvertMLIRSrcToBEF(string_view mlir_src,
                              bool disable_optional_sections,
                              mlir::MLIRContext* context = nullptr);

}  // namespace tfrt

#endif  // TFRT_BEF_CONVERTER_MLIR_SRC_TO_BEF_H_
