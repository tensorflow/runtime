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

// Emit a string to a BEF strings section.

#ifndef TFRT_LIB_BEF_CONVERTER_MLIR_TO_BEF_BEF_STRING_EMITTER_H_
#define TFRT_LIB_BEF_CONVERTER_MLIR_TO_BEF_BEF_STRING_EMITTER_H_

#include "llvm/ADT/StringMap.h"
#include "tfrt/bef_converter/bef_emitter.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// BEF tring emitter class.
class BefStringEmitter : public BefEmitter {
 public:
  // Emit a string.
  size_t EmitString(string_view str);

 private:
  llvm::StringMap<size_t> offset_map_;
};

}  // namespace tfrt

#endif  // TFRT_LIB_BEF_CONVERTER_MLIR_TO_BEF_BEF_STRING_EMITTER_H_
