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

//===- bef_attr_encoder.h ---------------------------------------*- C++ -*-===//
//
// This file declares the interface to the BEF Attr Encoder library.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BEF_CONVERTER_BEF_ATTR_ENCODER_H_
#define TFRT_BEF_CONVERTER_BEF_ATTR_ENCODER_H_

#include "tfrt/bef_converter/bef_emitter.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// TODO(zhangqiaorjc): Unify with BEFTypedAttributeEmitter.
// Consider a more open design using ADL. Encoder functions can be made free
// functions inside headers for TensorShape, and other user defined types.
class BEFTypedAttributeEncoder : public BEFEmitter {
 public:
  llvm::Error EncodeShapeAttr(ArrayRef<int64_t> dims);

  llvm::Error EncodeShapeListAttr(const int64_t** dims, const int* num_dims,
                                  int num_values);

  llvm::Error EncodeStringAttr(string_view sv);

  llvm::Error EncodeStringListAttr(const void* const* values,
                                   const size_t* lengths, int num_values);
};

}  // namespace tfrt

#endif  // TFRT_BEF_CONVERTER_BEF_ATTR_ENCODER_H_
