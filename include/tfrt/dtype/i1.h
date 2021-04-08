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

// This file defines the single byte integer type: i1.

#ifndef TFRT_SUPPORT_I1_H_
#define TFRT_SUPPORT_I1_H_

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// TODO(tfrt-devs): Add arithmetic operators and upgrade to DTYPE_NUMERIC.
// This is just a placeholder type telling core TFRT that i1 has the same
// size as uint8_t. The client should get its real C++ type via
// tfrt::TypeForDTypeKind<DType::Kind::I1>::Type.
struct i1 {
  i1() : value(0) {}
  explicit i1(uint8_t v) : value(v) {}
  uint8_t value;
};

inline raw_ostream& operator<<(raw_ostream& os, const i1& i1) {
  return os << llvm::format("%.*g", std::numeric_limits<uint8_t>::max_digits10,
                            i1.value);
}
}  // namespace tfrt

#endif  // TFRT_SUPPORT_I1_H_
