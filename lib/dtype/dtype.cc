// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements DType class.

#include "tfrt/dtype/dtype.h"

#include "llvm/Support/raw_ostream.h"

namespace tfrt {

const char *DType::GetName() const {
  return DispatchByDType(*this,
                         [](auto dtype_data) { return dtype_data.kName; });
}

// Return the size of one value of this dtype when represented on the host.
size_t DType::GetHostSize() const {
  return DispatchByDType(*this,
                         [](auto dtype_data) { return dtype_data.kByteSize; });
}

// Return the alignment of this dtype when represented on the host.
size_t DType::GetHostAlignment() const {
  return DispatchByDType(*this,
                         [](auto dtype_data) { return dtype_data.kAlignment; });
}

// Support printing of dtype enums.
raw_ostream &operator<<(raw_ostream &os, DType dtype) {
  return os << dtype.GetName();
}

}  // namespace tfrt
