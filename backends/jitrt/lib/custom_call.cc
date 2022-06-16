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

//===- custom_call.cc - ---------------------------------------------------===//
// JitRt custom calls library.
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/custom_call.h"

#include <string>

namespace tfrt {
namespace jitrt {

static void PrintArr(raw_ostream& os, string_view name, ArrayRef<int64_t> arr) {
  os << " " << name << ": [";
  auto i64_to_string = [](int64_t v) { return std::to_string(v); };
  os << llvm::join(llvm::map_range(arr, i64_to_string), ", ");
  os << "]";
}

raw_ostream& operator<<(raw_ostream& os, const StridedMemrefView& view) {
  os << "StridedMemrefView: dtype: " << view.dtype;
  PrintArr(os, "sizes", view.sizes);
  PrintArr(os, "strides", view.strides);
  return os;
}

raw_ostream& operator<<(raw_ostream& os, const MemrefView& view) {
  os << "MemrefView: dtype: " << view.dtype;
  PrintArr(os, "sizes", view.sizes);
  return os;
}

raw_ostream& operator<<(raw_ostream& os, const FlatMemrefView& view) {
  return os << "FlatMemrefView: dtype: " << view.dtype
            << " size_in_bytes: " << view.size_in_bytes;
}

}  // namespace jitrt
}  // namespace tfrt

JITRT_DEFINE_EXPLICIT_TYPE_ID(llvm::StringRef);
JITRT_DEFINE_EXPLICIT_TYPE_ID(tfrt::jitrt::StridedMemrefView);
JITRT_DEFINE_EXPLICIT_TYPE_ID(tfrt::jitrt::MemrefView);
JITRT_DEFINE_EXPLICIT_TYPE_ID(tfrt::jitrt::FlatMemrefView);
JITRT_DEFINE_EXPLICIT_TYPE_ID(int32_t);
JITRT_DEFINE_EXPLICIT_TYPE_ID(int64_t);
JITRT_DEFINE_EXPLICIT_TYPE_ID(float);
JITRT_DEFINE_EXPLICIT_TYPE_ID(double);
JITRT_DEFINE_EXPLICIT_TYPE_ID(ArrayRef<int32_t>);
JITRT_DEFINE_EXPLICIT_TYPE_ID(ArrayRef<int64_t>);
JITRT_DEFINE_EXPLICIT_TYPE_ID(ArrayRef<float>);
JITRT_DEFINE_EXPLICIT_TYPE_ID(ArrayRef<double>);
