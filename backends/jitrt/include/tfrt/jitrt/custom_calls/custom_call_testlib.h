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

#ifndef TFRT_BACKENDS_JITRT_CUSTOM_CALLS_CUSTOM_CALLS_TESTLIB_H_
#define TFRT_BACKENDS_JITRT_CUSTOM_CALLS_CUSTOM_CALLS_TESTLIB_H_

#include <cstdint>

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "tfrt/jitrt/custom_call.h"

// clang-format off
#include "tfrt/jitrt/custom_calls/custom_call_testlib_dialect.h.inc"
#include "tfrt/jitrt/custom_calls/custom_call_testlib_enums.h.inc"
// clang-format on

#define GET_ATTRDEF_CLASSES
#include "tfrt/jitrt/custom_calls/custom_call_testlib_attrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "tfrt/jitrt/custom_calls/custom_call_testlib_types.h.inc"

namespace tfrt {
namespace jitrt {

class CustomCallAttrEncodingSet;

DirectCustomCallLibrary CustomCallTestlib();

// Declare runtime enums corresponding to compile time enums to test
// attributes enum conversion.
enum class RuntimeEnumType : uint32_t { kFoo, kBar, kBaz };

// Runtime structure corresponding to the compile-time PairOfDims MLIR attribute
// to test attributes conversion.
struct RuntimePairOfDims {
  int64_t rank;
  llvm::ArrayRef<int64_t> a;
  llvm::ArrayRef<int64_t> b;
};

// Populate encoding for custom dialect attributes (enums and structs).
void PopulateCustomCallAttrEncoding(CustomCallAttrEncodingSet &encoding);

// Explicitly register attributes decoding for enums passed to the custom calls.
JITRT_REGISTER_ENUM_ATTR_DECODING(EnumType);
JITRT_REGISTER_ENUM_ATTR_DECODING(RuntimeEnumType);

// Explicitly register aggregate attributes decoding for structs.
JITRT_REGISTER_AGGREGATE_ATTR_DECODING(RuntimePairOfDims,
                                       JITRT_AGGREGATE_FIELDS("rank", "a", "b"),
                                       int64_t, ArrayRef<int64_t>,
                                       ArrayRef<int64_t>);

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_CUSTOM_CALLS_CUSTOM_CALLS_TESTLIB_H_
