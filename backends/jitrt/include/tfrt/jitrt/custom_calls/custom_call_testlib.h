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

#include "third_party/tensorflow/compiler/xla/mlir/runtime/ir/tests/testlib.h"
#include "third_party/tensorflow/compiler/xla/runtime/custom_call.h"
#include "third_party/tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "third_party/tensorflow/compiler/xla/runtime/type_id.h"

namespace xla {
namespace runtime {
class CustomCallAttrEncodingSet;
}  // namespace runtime
}  // namespace xla

namespace tfrt {
namespace jitrt {

void RegisterDirectCustomCallTestLib(
    xla::runtime::DirectCustomCallRegistry& registry);

void RegisterDynamicCustomCallTestLib(
    xla::runtime::DynamicCustomCallRegistry& registry);

// Declare runtime enums corresponding to compile time enums to test
// attributes enum conversion.
enum class RuntimeEnumType : uint32_t { kFoo, kBar, kBaz };

// Runtime structure corresponding to the compile-time PairOfDims MLIR attribute
// to test attributes conversion.
struct RuntimePairOfDims {
  int64_t rank;
  absl::Span<const int64_t> a;
  absl::Span<const int64_t> b;
};

// Populate type names for the custom enums and structs.
void PopulateCustomCallTypeIdNames(xla::runtime::TypeIDNameRegistry& registry);

// Populate encoding for custom dialect attributes (enums and structs).
void PopulateCustomCallAttrEncoding(
    xla::runtime::CustomCallAttrEncodingSet& encoding);

}  // namespace jitrt
}  // namespace tfrt

namespace xla {
namespace runtime {

// Explicitly register attributes decoding for enums passed to the custom calls.
XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(EnumType);
XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(tfrt::jitrt::RuntimeEnumType);

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    tfrt::jitrt::RuntimePairOfDims, AggregateMember<int64_t>("rank"),
    AggregateMember<absl::Span<const int64_t>>("a"),
    AggregateMember<absl::Span<const int64_t>>("b"));

}  // namespace runtime
}  // namespace xla

XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(tfrt::jitrt::RuntimeEnumType);

#endif  // TFRT_BACKENDS_JITRT_CUSTOM_CALLS_CUSTOM_CALLS_TESTLIB_H_
