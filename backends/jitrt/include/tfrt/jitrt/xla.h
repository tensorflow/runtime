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

#ifndef TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_XLA_H_
#define TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_XLA_H_

// We are in the process of migrating ::tfrt::jitrt under ::xla::runtime
// namespace. While we are in the mixed state, bring back types to the jitrt
// namespace.

namespace xla {
namespace runtime {

class RuntimeDialect;
class KernelContextType;
class StatusType;
class CustomCallOp;
class IsOkOp;
class SetOutputOp;
class SetErrorOp;

}  // namespace runtime
}  // namespace xla

namespace tfrt {
namespace jitrt {

using ::xla::runtime::CustomCallOp;       // NOLINT
using ::xla::runtime::IsOkOp;             // NOLINT
using ::xla::runtime::KernelContextType;  // NOLINT
using ::xla::runtime::RuntimeDialect;     // NOLINT
using ::xla::runtime::SetErrorOp;         // NOLINT
using ::xla::runtime::SetOutputOp;        // NOLINT
using ::xla::runtime::StatusType;         // NOLINT

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_XLA_H_
