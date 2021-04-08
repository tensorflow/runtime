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

// This file declares CpuOpRegistry::Impl for use by the registry and the
// CPU device.

#ifndef TFRT_BACKENDS_CPU_LIB_CORE_RUNTIME_CPU_OP_REGISTRY_IMPL_H_
#define TFRT_BACKENDS_CPU_LIB_CORE_RUNTIME_CPU_OP_REGISTRY_IMPL_H_

#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/support/op_registry_impl.h"

namespace tfrt {

// This is the pImpl implementation details for CpuOpRegistry.
struct CpuOpRegistry::Impl final
    : OpRegistryImpl<OpMetadataFn, CpuDispatchFn, CpuOpFlags> {};

using CpuOpEntry =
    OpRegistryImpl<OpMetadataFn, CpuDispatchFn, CpuOpFlags>::OpEntry;
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_CORE_RUNTIME_CPU_OP_REGISTRY_IMPL_H_
