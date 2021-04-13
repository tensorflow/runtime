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

// This file declares GpuOpRegistry::Impl for use by the registry and the
// GPU device.

#ifndef TFRT_BACKENDS_GPU_LIB_CORE_RUNTIME_GPU_OP_REGISTRY_IMPL_H_
#define TFRT_BACKENDS_GPU_LIB_CORE_RUNTIME_GPU_OP_REGISTRY_IMPL_H_

#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/support/op_registry_impl.h"

namespace tfrt {
namespace gpu {
struct GpuOpFlags {};

// This is the pImpl implementation details for GpuOpRegistry.
struct GpuOpRegistry::Impl final
    : OpRegistryImpl<OpMetadataFn, GpuDispatchFn, GpuOpFlags> {};

using GpuOpEntry =
    OpRegistryImpl<OpMetadataFn, GpuDispatchFn, GpuOpFlags>::OpEntry;

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_GPU_LIB_CORE_RUNTIME_GPU_OP_REGISTRY_IMPL_H_
