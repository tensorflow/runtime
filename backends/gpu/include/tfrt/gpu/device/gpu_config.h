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

// This file defines various helpers to configure GPU OpHandler.
#ifndef TFRT_GPU_DEVICE_GPU_CONFIG_H_
#define TFRT_GPU_DEVICE_GPU_CONFIG_H_

#include <optional>

#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"

namespace tfrt {
namespace gpu {

using GpuAllocatorFactory =
    std::function<std::unique_ptr<GpuAllocator>(const wrapper::Context&)>;

struct GpuResources {
  // Either CUcontext or hipCtx_t.
  wrapper::Context gpu_context;
  // Caller needs to ensure that the factory always returns the same allocator
  // for the given context, instead of creating a new one each time.
  GpuAllocatorFactory allocator_factory;
  // A stream pointer (either a CUstream or hipStream_t).
  wrapper::Stream stream;
};

void SetTfrtGpuResources(wrapper::Device device, GpuResources resources);

std::optional<GpuResources> GetTfrtGpuResources(wrapper::Device device);

}  // namespace gpu
}  // namespace tfrt
#endif  // TFRT_GPU_DEVICE_GPU_CONFIG_H_
