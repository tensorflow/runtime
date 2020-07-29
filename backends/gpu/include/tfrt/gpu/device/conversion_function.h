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

//===- conversion_function.h ------------------------------------*- C++ -*-===//
//
// This file declares GPU tensor util functions for copying between gpu and
// host.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_GPU_CORE_RUNTIME_TENSOR_UTIL_H_
#define TFRT_GPU_CORE_RUNTIME_TENSOR_UTIL_H_

#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class HostContext;
class GpuDispatchContext;
template <typename T>
class AsyncValueRef;
class DenseHostTensor;
class TensorConversionFnRegistry;

namespace gpu {

class GpuAllocator;
class DenseGpuTensor;

AsyncValueRef<DenseHostTensor> CopyDenseGpuTensorToHost(
    stream::CurrentContext current_context, stream::Stream stream,
    const DenseGpuTensor& tensor, HostContext* host);

Expected<DenseGpuTensor> CopyDenseHostTensorToGpu(
    stream::CurrentContext current_context, stream::Stream stream,
    GpuAllocator* allocator, const DenseHostTensor& tensor, HostContext* host);

void RegisterGpuTensorConversionFn(TensorConversionFnRegistry* registry);

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_CORE_RUNTIME_TENSOR_UTIL_H_
