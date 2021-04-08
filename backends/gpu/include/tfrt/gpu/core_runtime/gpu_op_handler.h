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

// This file declares CreateGpuOpHandler, which creates GpuOpHandler,
// responsible for executing ops on GPU.

#ifndef TFRT_BACKENDS_GPU_CORE_RUNTIME_GPU_OP_HANDLER_H_
#define TFRT_BACKENDS_GPU_CORE_RUNTIME_GPU_OP_HANDLER_H_

#include <memory>

#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
class GpuDevice;

llvm::Expected<OpHandler*> CreateGpuOpHandler(CoreRuntime* runtime,
                                              RCReference<GpuDevice> device,
                                              OpHandler* fallback);

}  // namespace tfrt

#endif  // TFRT_BACKENDS_GPU_CORE_RUNTIME_GPU_OP_HANDLER_H_
