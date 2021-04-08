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

// Kernel interface to CoreRuntime
//
// This library contains kernels that allows the bef_executor to drive the core
// runtime.

#ifndef TFRT_CORE_RUNTIME_KERNELS_H_
#define TFRT_CORE_RUNTIME_KERNELS_H_

namespace tfrt {
class KernelRegistry;

void RegisterCoreRuntimeKernels(KernelRegistry* registry);

void RegisterLoggingOpHandlerKernel(KernelRegistry* registry);

void RegisterCompositeOpHandlerKernels(KernelRegistry* registry);

void RegisterCoreRuntimeTestKernels(KernelRegistry* registry);

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_KERNELS_H_
