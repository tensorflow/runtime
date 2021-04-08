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

// This file declares kernels for string host tensor.

#ifndef TFRT_TENSOR_STRING_HOST_TENSOR_KERNELS_H_
#define TFRT_TENSOR_STRING_HOST_TENSOR_KERNELS_H_

#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

void RegisterStringHostTensorKernels(KernelRegistry* registry);

}  // namespace tfrt

#endif  // TFRT_TENSOR_STRING_HOST_TENSOR_KERNELS_H_
