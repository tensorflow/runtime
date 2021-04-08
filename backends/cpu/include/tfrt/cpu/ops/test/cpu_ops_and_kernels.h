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

// CPU specific ops and kernels
//
// This file contains helpers that register CPU specific ops and TFRT kernels.
#ifndef TFRT_BACKENDS_CPU_OPS_TEST_CPU_OPS_AND_KERNELS_H_
#define TFRT_BACKENDS_CPU_OPS_TEST_CPU_OPS_AND_KERNELS_H_

#include "tfrt/support/forward_decls.h"

namespace tfrt {

class KernelRegistry;
class CpuOpRegistry;

void RegisterBTFIOKernels(KernelRegistry* registry);

void RegisterMNISTTensorKernels(KernelRegistry* registry);

void RegisterResNetTensorKernels(KernelRegistry* registry);

void RegisterTestMnistCpuOps(CpuOpRegistry* registry);
void RegisterTestCpuOps(CpuOpRegistry* registry);

void RegisterCooKernels(KernelRegistry* registry);
void RegisterCooCpuOps(CpuOpRegistry* registry);

}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_OPS_TEST_CPU_OPS_AND_KERNELS_H_
