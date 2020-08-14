// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- static_registration.cc ---------------------------------------------===//
//
// This file uses a static constructor to automatically register all of the
// kernels in this directory. This can be used to simplify clients that don't
// care about selective registration of kernels.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/kernel_registry.h"

namespace tfrt {
namespace tf {

void RegisterUnaryCpuKernels(KernelRegistry* registry);
void RegisterBinaryCpuKernels(KernelRegistry* registry);
void RegisterSoftmaxCpuKernels(KernelRegistry* registry);
void RegisterConstCpuKernels(KernelRegistry* registry);
void RegisterFusedMatmulKernels(KernelRegistry* registry);
void RegisterConcatCpuKernels(KernelRegistry* registry);
void RegisterTileCpuKernels(KernelRegistry* registry);

TFRT_STATIC_KERNEL_REGISTRATION(RegisterUnaryCpuKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterBinaryCpuKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterSoftmaxCpuKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterConstCpuKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterFusedMatmulKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterConcatCpuKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterTileCpuKernels);

}  // namespace tf
}  // namespace tfrt
