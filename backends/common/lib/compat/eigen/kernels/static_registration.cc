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
// kernels in this directory.  This can be used to simplify clients that don't
// care about selective registration of kernels.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/kernel_registry.h"

namespace tfrt {

void RegisterBatchNormKernels(KernelRegistry* registry);
void RegisterBatchNormGradKernels(KernelRegistry* registry);
void RegisterConv2DKernels(KernelRegistry* registry);
void RegisterConv2DBatchNormKernels(KernelRegistry* registry);
void RegisterConv2DBatchNormReluKernels(KernelRegistry* registry);
void RegisterConv2DBiasKernels(KernelRegistry* registry);
void RegisterConv2DGradFilterKernels(KernelRegistry* registry);
void RegisterConv2DGradInputKernels(KernelRegistry* registry);
void RegisterMatMulKernels(KernelRegistry* registry);
void RegisterMaxPoolingKernels(KernelRegistry* registry);
void RegisterZeroPaddingKernels(KernelRegistry* registry);

TFRT_STATIC_KERNEL_REGISTRATION(RegisterBatchNormKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterBatchNormGradKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterConv2DKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterConv2DBatchNormKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterConv2DBatchNormReluKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterConv2DBiasKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterConv2DGradFilterKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterConv2DGradInputKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterMatMulKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterMaxPoolingKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterZeroPaddingKernels);

}  // namespace tfrt
