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

// This file uses a static constructor to automatically register all of the
// kernels in this directory.  This can be used to simplify clients that don't
// care about selective registration of kernels.

#include "tfrt/host_context/kernel_registry.h"

namespace tfrt {
namespace cuda {

void RegisterCudaKernels(KernelRegistry* kernel_reg);
void RegisterCudaBlasKernels(KernelRegistry* kernel_reg);
void RegisterCudaDnnKernels(KernelRegistry* kernel_reg);

namespace kernels {

TFRT_STATIC_KERNEL_REGISTRATION(RegisterCudaKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterCudaBlasKernels);
TFRT_STATIC_KERNEL_REGISTRATION(RegisterCudaDnnKernels);

}  // namespace kernels
}  // namespace cuda
}  // namespace tfrt
