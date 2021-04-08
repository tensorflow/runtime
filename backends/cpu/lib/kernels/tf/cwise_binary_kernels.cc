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

// Column wise binary Tensorflow kernels.

#include "../cwise_binary_kernels.h"

#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/host_context/sync_kernel_utils.h"

namespace tfrt {
namespace {

template <typename FunctorT>
void RegisterBinaryKernel(KernelRegistry* registry, string_view kernel_name) {
#define DTYPE_NUMERIC(ENUM)                                               \
  {                                                                       \
    using CPP_TYPE = EigenTypeForDTypeKind<DType::ENUM>;                  \
    using BinaryFunctor = typename FunctorT::template Functor<CPP_TYPE>;  \
    registry->AddSyncKernel(                                              \
        ("tf_sync." + kernel_name + "." + GetDType<CPP_TYPE>().GetName()) \
            .str(),                                                       \
        TFRT_SYNC_KERNEL(cpu::SyncBinaryKernel<BinaryFunctor>));          \
  }
#include "tfrt/dtype/dtype.def"
}
}  // namespace

namespace tf {
void RegisterBinaryCpuKernels(KernelRegistry* registry) {
  RegisterBinaryKernel<cpu::functor::Add>(registry, "AddV2");
  RegisterBinaryKernel<cpu::functor::Mul>(registry, "Mul");
  RegisterBinaryKernel<cpu::functor::Div>(registry, "RealDiv");
  RegisterBinaryKernel<cpu::functor::Sub>(registry, "Sub");
}
}  // namespace tf

}  // namespace tfrt
