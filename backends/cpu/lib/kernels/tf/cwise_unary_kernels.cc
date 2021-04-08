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

// Column wise unary Tensorflow kernels.

#ifndef TFRT_BACKENDS_CPU_KERNELS_TF_CWISE_UNARY_OPS_H_
#define TFRT_BACKENDS_CPU_KERNELS_TF_CWISE_UNARY_OPS_H_

#include "../cwise_unary_kernels.h"

#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/host_context/sync_kernel_utils.h"

namespace tfrt {
namespace {

template <typename FunctorT>
void RegisterUnaryKernel(KernelRegistry* registry, string_view kernel_name) {
#define DTYPE_FLOAT(ENUM)                                                 \
  {                                                                       \
    using CPP_TYPE = EigenTypeForDTypeKind<DType::ENUM>;                  \
    using UnaryFunctor = typename FunctorT::template Functor<CPP_TYPE>;   \
    registry->AddSyncKernel(                                              \
        ("tf_sync." + kernel_name + "." + GetDType<CPP_TYPE>().GetName()) \
            .str(),                                                       \
        TFRT_SYNC_KERNEL(cpu::SyncUnaryKernel<UnaryFunctor>));            \
  }
#include "tfrt/dtype/dtype.def"
}
}  // namespace

namespace tf {
void RegisterUnaryCpuKernels(KernelRegistry* registry) {
  RegisterUnaryKernel<cpu::functor::Log>(registry, "Log");
  RegisterUnaryKernel<cpu::functor::Log1p>(registry, "Log1p");
  RegisterUnaryKernel<cpu::functor::Rsqrt>(registry, "Rsqrt");
  RegisterUnaryKernel<cpu::functor::Sigmoid>(registry, "Sigmoid");
}
}  // namespace tf

}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_KERNELS_TF_CWISE_UNARY_OPS_H_
