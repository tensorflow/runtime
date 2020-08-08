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

//===- softmax_kernels.cc - -------------------------------------*- C++ -*-===//
//
// Softmax Tensorflow kernels.
//
//===----------------------------------------------------------------------===//

#include "../softmax_kernel.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/common/compat/eigen/eigen_evaluator.h"
#include "tfrt/host_context/sync_kernel_utils.h"

namespace tfrt {
namespace {

template <typename T, bool log>
void SyncSoftmax(const DenseHostTensor& logits, DenseHostTensor* softmax,
                 const ExecutionContext& exec_ctx) {
  cpu::Softmax<T, log, compat::SyncEigenEvaluator>(logits, softmax, exec_ctx);
}

}  // namespace
namespace tf {

void RegisterSoftmaxCpuKernels(KernelRegistry* registry) {
#define DTYPE_FLOAT(ENUM)                                              \
  {                                                                    \
    using CPP_TYPE = EigenTypeForDTypeKind<DType::ENUM>;               \
    registry->AddSyncKernel(                                           \
        StrCat("tf_sync.Softmax.", GetDType<CPP_TYPE>().GetName()),    \
        TFRT_SYNC_KERNEL(SyncSoftmax<CPP_TYPE, false>));               \
    registry->AddSyncKernel(                                           \
        StrCat("tf_sync.LogSoftmax.", GetDType<CPP_TYPE>().GetName()), \
        TFRT_SYNC_KERNEL(SyncSoftmax<CPP_TYPE, true>));                \
  }
#include "tfrt/dtype/dtype.def"
}
}  // namespace tf

}  // namespace tfrt
