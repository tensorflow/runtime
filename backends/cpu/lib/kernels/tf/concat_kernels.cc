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

// Concat Tensorflow kernels.

#include "../concat_kernel.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/common/compat/eigen/eigen_evaluator.h"
#include "tfrt/host_context/sync_kernel_utils.h"

namespace tfrt {
namespace {

template <typename T>
Error ConcatKernel(int axis, RepeatedSyncArguments<DenseHostTensor> args) {
  auto inputs = views::Counted(args.begin(), args.size() - 1);
  auto& output = args[args.size() - 1];
  return cpu::ConcatKernel<T>(inputs, axis, &output);
}
}  // namespace
namespace tf {

void RegisterConcatCpuKernels(KernelRegistry* registry) {
#define DTYPE_FLOAT(ENUM)                                                      \
  {                                                                            \
    using CPP_TYPE = EigenTypeForDTypeKind<DType::ENUM>;                       \
    registry->AddSyncKernel(StrCat("tf_sync.ConcatV2.", GetDType<CPP_TYPE>()), \
                            TFRT_SYNC_KERNEL(ConcatKernel<CPP_TYPE>));         \
  }
#include "tfrt/dtype/dtype.def"
}
}  // namespace tf

}  // namespace tfrt
