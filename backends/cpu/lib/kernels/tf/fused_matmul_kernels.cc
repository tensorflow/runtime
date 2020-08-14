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

//===- fused_matmul_kernels.cc - --------------------------------*- C++ -*-===//
//
// FusedMatmul Tensorflow kernels.
//
//===----------------------------------------------------------------------===//

#include "../fused_matmul_kernel.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/common/compat/eigen/eigen_evaluator.h"
#include "tfrt/host_context/sync_kernel_utils.h"

namespace tfrt {

namespace {
template <typename T>
Error FusedMatMul(const DenseHostTensor& a, const DenseHostTensor& b,
                  DenseHostTensor* output,
                  RepeatedSyncArguments<DenseHostTensor> fusion_inputs,
                  AggregateAttr fused_ops_attr, Attribute<bool> transpose_a,
                  Attribute<bool> transpose_b,
                  const ExecutionContext& exec_ctx) {
  return cpu::FusedMatMul<T, compat::SyncEigenEvaluator>(
      a, b, output, fusion_inputs, *transpose_a, *transpose_b, fused_ops_attr,
      exec_ctx);
}
}  // namespace
namespace tf {

void RegisterFusedMatmulKernels(KernelRegistry* registry) {
#define DTYPE_FLOAT(ENUM)                                                \
  {                                                                      \
    using CPP_TYPE = EigenTypeForDTypeKind<DType::ENUM>;                 \
    registry->AddSyncKernel(                                             \
        StrCat("tf_sync._FusedMatMul.", GetDType<CPP_TYPE>().GetName()), \
        TFRT_SYNC_KERNEL(FusedMatMul<CPP_TYPE>));                        \
  }
#include "tfrt/dtype/dtype.def"
}
}  // namespace tf

}  // namespace tfrt
