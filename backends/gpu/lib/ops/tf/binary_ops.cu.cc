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

// Collates list of all binary TF operations.

#include "eigen_helper.cu.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"

namespace tfrt {

// ComputeBinaryElementwiseOpViaEigen<Functor, K0, K1, K2> is a TFRT kernel for
// an operation implemented by the Eigen functor `Functor` and supports DType
// kinds K0, K1, K2 (via dynamic dispatch).

template <template <typename, typename> class Functor>
Expected<gpu::DenseGpuTensor> ComputeBinaryElementwiseOpViaEigen(
    GpuDispatchContext* dctx, const gpu::DenseGpuTensor& lhs,
    const gpu::DenseGpuTensor& rhs, const TensorMetadata& result_md) {
  return MakeStringError("unexpected type for operation");
}

template <template <typename, typename> class Functor, DType::Kind CurrentKind,
          DType::Kind... TrailingKinds>
Expected<gpu::DenseGpuTensor> ComputeBinaryElementwiseOpViaEigen(
    GpuDispatchContext* dctx, const gpu::DenseGpuTensor& lhs,
    const gpu::DenseGpuTensor& rhs, const TensorMetadata& result_md) {
  if (CurrentKind == result_md.dtype.kind()) {
    using T = EigenTypeForDTypeKind<CurrentKind>;
    using SpecializedFunctor = Functor<T, T>;
    return gpu::ComputeOpViaEigen<gpu::FunctorSignature<T, T, T>,
                                  SpecializedFunctor>(dctx, result_md,
                                                      {&lhs, &rhs});
  } else {
    return ComputeBinaryElementwiseOpViaEigen<Functor, TrailingKinds...>(
        dctx, lhs, rhs, result_md);
  }
}

void RegisterBinaryGpuTfOps(GpuOpRegistry* registry) {
  registry->AddOp("tf.AddV2", TFRT_GPU_OP(ComputeBinaryElementwiseOpViaEigen<
                                          Eigen::internal::scalar_sum_op,
                                          DType::F16, DType::F32, DType::F64>));
}
}  // namespace tfrt
