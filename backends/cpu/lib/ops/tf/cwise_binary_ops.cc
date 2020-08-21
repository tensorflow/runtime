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

//===- cwise_binary_ops.cc - ------------------------------------*- C++ -*-===//
//
// Coefficient wise binary Tensorflow operations.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_OPS_TF_CWISE_UNARY_OPS_H_
#define TFRT_BACKENDS_CPU_OPS_TF_CWISE_UNARY_OPS_H_

#include "cwise_binary_ops.h"

#include "../../kernels/cwise_binary_kernels.h"
#include "buffer_forwarding.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/common/ops/tf/metadata_functions.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/scalar_host_tensor.h"
#include "tfrt/tensor/tensor_serialize_utils.h"
#include "type_dispatch.h"

namespace tfrt {
namespace {

template <typename BinaryFunctor>
static AsyncValueRef<HostTensor> TfBinaryOp(Argument<HostTensor> lhs,
                                            Argument<HostTensor> rhs,
                                            const TensorMetadata& output_md,
                                            const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  internal::NumericTypeDispatch type_dispatch(lhs->dtype());

  auto unsupported = [&](DType dtype) -> AsyncValueRef<HostTensor> {
    return MakeErrorAsyncValueRef(host,
                                  StrCat("Unsupported input dtype: ", dtype));
  };

  // ------------------------------------------------------------------------ //
  // Handle scalar + scalar case, output is a scalar tensor.
  // ------------------------------------------------------------------------ //
  if (isa<AnyScalarHostTensor>(*lhs) && isa<AnyScalarHostTensor>(*rhs)) {
    auto dispatch = [&](auto type_tag) -> AsyncValueRef<HostTensor> {
      using T = decltype(type_tag);
      using F = typename BinaryFunctor::template Functor<T>;
      using R = typename F::Output;
      auto output =
          MakeAvailableAsyncValueRef<ScalarHostTensor<R>>(host, output_md);
      cpu::BinaryKernel<F, compat::AsyncEigenEvaluator>(
          *lhs, *rhs, &output.get(), exec_ctx, [](Error err) {});
      return output;
    };

    return type_dispatch(dispatch, unsupported);
  }

  // ------------------------------------------------------------------------ //
  // Handle dense host tensor case, output is a dense host tensor.
  // ------------------------------------------------------------------------ //
  auto dispatch = [&](auto type_tag) -> AsyncValueRef<HostTensor> {
    // Forward input tensor or allocate new output tensor.
    auto output = ForwardInputOrAllocateOutput(exec_ctx, output_md, lhs, rhs);
    if (output.IsError()) return output;

    using T = decltype(type_tag);
    using F = typename BinaryFunctor::template Functor<T>;

    auto on_done = [output = output.CopyRef()](Error err) {
      // Forward errors to the tensor output.
      err ? output.SetError(err) : output.SetStateConcrete();
    };
    cpu::BinaryKernel<F, compat::AsyncEigenEvaluator>(
        *lhs, *rhs, &output.get(), exec_ctx, std::move(on_done));

    return output;
  };

  return type_dispatch(dispatch, unsupported);
}

template <typename Functor>
void RegisterTfBinaryOp(CpuOpRegistry* op_registry, string_view op_name) {
  op_registry->AddOp(op_name, TFRT_CPU_OP(TfBinaryOp<Functor>),
                     CpuOpFlags::NoSideEffects | CpuOpFlags::AllowsScalar);
}

}  // namespace

void RegisterTfBinaryCpuOps(CpuOpRegistry* op_registry) {
  RegisterTfBinaryOp<cpu::functor::Add>(op_registry, "tf.AddV2");
  RegisterTfBinaryOp<cpu::functor::Mul>(op_registry, "tf.Mul");
  RegisterTfBinaryOp<cpu::functor::Div>(op_registry, "tf.RealDiv");
  RegisterTfBinaryOp<cpu::functor::Sub>(op_registry, "tf.Sub");
  RegisterTfBinaryOp<cpu::functor::Less>(op_registry, "tf.Less");
}

}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_OPS_TF_CWISE_UNARY_OPS_H_
