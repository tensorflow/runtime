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

//===- cwise_unary_kernels.h ------------------------------------*- C++ -*-===//
//
// Column wise unary kernels.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CWISE_UNARY_KERNELS_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CWISE_UNARY_KERNELS_H_

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {
namespace cpu {
namespace functor {

template <typename T, typename F, typename R = T>
struct UnaryFunctor {
  using Functor = F;
  using Input = T;
  using Output = R;
};

struct Log {
  template <typename T>
  using Functor = UnaryFunctor<T, Eigen::internal::scalar_log_op<T>>;
};

struct Log1p {
  template <typename T>
  using Functor = UnaryFunctor<T, Eigen::internal::scalar_log1p_op<T>>;
};

struct Rsqrt {
  template <typename T>
  using Functor = UnaryFunctor<T, Eigen::internal::scalar_rsqrt_op<T>>;
};

struct Sigmoid {
  template <typename T>
  using Functor = UnaryFunctor<T, Eigen::internal::scalar_logistic_op<T>>;
};

}  // namespace functor

template <typename UnaryFunctor, typename OnDone>
static void UnaryKernel(const DenseHostTensor& input, DenseHostTensor* output,
                        const ExecutionContext& exec_ctx, OnDone on_done) {
  using F = typename UnaryFunctor::Functor;
  using T = typename UnaryFunctor::Input;
  using R = typename UnaryFunctor::Output;

  HostContext* host = exec_ctx.host();
  auto& ctx = host->GetOrCreateSharedContext<compat::EigenHostContext>();

  auto input_t = compat::AsEigenConstTensor(DHTArrayView<T>(&input));
  auto output_t = compat::AsEigenTensor(MutableDHTArrayView<R>(output));

  auto expr = input_t.unaryExpr(F());

  compat::AsyncAssign(
      ctx, output_t, std::move(expr),
      [buffers = compat::KeepBuffers::alive(&input, output),
       on_done = std::move(on_done)]() { on_done(Error::success()); });
}

template <typename UnaryFunctor>
static AsyncValueRef<Chain> UnaryKernel(const DenseHostTensor& input,
                                        DenseHostTensor* output,
                                        const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  AsyncValueRef<Chain> chain = host->MakeConstructedAsyncValueRef<Chain>();

  auto on_done = [chain = chain.CopyRef()](Error err) {
    err ? chain.SetError(err) : chain.SetStateConcrete();
  };

  UnaryKernel<UnaryFunctor>(input, output, exec_ctx, std::move(on_done));

  return chain;
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CWISE_UNARY_KERNELS_H_
