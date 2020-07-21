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

//===- softmax_kernel.h -----------------------------------------*- C++ -*-===//
//
// Softmax and LogSoftmax kernels implementation.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_SOFTMAX_KERNEL_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_SOFTMAX_KERNEL_H_

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {
namespace cpu {

template <typename T>
static AsyncValueRef<Chain> Softmax(const DenseHostTensor& logits, bool log,
                                    DenseHostTensor* softmax,
                                    const ExecutionContext& exec_ctx) {
  DHTIndexableView<T, 2> logits_view(&logits);
  MutableDHTIndexableView<T, 2> softmax_view(softmax);

  auto logits_t = compat::AsEigenConstTensor(logits_view);
  auto softmax_t = compat::AsEigenTensor(softmax_view);

  static constexpr int kBatchDim = 0;
  static constexpr int kClassDim = 1;

  const int batch_size = logits_t.dimension(kBatchDim);
  const int num_classes = logits_t.dimension(kClassDim);

  // Reduce along the class dimension.
  Eigen::IndexList<Eigen::type2index<kClassDim>> along_class;

  // Broadcast from [num_classes] to [batch, num_classes]
  Eigen::IndexList<int, Eigen::type2index<1>> batch_by_one;
  batch_by_one.set(0, batch_size);

  // Broadcast from [batch] to [batch, num_classes]
  Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
  one_by_class.set(1, num_classes);

  // In both cases we first pre-compute temporary logits reduction into the
  // output (softmax) tensor, and after that evaluate a second expression that
  // uses pre-computed values to get a final result.

  // shifted_logits = logits - max(logits along classes);
  auto shifted_logits_expr = (logits_t - logits_t.maximum(along_class)
                                             .eval()
                                             .reshape(batch_by_one)
                                             .broadcast(one_by_class));

  auto& ctx =
      exec_ctx.host()->GetOrCreateSharedContext<compat::EigenHostContext>();

  if (log) {
    // softmax = logits - max(logits along classes);
    AsyncValueRef<Chain> logits_ready =
        AsyncAssign(ctx, softmax_t, std::move(shifted_logits_expr),
                    compat::KeepBuffers::alive(&logits, softmax));

    // softmax = softmax - log(sum(exp(softmax along classes)));
    auto softmax_expr = (softmax_t - softmax_t.exp()
                                         .sum(along_class)
                                         .log()
                                         .eval()
                                         .reshape(batch_by_one)
                                         .broadcast(one_by_class));

    return AsyncAssign(ctx, logits_ready, softmax_t, std::move(softmax_expr),
                       compat::KeepBuffers::alive(&logits, softmax));

  } else {
    // softmax = exp(logits - max(logits along classes));
    AsyncValueRef<Chain> logits_ready =
        AsyncAssign(ctx, softmax_t, std::move(shifted_logits_expr.exp()),
                    compat::KeepBuffers::alive(&logits, softmax));

    // softmax = softmax * (1 / sum(softmax along classes));
    auto softmax_expr = (softmax_t * softmax_t.sum(along_class)
                                         .inverse()
                                         .eval()
                                         .reshape(batch_by_one)
                                         .broadcast(one_by_class));

    return AsyncAssign(ctx, logits_ready, softmax_t, std::move(softmax_expr),
                       compat::KeepBuffers::alive(&logits, softmax));
  }
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_SOFTMAX_KERNEL_H_
