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

// Softmax and LogSoftmax kernels implementation.

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_SOFTMAX_KERNEL_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_SOFTMAX_KERNEL_H_

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace cpu {

template <typename T, bool log, typename EigenEvaluator>
static typename EigenEvaluator::DependencyToken Softmax(
    const DenseHostTensor& logits, DenseHostTensor* softmax,
    const ExecutionContext& exec_ctx) {
  // TODO(b/172291736): Avoid creating another DHT by having a generic view
  // class that operates on only a shape and a pointer.
  DenseHostTensor reshaped_logits(
      TensorMetadata(logits.dtype(), GetFlattenedInnerDimsShape(
                                         logits.shape(), /*num_out_dims=*/2)),
      logits.buffer());
  DenseHostTensor reshaped_softmax(
      TensorMetadata(
          softmax->dtype(),
          GetFlattenedInnerDimsShape(softmax->shape(), /*num_out_dims=*/2)),
      softmax->buffer());
  DHTIndexableView<T, 2> logits_view(&reshaped_logits);
  MutableDHTIndexableView<T, 2> softmax_view(&reshaped_softmax);

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
  EigenEvaluator eigen{exec_ctx.host()};

  if (log) {
    // softmax = logits - max(logits along classes);
    auto logits_ready =
        eigen.Evaluate(softmax_t, std::move(shifted_logits_expr),
                       eigen.KeepAlive(&logits, softmax));

    // softmax = softmax - log(sum(exp(softmax along classes)));
    auto softmax_expr = (softmax_t - softmax_t.exp()
                                         .sum(along_class)
                                         .log()
                                         .eval()
                                         .reshape(batch_by_one)
                                         .broadcast(one_by_class));

    return eigen.Evaluate(logits_ready, softmax_t, std::move(softmax_expr),
                          eigen.KeepAlive(&logits, softmax));

  } else {
    // softmax = exp(logits - max(logits along classes));
    auto logits_ready =
        eigen.Evaluate(softmax_t, std::move(shifted_logits_expr.exp()),
                       eigen.KeepAlive(&logits, softmax));

    // softmax = softmax * (1 / sum(softmax along classes));
    auto softmax_expr = (softmax_t * softmax_t.sum(along_class)
                                         .inverse()
                                         .eval()
                                         .reshape(batch_by_one)
                                         .broadcast(one_by_class));

    return eigen.Evaluate(logits_ready, softmax_t, std::move(softmax_expr),
                          eigen.KeepAlive(&logits, softmax));
  }
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_SOFTMAX_KERNEL_H_
