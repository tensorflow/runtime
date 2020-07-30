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

//===- buffer_forwarding.h - ------------------------------------*- C++ -*-===//
//
// Tensorflow operations input buffer forwarding.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_OPS_TF_BUFFER_FORWARDING_H_
#define TFRT_BACKENDS_CPU_OPS_TF_BUFFER_FORWARDING_H_

#include <type_traits>

#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {

// Forwards one of the input buffers to the output, or allocates a new
// DenseHostTensor if buffer forwarding is not possible.
AsyncValueRef<DenseHostTensor> ForwardInputOrAllocateOutput(
    const ExecutionContext& exec_ctx, const TensorMetadata& output_md,
    ArrayRef<Argument<DenseHostTensor>> inputs);

// ForwardInputOrAllocateOutput overload that supports arguments of different
// tensor types. Forwards only inputs that are DenseHostTensors.
template <typename Tensor, typename = std::enable_if_t<
                               !std::is_same<Tensor, DenseHostTensor>::value>>
AsyncValueRef<DenseHostTensor> ForwardInputOrAllocateOutput(
    const ExecutionContext& exec_ctx, const TensorMetadata& output_md,
    ArrayRef<Argument<Tensor>> inputs) {
  static_assert(std::is_base_of<HostTensor, Tensor>::value,
                "Argument must be a HostTensor");

  SmallVector<Argument<DenseHostTensor>, 4> dht_inputs;
  dht_inputs.reserve(inputs.size());

  for (int i = 0; i < inputs.size(); ++i) {
    if (isa<DenseHostTensor>(*inputs[i]))
      dht_inputs.emplace_back(inputs[i].value());
  }

  return ForwardInputOrAllocateOutput(exec_ctx, output_md, dht_inputs);
}

template <typename Tensor>
AsyncValueRef<DenseHostTensor> ForwardInputOrAllocateOutput(
    const ExecutionContext& exec_ctx, const TensorMetadata& output_md,
    Argument<Tensor> input) {
  std::array<Argument<Tensor>, 1> inputs{input};
  return ForwardInputOrAllocateOutput(exec_ctx, output_md,
                                      ArrayRef<Argument<Tensor>>(inputs));
}

template <typename Tensor>
AsyncValueRef<DenseHostTensor> ForwardInputOrAllocateOutput(
    const ExecutionContext& exec_ctx, const TensorMetadata& output_md,
    Argument<Tensor> input0, Argument<Tensor> input1) {
  std::array<Argument<Tensor>, 2> inputs{input0, input1};
  return ForwardInputOrAllocateOutput(exec_ctx, output_md,
                                      ArrayRef<Argument<Tensor>>(inputs));
}

}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_OPS_TF_BUFFER_FORWARDING_H_
