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

//===- buffer_forwarding.cc -------------------------------------*- C++ -*-===//
//
// Tensorflow operations input buffer forwarding.
//
//===----------------------------------------------------------------------===//

#include "buffer_forwarding.h"

#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {

AsyncValueRef<DenseHostTensor> ForwardInputOrAllocateOutput(
    const ExecutionContext& exec_ctx, const TensorMetadata& output_md,
    ArrayRef<Argument<DenseHostTensor>> inputs) {
  HostContext* host = exec_ctx.host();

  // Try to find a compatible input tensor.
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Argument<DenseHostTensor>& input = inputs[i];

    // Check that we are the last user of the async value.
    if (!input.value()->IsUnique()) continue;

    // Check that output metadata is compatible with the input.
    if (input->dtype() != output_md.dtype) continue;
    if (input->shape() != output_md.shape) continue;

    // Check that no other tensors share the buffer with the input.
    if (!input->buffer()->IsExclusiveDataOwner()) continue;

    // We can't forward the input AsyncValue because we must return constructed
    // but not yet available value.
    auto dht = input->CopyRef();
    return host->MakeConstructedAsyncValueRef<DenseHostTensor>(std::move(dht));
  }

  AsyncValueRef<DenseHostTensor> allocated =
      DenseHostTensor::MakeConstructedAsyncValueRef(output_md, host);
  if (!allocated)
    return EmitErrorAsync(exec_ctx, "out of memory allocating result");

  return allocated;
}

}  // namespace tfrt
