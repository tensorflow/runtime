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

//===- pad_op_noncuda.cc - -------------------------------------*- C++ -*--===//
//
// Implements utilities to call tf.Pad op. This calls into EnqueueGpuPadOp from
// `tf_gpu_pad_op` cuda_library target. This avoids header clash between cudnn
#include "pad_op_noncuda.h"

#include "pad_op.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"

namespace tfrt {
namespace gpu {

// Enqueue a Pad op to GPU stream with the given `paddings`.
llvm::Expected<DenseGpuTensor> CallGpuPadOp(GpuDispatchContext* dctx,
                                            const DenseGpuTensor& input,
                                            const DenseView& paddings,
                                            const TensorMetadata& result_md) {
  return EnqueueGpuPadOp(dctx, input, paddings, result_md);
}

}  // namespace gpu
}  // namespace tfrt
