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

//===- nullary_ops.cc -------------------------------------------*- C++ -*-===//
//
// Collates list of all nullary TF operations.
//
//===----------------------------------------------------------------------===//

#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"
#include "tfrt/gpu/memory/gpu_buffer.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {
namespace gpu {

static llvm::Expected<DenseGpuTensor> GpuConstOp(
    GpuDispatchContext* dctx, const OpAttrsRef& attrs,
    const TensorMetadata& result_md, const ExecutionContext& exec_ctx) {
  size_t size_in_bytes = result_md.GetHostSizeInBytes();

  TFRT_ASSIGN_OR_RETURN(RCReference<GpuBuffer> buffer,
                        dctx->allocator()->Allocate(
                            /*size=*/size_in_bytes, dctx->stream()));

  if (size_in_bytes == 0) {
    return DenseGpuTensor(result_md.shape, result_md.dtype, std::move(buffer));
  }

  // Make a copy of attrs on the heap.
  OpAttrsRef frozen_attrs = attrs.freeze();
  tfrt::DenseAttr dense_attr =
      frozen_attrs.GetAsserting<tfrt::DenseAttr>("value");

  auto dense_view = CreateDenseView(dense_attr);
  stream::Pointer<const void> memcpy_src(dense_view.data(),
                                         dctx->current_context().platform());
  if (auto error = stream::MemcpyAsync(dctx->current_context(),
                                       /*dst=*/buffer->pointer(),
                                       /*src=*/memcpy_src, size_in_bytes,
                                       dctx->stream())) {
    return std::move(error);
  }

  TFRT_ASSIGN_OR_RETURN(
      auto event, stream::EventCreate(dctx->current_context(),
                                      stream::EventFlags::DISABLE_TIMING));

  if (auto error = stream::EventRecord(event.get(), dctx->stream()))
    return std::move(error);

  // `frozen_attrs` needs to live until the memcpy is done.
  bool work_enqueued = exec_ctx.host()->EnqueueBlockingWork(
      [frozen_attrs = std::move(frozen_attrs), event = std::move(event)] {
        // FIXME(sanjoy): How do we handle an error from EventSynchronize here?
        llvm::ExitOnError die_if_error;
        die_if_error(stream::EventSynchronize(event.get()));
      });
  if (!work_enqueued)
    return MakeStringError("enqueue to blocking work queue failed");

  return DenseGpuTensor(result_md.shape, result_md.dtype, std::move(buffer));
}

}  // namespace gpu

void RegisterNullaryGpuTfOps(GpuOpRegistry* registry) {
  registry->AddOp("tf.Const", TFRT_GPU_OP(gpu::GpuConstOp), {"value"});
}

}  // namespace tfrt
