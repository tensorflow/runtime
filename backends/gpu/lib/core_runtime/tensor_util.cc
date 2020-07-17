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

//===- gpu_device/tensor_util.cc ------------------------------------------===//
//
// This file implements GPU tensor util functions for copying between gpu and
// host.
//
//===----------------------------------------------------------------------===//

#include "tfrt/gpu/core_runtime/tensor_util.h"

#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {
namespace gpu {

using gpu::stream::EventFlags;
using gpu::stream::EventRecord;
using gpu::stream::EventSynchronize;
using gpu::stream::OwningEvent;
using gpu::stream::Pointer;

AsyncValueRef<DenseHostTensor> CopyDenseGpuTensorToHost(
    GpuDispatchContext* dctx, const DenseGpuTensor& gpu_tensor,
    HostContext* host) {
  llvm::Optional<DenseHostTensor> result_or_error =
      DenseHostTensor::CreateUninitialized(gpu_tensor.metadata(), host);
  if (!result_or_error) {
    return host->MakeErrorAsyncValueRef("cannot allocate result tensor");
  }
  DenseHostTensor result = std::move(*result_or_error);

  Pointer<void> memcpy_dst(result.data(), dctx->current_context().platform());
  Pointer<void> memcpy_src = gpu_tensor.buffer().pointer();

  size_t size_in_bytes = gpu_tensor.metadata().GetHostSizeInBytes();
  llvm::Error memcpy_error =
      MemcpyAsync(dctx->current_context(), /*dst=*/memcpy_dst,
                  /*src=*/memcpy_src, size_in_bytes, dctx->stream());
  if (memcpy_error) {
    return host->MakeErrorAsyncValueRef(
        "failed to enqueue host to device memcpy: " +
        toString(std::move(memcpy_error)));
  }

  llvm::Expected<OwningEvent> event_or_error =
      EventCreate(dctx->current_context(), EventFlags::DISABLE_TIMING);
  if (!event_or_error) {
    return host->MakeErrorAsyncValueRef(
        "could not create event to wait for host to device memcpy: " +
        toString(event_or_error.takeError()));
  }

  OwningEvent event = std::move(*event_or_error);

  llvm::Error event_record_error = EventRecord(event.get(), dctx->stream());
  if (event_record_error) {
    return host->MakeErrorAsyncValueRef(
        "could not enqueue event to wait for host to device memcpy: " +
        toString(std::move(event_record_error)));
  }

  return host->EnqueueBlockingWork(
      [event = std::move(event),
       result = std::move(result)]() mutable -> Expected<DenseHostTensor> {
        if (EventSynchronize(event.get())) {
          return MakeStringError("could not wait for event");
        } else {
          return std::move(result);
        }
      });
}

Expected<DenseGpuTensor> CopyDenseHostTensorToGpu(GpuDispatchContext* dctx,
                                                  const DenseHostTensor& tensor,
                                                  HostContext* host) {
  size_t size_in_bytes = tensor.metadata().GetHostSizeInBytes();

  llvm::Expected<RCReference<gpu::GpuBuffer>> buffer_or_error =
      dctx->allocator()->Allocate(
          /*size=*/size_in_bytes, dctx->stream());
  if (!buffer_or_error) return buffer_or_error.takeError();
  RCReference<gpu::GpuBuffer> buffer = std::move(*buffer_or_error);

  Pointer<const void> memcpy_src(tensor.data(),
                                 dctx->current_context().platform());
  if (auto error =
          MemcpyAsync(dctx->current_context(), /*dst=*/buffer->pointer(),
                      /*src=*/memcpy_src, size_in_bytes, dctx->stream()))
    return std::move(error);

  llvm::Expected<OwningEvent> event_or_error =
      EventCreate(dctx->current_context(), EventFlags::DISABLE_TIMING);
  if (!event_or_error) return event_or_error.takeError();

  OwningEvent event = std::move(*event_or_error);

  if (auto error = EventRecord(event.get(), dctx->stream()))
    return std::move(error);

  // The underlying buffer of `tensor` needs to live until the memcpy is done.
  bool work_enqueued = host->EnqueueBlockingWork(
      [tensor = tensor.CopyRef(), event = std::move(event)] {
        // FIXME(sanjoy): How do we handle an error from EventSynchronize here?
        llvm::ExitOnError die_if_error;
        die_if_error(EventSynchronize(event.get()));
      });
  if (!work_enqueued) {
    return MakeStringError(
        "could not enqueue work to synchronize after issuing host to device "
        "memcpy in CreateDenseTensorOp");
  }
  return gpu::DenseGpuTensor(tensor.metadata(), std::move(buffer));
}

}  // namespace gpu
}  // namespace tfrt
