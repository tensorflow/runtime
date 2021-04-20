// Copyright 2021 The TensorFlow Runtime Authors
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

// Implement a few thin wrapper for GPU APIs.

#include "tfrt/gpu/system/system.h"

#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace gpu {

Program::Program(AlignedBuffer<8>&& file_buffer, llvm::StringRef function_name,
                 HostContext* host)
    : file_buffer_(std::move(file_buffer)) {
  bef_file_ = tfrt::BEFFile::Open(file_buffer_, host->GetKernelRegistry(),
                                  host->diag_handler(), host->allocator());
  assert(bef_file_);
  function_ = bef_file_->GetFunction(function_name);
}

Stream::Stream(HostContext* host_ctx, wrapper::OwningContext context,
               wrapper::OwningStream stream,
               wrapper::OwningBlasHandle blas_handle)
    : context_(
          MakeAvailableAsyncValueRef<GpuContext>(host_ctx, std::move(context))),
      allocator_(MakeAvailableAsyncValueRef<GpuAllocator>(host_ctx,
                                                          context_.CopyRef())),
      stream_(MakeAvailableAsyncValueRef<GpuStream>(
          host_ctx, context_.CopyRef(), std::move(stream))),
      blas_handle_(MakeAvailableAsyncValueRef<GpuBlasHandle>(
          host_ctx, stream_.CopyRef(), std::move(blas_handle))) {}

/*static*/
Expected<Stream> Stream::Create(HostContext* host_ctx, int gpu_ordinal) {
  // NOTE(fishx): Right now we create a new context for each GPU stream.
  // TODO(tfrt-devs): Find a way to reuse the same context for multiple stream.
  auto device = wrapper::DeviceGet(wrapper::Platform::CUDA, gpu_ordinal);
  if (!device) return device.takeError();

  auto context = wrapper::CtxCreate(wrapper::CtxFlags::SCHED_AUTO, *device);
  if (!context) return context.takeError();
  auto current = wrapper::CtxSetCurrent(context->get());
  if (!current) return current.takeError();
  auto stream =
      wrapper::StreamCreate(*current, wrapper::StreamFlags::NON_BLOCKING);
  if (!stream) return stream.takeError();
  auto blas_handle = wrapper::BlasCreate(*current);
  if (!blas_handle) return blas_handle.takeError();
  if (auto error = wrapper::BlasSetStream(blas_handle->get(), stream->get()))
    return std::move(error);

  return Stream(host_ctx, std::move(*context), std::move(*stream),
                std::move(*blas_handle));
}

/*static*/
AsyncValueRef<System> System::Initialize(wrapper::Platform platform,
                                         llvm::StringRef prefix,
                                         HostContext* host) {
  if (auto error = wrapper::Init(platform))
    return tfrt::MakeErrorAsyncValueRef(host, DecodedDiagnostic(error));
  return MakeAvailableAsyncValueRef<System>(host, System{});
}

Expected<Stream> System::CreateStream(ExecutionContext& exec_ctx,
                                      int gpu_ordinal) {
  return Stream::Create(exec_ctx.host(), gpu_ordinal);
}

AsyncValueRef<GpuBuffer> System::Allocate(ExecutionContext& exec_ctx,
                                          Stream& stream, size_t size) {
  auto buffer = GpuBuffer::Allocate(stream.GetAllocator().CopyRef(), size,
                                    stream.GetStream()->get());
  if (!buffer) {
    return MakeErrorAsyncValueRef(exec_ctx.host(),
                                  DecodedDiagnostic(buffer.takeError()));
  }
  return MakeAvailableAsyncValueRef<GpuBuffer>(exec_ctx.host(),
                                               std::move(*buffer));
}

AsyncValueRef<Chain> System::TransferToDevice(ExecutionContext& exec_ctx,
                                              Stream& stream,
                                              AsyncValueRef<GpuBuffer> dst,
                                              ArrayRef<uint8_t> src,
                                              AsyncValueRef<Chain> chain) {
  auto out_chain = MakeUnconstructedAsyncValueRef<Chain>(exec_ctx.host());
  RunWhenReady(
      {dst.GetAsyncValue(), chain.GetAsyncValue()},
      [stream = stream.GetStream().CopyRef(), dst = std::move(dst), src,
       chain = std::move(chain), out_chain = out_chain.CopyRef()] {
        if (dst.IsError()) return out_chain.SetError(dst.GetError());

        if (chain.IsError()) return out_chain.SetError(chain.GetError());

        if (dst->size() < src.size()) {
          return out_chain.SetError(tfrt::StrCat(
              "TransferToDevice failed: "
              "destination buffer size (",
              dst->size(), ") is less than number of bytes to copy (",
              src.size(), ")"));
        }

        auto current = wrapper::CtxSetCurrent(stream->context());
        if (!current) return out_chain.SetError(current.takeError());

        if (auto error =
                wrapper::MemcpyAsync(*current, dst->pointer(),
                                     wrapper::Pointer<const void>(
                                         static_cast<const void*>(src.data()),
                                         wrapper::Platform::CUDA),
                                     src.size(), stream->get()))
          return out_chain.SetError(error);
        out_chain.emplace();
      });
  return out_chain;
}

AsyncValueRef<Chain> System::TransferFromDevice(ExecutionContext& exec_ctx,
                                                Stream& stream,
                                                MutableArrayRef<uint8_t> dst,
                                                AsyncValueRef<GpuBuffer> src,
                                                AsyncValueRef<Chain> chain) {
  auto out_chain = MakeUnconstructedAsyncValueRef<Chain>(exec_ctx.host());
  RunWhenReady(
      {src.GetAsyncValue(), chain.GetAsyncValue()},
      [exec_ctx, stream = stream.GetStream().CopyRef(), dst = std::move(dst),
       src = std::move(src), chain = std::move(chain),
       out_chain = out_chain.CopyRef()] {
        if (src.IsError()) return out_chain.SetError(src.GetError());

        if (chain.IsError()) return out_chain.SetError(chain.GetError());
        if (dst.size() < src->size()) {
          return out_chain.SetError(tfrt::StrCat(
              "TransferFromDevice failed: "
              "destination buffer size (",
              dst.size(), ") is less than number of bytes to copy (",
              src->size(), ")"));
        }

        auto current = wrapper::CtxSetCurrent(stream->context());
        if (!current) return out_chain.SetError(current.takeError());

        if (auto error = wrapper::MemcpyAsync(
                *current,
                wrapper::Pointer<void>(static_cast<void*>(dst.data()),
                                       wrapper::Platform::CUDA),
                src->pointer(), src->size(), stream->get()))
          return out_chain.SetError(error);

        // At this point, memcpy has been scheduled on the stream. However, the
        // dst buffer is not ready yet. We need to insert a gpu event to notify
        // the host when the memcpy is finished.
        auto event =
            wrapper::EventCreate(*current, wrapper::EventFlags::DISABLE_TIMING);
        if (!event) return out_chain.SetError(event.takeError());

        // Record the event on the stream.
        if (auto error = wrapper::EventRecord(event->get(), stream->get()))
          return out_chain.SetError(error);

        // EventSynchronize needs to be scheduled in the blocking work queue
        // because it will block caller thread until the event is completed.
        auto enqueued = EnqueueBlockingWork(
            exec_ctx,
            [event = std::move(*event), out_chain = out_chain.CopyRef()] {
              if (auto error = wrapper::EventSynchronize(event.get()))
                return out_chain.SetError(error);
              out_chain.emplace();
            });
        if (!enqueued) {
          return out_chain.SetError(
              "TransferFromDevice failed: failed to enqueue blocking work");
        }
      });
  return out_chain;
}

AsyncValueRef<Chain> System::Execute(ExecutionContext& exec_ctx,
                                     Program& program, Stream& stream,
                                     ArrayRef<AsyncValueRef<GpuBuffer>> inputs,
                                     ArrayRef<AsyncValueRef<GpuBuffer>> outputs,
                                     AsyncValueRef<Chain> chain) {
  auto out_chain = MakeUnconstructedAsyncValueRef<Chain>(exec_ctx.host());
  RunWhenReady(
      {chain.GetAsyncValue()},
      [exec_ctx, &program, context = stream.GetContext().CopyRef(),
       stream = stream.GetStream().CopyRef(),
       blas_handle = stream.GetBlasHandle().CopyRef(),
       inputs = std::move(inputs), outputs = std::move(outputs),
       chain = std::move(chain), out_chain = out_chain.CopyRef()] {
        if (chain.IsError()) return out_chain.SetError(chain.GetError());
        const Function* fn = program.GetFunction();
        assert(fn->num_results() == 0);
        auto num_args = fn->num_arguments();

        // Today, lowering pass for HLO will generate BEF Function that requires
        // following arguments:
        // {context, blas_handle, stream, ...inputs, ...outputs}
        // So we need to prepare and check the arguments first.
        SmallVector<AsyncValue*, 8> args;
        args.reserve(num_args);

        args.push_back(context.GetAsyncValue());
        args.push_back(blas_handle.GetAsyncValue());
        args.push_back(stream.GetAsyncValue());

        for (auto& input : inputs) {
          // TODO(b/184696034): Remove this once the bef function has proper
          // output.
          if (input.IsError()) return out_chain.SetError(chain.GetError());
          args.push_back(input.GetAsyncValue());
        }
        for (auto& output : outputs) {
          // TODO(b/184696034): Remove this once the bef function has proper
          // output.
          if (output.IsError()) return out_chain.SetError(output.GetError());
          args.push_back(output.GetAsyncValue());
        }

        if (args.size() != num_args) {
          return out_chain.SetError(StrCat(
              "Failed to execute lowered function: argument size mismatch: ",
              args.size(), " v.s. ", num_args));
        }

        // Right now, lowered bef function does not have output. This is
        // incorrect. We should either add an output to indicate all gpu
        // activities have been dispatched to stream OR we should change the
        // lowered function to synchronous bef function.
        // TODO(b/184696034): Fix this issue.
        fn->Execute(exec_ctx, args, {});

        out_chain.emplace();
      });
  return out_chain;
}

}  // namespace gpu
}  // namespace tfrt
