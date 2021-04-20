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

/*static*/
Expected<Stream> Stream::Create(ExecutionContext& exec_ctx, int gpu_ordinal) {
  // NOTE(fishx): Right now we create a new context for each GPU stream.
  // TODO(tfrt-devs): Find a way to reuse the same context for multiple stream.
  auto expected_device =
      wrapper::DeviceGet(wrapper::Platform::CUDA, gpu_ordinal);
  if (!expected_device) return expected_device.takeError();

  auto expected_context =
      wrapper::CtxCreate(wrapper::CtxFlags::SCHED_AUTO, *expected_device);
  if (!expected_context) return expected_context.takeError();
  auto gpu_context = MakeAvailableAsyncValueRef<GpuContext>(
      exec_ctx.host(), std::move(*expected_context));

  auto current = wrapper::CtxSetCurrent(gpu_context->get());

  // Create a new stream.
  auto expected_stream =
      wrapper::StreamCreate(*current, wrapper::StreamFlags::NON_BLOCKING);
  if (!expected_stream) return expected_stream.takeError();
  auto gpu_stream = MakeAvailableAsyncValueRef<GpuStream>(
      exec_ctx.host(), gpu_context.CopyRef(), std::move(*expected_stream));

  // Create a new blas handle and associate the stream with it.
  auto expected_blas_handle = wrapper::BlasCreate(*current);
  if (!expected_blas_handle) return expected_blas_handle.takeError();

  llvm::Error error =
      wrapper::BlasSetStream(expected_blas_handle->get(), gpu_stream->get());
  if (error) return std::move(error);

  auto blas_handle = MakeAvailableAsyncValueRef<GpuBlasHandle>(
      exec_ctx.host(), gpu_stream.CopyRef(), std::move(*expected_blas_handle));

  return Stream{std::move(gpu_context), std::move(gpu_stream),
                std::move(blas_handle), gpu_ordinal};
}

/*static*/
AsyncValueRef<System> System::Initialize(wrapper::Platform platform,
                                         llvm::StringRef prefix,
                                         HostContext* host) {
  auto error = wrapper::Init(platform);
  if (error)
    return tfrt::MakeErrorAsyncValueRef(host, DecodedDiagnostic(error));

  auto expected_size = wrapper::DeviceGetCount(platform);
  if (!expected_size)
    return tfrt::MakeErrorAsyncValueRef(
        host, DecodedDiagnostic(expected_size.takeError()));

  // Create GPU devices.
  SmallVector<RCReference<GpuDevice>, 4> devices;
  devices.reserve(expected_size.get());
  for (int i = 0; i < expected_size.get(); ++i) {
    devices.emplace_back(TakeRef(new GpuDevice(StrCat(prefix, i), i)));
    error = devices[i]->Initialize();
    if (error)
      return tfrt::MakeErrorAsyncValueRef(host, DecodedDiagnostic(error));
  }

  return MakeAvailableAsyncValueRef<System>(host, std::move(devices));
}

Expected<Stream> System::CreateStream(ExecutionContext& exec_ctx,
                                      int gpu_ordinal) {
  if (gpu_ordinal >= static_cast<int>(devices_.size()))
    return MakeStringError(
        "Failed to create stream. gpu_ordinal: ", gpu_ordinal,
        " is bigger than gpu devices number: ", devices_.size());

  return Stream::Create(exec_ctx, gpu_ordinal);
}

AsyncValueRef<RCReference<GpuCrtBuffer>> System::Allocate(
    ExecutionContext& exec_ctx, Stream& stream, size_t size) {
  assert(stream.GetOrdinal() < devices_.size());
  auto allocator = devices_[stream.GetOrdinal()]->allocator();
  auto expected_buffer = allocator->Allocate(size, stream.GetStream()->get());
  if (!expected_buffer) {
    return MakeErrorAsyncValueRef(
        exec_ctx.host(), DecodedDiagnostic(expected_buffer.takeError()));
  }
  return MakeAvailableAsyncValueRef<RCReference<GpuCrtBuffer>>(
      exec_ctx.host(), std::move(*expected_buffer));
}

AsyncValueRef<Chain> System::TransferToDevice(
    ExecutionContext& exec_ctx, Stream& stream,
    AsyncValueRef<RCReference<GpuCrtBuffer>> dst, ArrayRef<uint8_t> src,
    AsyncValueRef<Chain> chain) {
  auto out_chain = MakeUnconstructedAsyncValueRef<Chain>(exec_ctx.host());
  RunWhenReady(
      {dst.GetAsyncValue(), chain.GetAsyncValue()},
      [stream = stream.GetStream().CopyRef(), dst = std::move(dst), src,
       chain = std::move(chain), out_chain = out_chain.CopyRef()] {
        if (dst.IsError()) {
          out_chain.SetError(dst.GetError());
          return;
        } else if (chain.IsError()) {
          out_chain.SetError(chain.GetError());
          return;
        }
        if (dst->get()->size() < src.size()) {
          out_chain.SetError(tfrt::StrCat(
              "TransferToDevice failed: "
              "destination buffer size (",
              dst->get()->size(), ") is less than number of bytes to copy (",
              src.size(), ")"));
          return;
        }

        auto expected_current_context =
            wrapper::CtxSetCurrent(stream->context());
        if (!expected_current_context) {
          out_chain.SetError(expected_current_context.takeError());
          return;
        }

        auto error = wrapper::MemcpyAsync(
            *expected_current_context, dst.get()->pointer(),
            wrapper::Pointer<const void>(static_cast<const void*>(src.data()),
                                         wrapper::Platform::CUDA),
            src.size(), stream->get());
        if (error) {
          out_chain.SetError(error);
          return;
        }
        out_chain.emplace();
      });
  return out_chain;
}

AsyncValueRef<Chain> System::TransferFromDevice(
    ExecutionContext& exec_ctx, Stream& stream, MutableArrayRef<uint8_t> dst,
    AsyncValueRef<RCReference<GpuCrtBuffer>> src, AsyncValueRef<Chain> chain) {
  auto out_chain = MakeUnconstructedAsyncValueRef<Chain>(exec_ctx.host());
  RunWhenReady(
      {src.GetAsyncValue(), chain.GetAsyncValue()},
      [exec_ctx, stream = stream.GetStream().CopyRef(),
       ordinal = stream.GetOrdinal(), dst = std::move(dst),
       src = std::move(src), chain = std::move(chain),
       out_chain = out_chain.CopyRef()] {
        if (src.IsError()) {
          out_chain.SetError(src.GetError());
          return;
        } else if (chain.IsError()) {
          out_chain.SetError(chain.GetError());
          return;
        }
        if (dst.size() < src->get()->size()) {
          out_chain.SetError(tfrt::StrCat(
              "TransferFromDevice failed: "
              "destination buffer size (",
              dst.size(), ") is less than number of bytes to copy (",
              src->get()->size(), ")"));
          return;
        }

        auto expected_current_context =
            wrapper::CtxSetCurrent(stream->context());
        if (!expected_current_context) {
          out_chain.SetError(expected_current_context.takeError());
          return;
        }

        auto error = wrapper::MemcpyAsync(
            *expected_current_context,
            wrapper::Pointer<void>(static_cast<void*>(dst.data()),
                                   wrapper::Platform::CUDA),
            src->get()->pointer(), src->get()->size(), stream->get());
        if (error) {
          out_chain.SetError(error);
          return;
        }

        // At this point, memcpy has been scheduled on the stream. However, the
        // dst buffer is not ready yet. We need to insert a gpu event to notify
        // the host when the memcpy is finished.
        auto expected_event = wrapper::EventCreate(
            *expected_current_context, wrapper::EventFlags::DISABLE_TIMING);
        if (!expected_event) {
          out_chain.SetError(expected_event.takeError());
          return;
        }

        // Record the event on the stream.
        if (auto error =
                wrapper::EventRecord(expected_event->get(), stream->get())) {
          out_chain.SetError(error);
          return;
        }

        // EventSynchronize needs to be scheduled in the blocking work queue
        // because it will block caller thread until the event is completed.
        auto enqueued =
            EnqueueBlockingWork(exec_ctx, [event = std::move(*expected_event),
                                           out_chain = out_chain.CopyRef()] {
              auto error = wrapper::EventSynchronize(event.get());
              if (error) {
                out_chain.SetError(error);
                return;
              }
              out_chain.emplace();
            });
        if (!enqueued) {
          out_chain.SetError(
              "TransferFromDevice failed: failed to enqueue blocking work");
          return;
        }
      });
  return out_chain;
}

AsyncValueRef<Chain> System::Execute(
    ExecutionContext& exec_ctx, Program& program, Stream& stream,
    ArrayRef<AsyncValueRef<RCReference<GpuCrtBuffer>>> inputs,
    ArrayRef<AsyncValueRef<RCReference<GpuCrtBuffer>>> outputs,
    AsyncValueRef<Chain> chain) {
  auto out_chain = MakeUnconstructedAsyncValueRef<Chain>(exec_ctx.host());
  RunWhenReady(
      {chain.GetAsyncValue()},
      [exec_ctx, &program, context = stream.GetContext().CopyRef(),
       stream = stream.GetStream().CopyRef(),
       blas_handle = stream.GetBlasHandle().CopyRef(),
       inputs = std::move(inputs), outputs = std::move(outputs),
       chain = std::move(chain), out_chain = out_chain.CopyRef()] {
        if (chain.IsError()) {
          out_chain.SetError(chain.GetError());
          return;
        }
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
          if (input.IsError()) {
            out_chain.SetError(chain.GetError());
            return;
          }
          args.push_back(input.GetAsyncValue());
        }
        for (auto& output : outputs) {
          // TODO(b/184696034): Remove this once the bef function has proper
          // output.
          if (output.IsError()) {
            out_chain.SetError(output.GetError());
            return;
          }
          args.push_back(output.GetAsyncValue());
        }

        if (args.size() != num_args) {
          out_chain.SetError(StrCat(
              "Failed to execute lowered function: argument size mismatch: ",
              args.size(), " v.s. ", num_args));
          return;
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
