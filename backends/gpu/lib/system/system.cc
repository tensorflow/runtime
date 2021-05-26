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

Program::Program(BefBuffer&& file_buffer, llvm::StringRef function_name,
                 HostContext* host)
    : file_buffer_(std::move(file_buffer)) {
  bef_file_ = tfrt::BEFFile::Open(file_buffer_, host->GetKernelRegistry(),
                                  host->diag_handler(), host->allocator());
  assert(bef_file_);
  function_ = bef_file_->GetFunction(function_name);
}

/*static*/
AsyncValueRef<System> System::Initialize(wrapper::Platform platform,
                                         llvm::StringRef prefix,
                                         HostContext* host) {
  if (auto error = wrapper::Init(platform))
    return tfrt::MakeErrorAsyncValueRef(host, DecodedDiagnostic(error));
  return Instantiate(host);
}

/*static*/
AsyncValueRef<System> System::Instantiate(HostContext* host) {
  return MakeAvailableAsyncValueRef<System>(host, System{});
}

AsyncValueRef<GpuStream> System::CreateStream(ExecutionContext& exec_ctx,
                                              int gpu_ordinal) {
  // NOTE(fishx): Right now we create a new context for each GPU stream.
  // TODO(tfrt-devs): Find a way to reuse the same context for multiple stream.
  auto device = wrapper::DeviceGet(wrapper::Platform::CUDA, gpu_ordinal);
  if (!device) {
    return MakeErrorAsyncValueRef(exec_ctx.host(),
                                  DecodedDiagnostic(device.takeError()));
  }
  auto context = wrapper::CtxCreate(wrapper::CtxFlags::SCHED_AUTO, *device);
  if (!context) {
    return MakeErrorAsyncValueRef(exec_ctx.host(),
                                  DecodedDiagnostic(context.takeError()));
  }
  auto current = wrapper::CtxSetCurrent(context->get());
  if (!current) {
    return MakeErrorAsyncValueRef(exec_ctx.host(),
                                  DecodedDiagnostic(current.takeError()));
  }
  auto stream =
      wrapper::StreamCreate(*current, wrapper::StreamFlags::NON_BLOCKING);
  if (!stream) {
    return MakeErrorAsyncValueRef(exec_ctx.host(),
                                  DecodedDiagnostic(stream.takeError()));
  }

  auto gpu_context = MakeAvailableAsyncValueRef<GpuContext>(
      exec_ctx.host(), std::move(*context));
  return tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuStream>(
      exec_ctx.host(), std::move(gpu_context), std::move(*stream));
}

AsyncValueRef<GpuAllocator> System::CreateAllocator(
    ExecutionContext& exec_ctx, AsyncValueRef<GpuStream> stream) {
  return MakeAvailableAsyncValueRef<GpuDefaultAllocator>(exec_ctx.host(),
                                                         stream->gpu_context());
}

AsyncValueRef<GpuBuffer> System::Allocate(ExecutionContext& exec_ctx,
                                          AsyncValueRef<GpuStream> stream,
                                          AsyncValueRef<GpuAllocator> allocator,
                                          size_t size) {
  auto buffer = GpuBuffer::Allocate(std::move(allocator), size, stream->get());
  if (!buffer) {
    return MakeErrorAsyncValueRef(exec_ctx.host(),
                                  DecodedDiagnostic(buffer.takeError()));
  }
  return MakeAvailableAsyncValueRef<GpuBuffer>(exec_ctx.host(),
                                               std::move(*buffer));
}

AsyncValueRef<Chain> System::TransferToDevice(ExecutionContext& exec_ctx,
                                              AsyncValueRef<GpuStream> stream,
                                              AsyncValueRef<GpuBuffer> dst,
                                              ArrayRef<uint8_t> src,
                                              AsyncValueRef<Chain> chain) {
  auto out_chain = MakeUnconstructedAsyncValueRef<Chain>(exec_ctx.host());
  RunWhenReady({dst.GetAsyncValue(), chain.GetAsyncValue()},
               [stream = std::move(stream), dst = std::move(dst), src,
                chain = std::move(chain), out_chain = out_chain.CopyRef()] {
                 if (dst.IsError()) return out_chain.SetError(dst.GetError());

                 if (chain.IsError())
                   return out_chain.SetError(chain.GetError());

                 if (dst->size() < src.size()) {
                   return out_chain.SetError(tfrt::StrCat(
                       "TransferToDevice failed: "
                       "destination buffer size (",
                       dst->size(), ") is less than number of bytes to copy (",
                       src.size(), ")"));
                 }

                 auto current = wrapper::CtxSetCurrent(stream->context());
                 if (!current) return out_chain.SetError(current.takeError());

                 if (auto error = wrapper::MemcpyAsync(
                         *current, dst->pointer(),
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
                                                AsyncValueRef<GpuStream> stream,
                                                MutableArrayRef<uint8_t> dst,
                                                AsyncValueRef<GpuBuffer> src,
                                                AsyncValueRef<Chain> chain) {
  auto out_chain = MakeUnconstructedAsyncValueRef<Chain>(exec_ctx.host());
  RunWhenReady(
      {src.GetAsyncValue(), chain.GetAsyncValue()},
      [exec_ctx, stream = std::move(stream), dst = std::move(dst),
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
                                     Program& program,
                                     AsyncValueRef<GpuStream> stream,
                                     ArrayRef<AsyncValueRef<GpuBuffer>> inputs,
                                     ArrayRef<AsyncValueRef<GpuBuffer>> outputs,
                                     AsyncValueRef<Chain> chain) {
  const Function* fn = program.GetFunction();
  if (fn->num_results() != 1) {
    return MakeErrorAsyncValueRef(
        "Failed to execute lowered function: expected one result");
  }

  auto num_args = fn->num_arguments();

  // Lowering pass for HLO will generate BEF Function with the following
  // signature: {chain, stream, ...inputs, ...outputs} -> chain
  // So we need to prepare and check the arguments first.
  SmallVector<AsyncValue*, 8> args;
  args.reserve(num_args);

  args.push_back(chain.GetAsyncValue());
  args.push_back(stream.GetAsyncValue());

  for (auto& input : inputs) {
    args.push_back(input.GetAsyncValue());
  }
  for (auto& output : outputs) {
    args.push_back(output.GetAsyncValue());
  }

  if (args.size() != num_args) {
    return MakeErrorAsyncValueRef(
        StrCat("Failed to execute lowered function: argument size mismatch: ",
               args.size(), " v.s. ", num_args));
  }

  tfrt::RCReference<tfrt::AsyncValue> result;
  fn->Execute(exec_ctx, args, {result});

  return AsyncValueRef<Chain>(std::move(result));
}

}  // namespace gpu
}  // namespace tfrt
