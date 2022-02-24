/* Copyright 2021 The TensorFlow Runtime Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <utility>

#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/kernels/kernels_detail.h"
#include "tfrt/gpu/wrapper/ccl_wrapper.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/string_util.h"

namespace tfrt {
namespace gpu {

static Expected<GpuCclId> CclUniqueId(Attribute<int32_t> platform) {
  return wrapper::CclGetUniqueId(static_cast<wrapper::Platform>(*platform));
}

static AsyncValueRef<GpuCclHandle> CclCreate(Argument<GpuContext> context,
                                             int32_t rank, int32_t count,
                                             const GpuCclId& id,
                                             const ExecutionContext& exec_ctx) {
  // CclCommInitRank() blocks to wait for all participants and therefore needs
  // to run inside a blocking task.
  return RunBlockingWork(
      exec_ctx.host(),
      DestroyCapturesOnInvoke(
          [=, context = context.ValueRef()]() -> Expected<GpuCclHandle> {
            auto current = wrapper::CtxSetCurrent(context->get());
            if (!current) return current.takeError();
            auto comm = wrapper::CclCommInitRank(*current, count, id, rank);
            if (!comm) return comm.takeError();
            return GpuCclHandle(context.CopyRef(), std::move(*comm));
          }));
}

static Error CclAllGather(
    Argument<GpuCclHandle> handle, Argument<GpuBuffer> input,
    Argument<GpuBuffer> output,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> data_type) {
  auto type = static_cast<ncclDataType_t>(*data_type);
  auto width = wrapper::GetCclDataTypeSizeBytes(type);
  if (!width) return width.takeError();
  assert(*width != 0);

  handle->AddCallback([input = input.ValueRef(), output = output.ValueRef(),
                       count = input->size() / *width,
                       type](wrapper::CurrentContext current,
                             wrapper::Stream stream,
                             wrapper::CclComm comm) -> llvm::Error {
    return wrapper::CclAllGather(current, input->pointer(), output->pointer(),
                                 count, type, comm, stream);
  });

  return Error::success();
}

static Error CclAllReduce(
    Argument<GpuCclHandle> handle, Argument<GpuBuffer> input,
    Argument<GpuBuffer> output,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> data_type, Attribute<int32_t> reduction_op) {
  auto type = static_cast<ncclDataType_t>(*data_type);
  auto width = wrapper::GetCclDataTypeSizeBytes(type);
  if (!width) return width.takeError();
  assert(*width != 0);

  handle->AddCallback([input = input.ValueRef(), output = output.ValueRef(),
                       count = input->size() / *width, type,
                       op = static_cast<ncclRedOp_t>(*reduction_op)](
                          wrapper::CurrentContext current,
                          wrapper::Stream stream,
                          wrapper::CclComm comm) -> llvm::Error {
    return wrapper::CclAllReduce(current, input->pointer(), output->pointer(),
                                 count, type, op, comm, stream);
  });

  return Error::success();
}

static Error CclReduceScatter(
    Argument<GpuCclHandle> handle, Argument<GpuBuffer> input,
    Argument<GpuBuffer> output,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> data_type, Attribute<int32_t> reduction_op) {
  auto type = static_cast<ncclDataType_t>(*data_type);
  auto width = wrapper::GetCclDataTypeSizeBytes(type);
  if (!width) return width.takeError();
  assert(*width != 0);

  handle->AddCallback([input = input.ValueRef(), output = output.ValueRef(),
                       recvcount = output->size() / *width, type,
                       op = static_cast<ncclRedOp_t>(*reduction_op)](
                          wrapper::CurrentContext current,
                          wrapper::Stream stream,
                          wrapper::CclComm comm) -> llvm::Error {
    auto count = wrapper::CclCommCount(comm);
    if (!count) return count.takeError();
    if (input->size() != output->size() * *count)
      return MakeStringError("Input size must be output size times ranks.");
    return wrapper::CclReduceScatter(current, input->pointer(),
                                     output->pointer(), recvcount, type, op,
                                     comm, stream);
  });

  return Error::success();
}

static Error CclSend(Argument<GpuCclHandle> handle, Argument<GpuBuffer> input,
                     int32_t peer,
                     // Needs to be sorted alphabetically by attribute name!
                     Attribute<int32_t> data_type) {
  auto type = static_cast<ncclDataType_t>(*data_type);
  auto width = wrapper::GetCclDataTypeSizeBytes(type);
  if (!width) return width.takeError();
  assert(*width != 0);

  handle->AddCallback([input = input.ValueRef(), count = input->size() / *width,
                       type, peer](wrapper::CurrentContext current,
                                   wrapper::Stream stream,
                                   wrapper::CclComm comm) -> llvm::Error {
    return wrapper::CclSend(current, input->pointer(), count, type, peer, comm,
                            stream);
  });

  return Error::success();
}

static Error CclRecv(Argument<GpuCclHandle> handle, Argument<GpuBuffer> output,
                     int32_t peer,
                     // Needs to be sorted alphabetically by attribute name!
                     Attribute<int32_t> data_type) {
  auto type = static_cast<ncclDataType_t>(*data_type);
  auto width = wrapper::GetCclDataTypeSizeBytes(type);
  if (!width) return width.takeError();
  assert(*width != 0);

  handle->AddCallback(
      [output = output.ValueRef(), count = output->size() / *width, type, peer](
          wrapper::CurrentContext current, wrapper::Stream stream,
          wrapper::CclComm comm) -> llvm::Error {
        return wrapper::CclRecv(current, output->pointer(), count, type, peer,
                                comm, stream);
      });

  return Error::success();
}

static Error CclAllToAll(Argument<GpuCclHandle> handle,
                         Argument<GpuBuffer> input, Argument<GpuBuffer> output,
                         // Needs to be sorted alphabetically by attribute name!
                         Attribute<int32_t> data_type) {
  if (input->size() != output->size())
    return MakeStringError("Input size must equal output size.");

  auto type = static_cast<ncclDataType_t>(*data_type);
  auto width = wrapper::GetCclDataTypeSizeBytes(type);
  if (!width) return width.takeError();
  assert(*width != 0);

  auto comm_count = wrapper::CclCommCount(handle->get());
  if (!comm_count) return comm_count.takeError();
  assert(*comm_count > 0);

  size_t count = input->size() / (*width * *comm_count);  // Elements per chunk.
  size_t size = count * *width;                           // Bytes per chunk.
  if (input->size() != size * *comm_count)
    return MakeStringError(
        "Total element count must be exact multiple of comm count.");

  auto send_ptr = static_cast<wrapper::Pointer<char>>(input->pointer());
  auto recv_ptr = static_cast<wrapper::Pointer<char>>(output->pointer());

  for (int peer = 0; peer < *comm_count; ++peer) {
    handle->AddCallback([=](wrapper::CurrentContext current,
                            wrapper::Stream stream,
                            wrapper::CclComm comm) -> llvm::Error {
      return llvm::joinErrors(
          wrapper::CclSend(current, send_ptr, count, type, peer, comm, stream),
          wrapper::CclRecv(current, recv_ptr, count, type, peer, comm, stream));
    });
    send_ptr += size;
    recv_ptr += size;
  }

  // Add callback that simply holds on to a ref-count of input and output.
  handle->AddCallback([input = input.ValueRef(), output = output.ValueRef()](
                          auto...) { return Error::success(); });

  return Error::success();
}

static AsyncValueRef<Chain> CclExecute(Argument<GpuStream> stream,
                                       Argument<GpuCclHandle> handle,
                                       const ExecutionContext& exec_ctx) {
  // CclGroupEnd() blocks to wait for all participants and therefore needs to
  // run inside a blocking task.
  return RunBlockingWork(
      exec_ctx.host(),
      DestroyCapturesOnInvoke(
          [stream = stream.ValueRef(),
           handle = handle.ValueRef()]() -> Expected<Chain> {
            auto current = wrapper::CtxSetCurrent(stream->context()->get());
            if (!current) return current.takeError();
            if (auto error = handle->ExecuteCallbacks(*current, stream->get()))
              return std::move(error);
            return Chain();
          }));
}

void RegisterGpuCclKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_gpu.ccl.unique_id", TFRT_KERNEL(CclUniqueId));
  kernel_reg->AddKernel("tfrt_gpu.ccl.create", TFRT_KERNEL(CclCreate));
  kernel_reg->AddKernel("tfrt_gpu.ccl.all_gather",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(CclAllGather));
  kernel_reg->AddKernel("tfrt_gpu.ccl.all_reduce",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(CclAllReduce));
  kernel_reg->AddKernel("tfrt_gpu.ccl.reduce_scatter",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(CclReduceScatter));
  kernel_reg->AddKernel("tfrt_gpu.ccl.send",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(CclSend));
  kernel_reg->AddKernel("tfrt_gpu.ccl.recv",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(CclRecv));
  kernel_reg->AddKernel("tfrt_gpu.ccl.all_to_all",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(CclAllToAll));
  kernel_reg->AddKernel("tfrt_gpu.ccl.execute", TFRT_KERNEL(CclExecute));
}
}  // namespace gpu
}  // namespace tfrt
