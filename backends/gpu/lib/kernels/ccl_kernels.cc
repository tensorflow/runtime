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

#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/ccl_wrapper.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/string_util.h"

namespace tfrt {
namespace gpu {

namespace {

Expected<int> ToWidthInBytes(ncclDataType_t data_type) {
  switch (data_type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
#endif
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return MakeStringError(
          tfrt::StrCat("Unexpected ncclDataType_t: ", data_type));
  }
}

}  // namespace

static void CclAllReduce(Argument<GpuCclHandle> handle,
                         Argument<GpuBuffer> input, Argument<GpuBuffer> output,
                         Argument<Chain> in_chain, Result<Chain> out_chain,
                         // Needs to be sorted alphabetically by attribute name!
                         Attribute<int32_t> nccl_data_type,
                         Attribute<int32_t> reduction_op) {
  auto result = out_chain.Allocate();

  const auto data_type = static_cast<ncclDataType_t>(*nccl_data_type);
  auto expected_data_width = ToWidthInBytes(data_type);
  if (!expected_data_width) {
    result.SetError(expected_data_width.takeError());
    return;
  }
  assert(*expected_data_width != 0);
  const size_t element_count = input->size() / *expected_data_width;

  const auto op = static_cast<ncclRedOp_t>(*reduction_op);
  handle->AddCallback([input = input.value()->AddRef(),
                       output = output.value()->AddRef(), element_count,
                       data_type, op](const wrapper::CurrentContext current,
                                      wrapper::Stream stream,
                                      wrapper::CclComm comm) -> llvm::Error {
    return wrapper::CclAllReduce(current, input->get<GpuBuffer>().pointer(),
                                 output->get<GpuBuffer>().pointer(),
                                 element_count, data_type, op, comm, stream);
  });

  result.emplace();
}

static AsyncValueRef<Chain> CclExecute(Argument<GpuStream> stream,
                                       Argument<GpuCclHandle> handle,
                                       const ExecutionContext& exec_ctx) {
  return RunBlockingWork(
      exec_ctx,
      [stream = stream.ValueRef(),
       handle = handle.ValueRef()]() -> Expected<Chain> {
        auto current = wrapper::CtxSetCurrent(stream->context());
        if (!current) return current.takeError();
        if (auto error = wrapper::CclGroupStart(current->platform()))
          return std::move(error);
        if (auto error = handle->ExecuteCallbacks(*current, stream->get()))
          return std::move(error);
        if (auto error = wrapper::CclGroupEnd(current->platform()))
          return std::move(error);
        return Chain();
      });
}

void RegisterGpuCclKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_gpu.ccl.all_reduce", TFRT_KERNEL(CclAllReduce));
  kernel_reg->AddKernel("tfrt_gpu.ccl.execute", TFRT_KERNEL(CclExecute));
}
}  // namespace gpu
}  // namespace tfrt
