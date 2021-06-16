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

// This file implements the tfrt_gpu kernels that talk to the driver API.
#include <memory>
#include <tuple>

#include "kernels_detail.h"
#include "llvm/Support/Error.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace gpu {

// tfrt_gpu.device.get returns the gpu device at the given index.
static Expected<wrapper::Device> GpuDeviceGet(int32_t ordinal,
                                              Attribute<int32_t> platform) {
  auto wrapper_platform = static_cast<wrapper::Platform>(*platform);
  if (auto error = wrapper::Init(wrapper_platform)) return std::move(error);
  return wrapper::DeviceGet(wrapper_platform, ordinal);
}

// tfrt_gpu.context.create creates a gpu context for the given
// device.
static Expected<GpuContext> GpuContextCreate(wrapper::Device device) {
  auto context = wrapper::CtxCreate(wrapper::CtxFlags::SCHED_AUTO, device);
  if (!context) return context.takeError();
  return GpuContext(std::move(*context));
}

// tfrt_gpu.stream.create creates a new stream that does not implicitly
// synchronize with stream 0.
static Expected<GpuStream> GpuStreamCreate(Argument<GpuContext> context) {
  auto current = wrapper::CtxSetCurrent(context->get());
  if (!current) return current.takeError();
  auto stream =
      wrapper::StreamCreate(*current, wrapper::StreamFlags::NON_BLOCKING);
  if (!stream) return stream.takeError();
  return GpuStream(context.ValueRef(), std::move(*stream));
}

// tfrt_gpu.stream.get_context returns the context the stream was created with.
static AsyncValueRef<GpuContext> GpuStreamGetContext(const GpuStream& stream) {
  return stream.gpu_context();
}

// tfrt_gpu.stream.wait makes a stream wait on an event.
static Error GpuStreamWait(const GpuStream& stream, const GpuEvent& event) {
  return wrapper::StreamWaitEvent(stream.get(), event.get());
}

// tfrt_gpu.stream.synchronize waits until all stream's tasks are completed.
//
// Result: Sets the output chain when all tasks submitted on a stream are
// completed. This kernel will block the caller thread.
static void GpuStreamSynchronize(Argument<GpuStream> stream, Chain in_chain,
                                 Result<Chain> out_chain,
                                 const ExecutionContext& exec_ctx) {
  auto result = out_chain.Allocate();
  bool enqueued = EnqueueBlockingWork(
      exec_ctx, [result = result.CopyRef(), stream = stream.ValueRef(),
                 in_chain = in_chain]() mutable {
        if (auto error = wrapper::StreamSynchronize(stream->get()))
          return result.SetError(StrCat(error));
        result.emplace(in_chain);
      });
  if (!enqueued) return result.SetError("Failed to enqueue blocking work.");
}

// tfrt_gpu.event.create creates a new cuda event.
//
// Result: new cuda event.
static Expected<GpuEvent> GpuEventCreate(Argument<GpuContext> context) {
  auto current = wrapper::CtxSetCurrent(context->get());
  if (!current) return current.takeError();
  auto event =
      wrapper::EventCreate(*current, wrapper::EventFlags::DISABLE_TIMING);
  if (!event) return event.takeError();
  return GpuEvent(context.ValueRef(), std::move(*event));
}

// tfrt_gpu.event.record records an event on a stream.
static Error GpuEventRecord(const GpuEvent& event, const GpuStream& stream) {
  return wrapper::EventRecord(event.get(), stream.get());
}

// tfrt_gpu.event.synchronize sets the output chain when the event has been
// reached, i.e. all work scheduled prior to the last call to
// tfrt_gpu.event.record has been completed.
static void GpuEventSynchronize(Argument<GpuEvent> event, Chain in_chain,
                                Result<Chain> out_chain,
                                const ExecutionContext& exec_ctx) {
  auto result = out_chain.Allocate();
  // Check if event has already completed and we can skip enqueuing work.
  auto ready = wrapper::EventQuery(event->get());
  if (!ready) return result.SetError(StrCat(ready.takeError()));
  if (*ready) return result.emplace(in_chain);
  bool enqueued = EnqueueBlockingWork(
      exec_ctx, [result = result.CopyRef(), event = event.ValueRef(),
                 in_chain = in_chain]() mutable {
        if (auto error = wrapper::EventSynchronize(event->get()))
          return result.SetError(StrCat(error));
        result.emplace(in_chain);
      });
  if (!enqueued) return result.SetError("Failed to enqueue blocking work.");
}

// tfrt_gpu.allocator.create creates a new allocator.
//
// Result: new allocator.
static GpuDefaultAllocator GpuAllocatorCreate(Argument<GpuContext> context) {
  return GpuDefaultAllocator(context.ValueRef());
}

// tfrt_gpu.mem.allocate allocates a new gpu buffer.
static Expected<GpuBuffer> GpuMemAllocate(Argument<GpuAllocator> allocator,
                                          const GpuStream& stream,
                                          int64_t size) {
  return GpuBuffer::Allocate(allocator.ValueRef(), size, stream.get());
}

// tfrt_gpu.mem.print_metadata prints `buffer`'s metadata.
static void GpuMemPrintMetadata(const GpuBuffer& buffer) {
  // The check for buffer validity is not done intentionally. Printing invalid
  // buffers can be useful for debugging.
  tfrt::outs() << "GpuBuffer<pointer=" << buffer.pointer()
               << ", size=" << buffer.size() << ">";
  tfrt::outs().flush();
}

// tfrt_gpu.tensor.make creates a tensor of type T from a shape and buffer.
template <typename T>
static Expected<DenseGpuTensor> GpuTensorMake(Argument<GpuBuffer> buffer,
                                              TensorShape shape) {
  if (!*buffer) {
    return MakeStringError(
        "Cannot make tensor from invalid (moved from?) buffer");
  }
  if (buffer->size() != shape.GetNumElements() * GetDType<T>().GetHostSize()) {
    return MakeStringError(
        "tfrt_gpu.tensor.make failed: buffer_size (", buffer->size(),
        ") is not equal to the number of elements in shape (", shape,
        ") times element size (", GetDType<T>().GetHostSize(), ")");
  }
  return DenseGpuTensor(shape, GetDType<T>(), std::move(buffer.ValueRef()));
}

// tfrt_gpu.tensor.print_metadata prints `tensor`'s metadata.
static void GpuTensorPrintMetadata(const DenseGpuTensor& tensor) {
  (tfrt::outs() << tensor << "\n").flush();
}

static Error CheckMemcpySizes(size_t dst_size, size_t src_size,
                              int64_t copy_size) {
  if (src_size < copy_size) {
    return MakeStringError("source buffer is smaller (", src_size,
                           ") than number of bytes to copy (", copy_size, ")");
  }
  if (dst_size < copy_size) {
    return MakeStringError("destination buffer is smaller (", dst_size,
                           ") than number of bytes to copy (", copy_size, ")");
  }
  return Error::success();
}

// tfrt_gpu.mem.copy_host_to_device copies memory from host to device.
static Error GpuMemcpyHtoD(const GpuBuffer& dst,
                           const RCReference<HostBuffer>& src,
                           int64_t bytes_count, const GpuStream& stream) {
  if (auto error = CheckMemcpySizes(dst.size(), src->size(), bytes_count))
    return error;
  auto current = wrapper::CtxSetCurrent(stream.context());
  if (!current) return current.takeError();
  return wrapper::MemcpyAsync(
      *current, dst.pointer(),
      wrapper::Pointer<const void>(src->data(), current->platform()),
      bytes_count, stream.get());
}

// tfrt_gpu.mem.copy_host_to_device copies memory from host to device.
static Error GpuMemcpyDtoH(const RCReference<HostBuffer>& dst,
                           const GpuBuffer& src, int64_t bytes_count,
                           const GpuStream& stream) {
  if (auto error = CheckMemcpySizes(dst->size(), src.size(), bytes_count))
    return error;
  auto current = wrapper::CtxSetCurrent(stream.context());
  if (!current) return current.takeError();
  return wrapper::MemcpyAsync(*current,
                              GpuPointer(dst->data(), current->platform()),
                              src.pointer(), bytes_count, stream.get());
}

static Expected<GpuModule> GpuModuleLoad(
    Argument<GpuContext> context,
    // Note: Attributes must be in alphabetical order (see b/140896071).
    StringAttribute data, Attribute<uint64_t> key) {
  auto module = context->LoadModule(key.get(), data.get());
  if (!module) return module.takeError();
  return GpuModule(context.ValueRef(), *module);
}

static Expected<GpuFunction> GpuFunctionGet(const GpuModule& module,
                                            StringAttribute name) {
  auto result = wrapper::ModuleGetFunction(module.get(), name.str().c_str());
  return result;
}

static Error GpuFunctionLaunch(const GpuStream& stream, GpuFunction function,
                               uint32_t grid_dim_x, uint32_t grid_dim_y,
                               uint32_t grid_dim_z, uint32_t block_dim_x,
                               uint32_t block_dim_y, uint32_t block_dim_z,
                               uint32_t shared_memory_size_bytes, Chain,
                               RemainingArguments args) {
  auto current = wrapper::CtxSetCurrent(stream.context());
  if (!current) return current.takeError();

  // Kernel params are a vector of pointers to the kernel args, so we must first
  // materialize the kernel arg values.
  llvm::SmallVector<uintptr_t, 16> arg_values;
  arg_values.reserve(args.size());
  for (const auto& arg : args.values()) {
    if (arg->IsType<GpuBuffer>()) {
      auto pointer = arg->get<GpuBuffer>().pointer();
      arg_values.push_back(reinterpret_cast<uintptr_t>(pointer.raw()));
    } else if (arg->IsType<int32_t>()) {
      arg_values.push_back(arg->get<int32_t>());
    } else {
      return MakeStringError("Unsupported argument type");
    }
  }

  // Add required layer of indirection for kernel params.
  // TODO(idan): Consider using packed params interface.
  llvm::SmallVector<void*, 16> arg_pointers;
  arg_pointers.reserve(args.size());
  for (auto& arg_value : arg_values) arg_pointers.push_back(&arg_value);

  return wrapper::LaunchKernel(*current, function, grid_dim_x, grid_dim_y,
                               grid_dim_z, block_dim_x, block_dim_y,
                               block_dim_z, shared_memory_size_bytes,
                               stream.get(), arg_pointers, {});
}

void RegisterGpuDriverKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_gpu.device.get", TFRT_KERNEL(GpuDeviceGet));
  kernel_reg->AddKernel("tfrt_gpu.context.create",
                        TFRT_KERNEL(GpuContextCreate));

  kernel_reg->AddKernel("tfrt_gpu.stream.create", TFRT_KERNEL(GpuStreamCreate));
  kernel_reg->AddKernel("tfrt_gpu.stream.get_context",
                        TFRT_KERNEL(GpuStreamGetContext));
  kernel_reg->AddKernel("tfrt_gpu.stream.wait", TFRT_KERNEL(GpuStreamWait));
  kernel_reg->AddKernel("tfrt_gpu.stream.synchronize",
                        TFRT_KERNEL(GpuStreamSynchronize));

  kernel_reg->AddKernel("tfrt_gpu.event.create", TFRT_KERNEL(GpuEventCreate));
  kernel_reg->AddKernel("tfrt_gpu.event.record",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(GpuEventRecord));
  kernel_reg->AddKernel("tfrt_gpu.event.synchronize",
                        TFRT_KERNEL(GpuEventSynchronize));

  kernel_reg->AddKernel("tfrt_gpu.allocator.create",
                        TFRT_KERNEL(GpuAllocatorCreate));

  kernel_reg->AddKernel("tfrt_gpu.mem.allocate", TFRT_KERNEL(GpuMemAllocate));
  kernel_reg->AddKernel("tfrt_gpu.mem.print_metadata",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(GpuMemPrintMetadata));

  kernel_reg->AddKernel("tfrt_gpu.tensor.make.i8",
                        TFRT_KERNEL(GpuTensorMake<int8_t>));
  kernel_reg->AddKernel("tfrt_gpu.tensor.make.i32",
                        TFRT_KERNEL(GpuTensorMake<int32_t>));
  kernel_reg->AddKernel("tfrt_gpu.tensor.make.i64",
                        TFRT_KERNEL(GpuTensorMake<int64_t>));
  kernel_reg->AddKernel("tfrt_gpu.tensor.make.f32",
                        TFRT_KERNEL(GpuTensorMake<float>));
  kernel_reg->AddKernel("tfrt_gpu.tensor.make.f64",
                        TFRT_KERNEL(GpuTensorMake<double>));

  kernel_reg->AddKernel("tfrt_gpu.tensor.print_metadata",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(GpuTensorPrintMetadata));
  kernel_reg->AddKernel("tfrt_gpu.mem.copy_host_to_device",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(GpuMemcpyHtoD));
  kernel_reg->AddKernel("tfrt_gpu.mem.copy_device_to_host",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(GpuMemcpyDtoH));

  kernel_reg->AddKernel("tfrt_gpu.module.load", TFRT_KERNEL(GpuModuleLoad));
  kernel_reg->AddKernel("tfrt_gpu.function.get", TFRT_KERNEL(GpuFunctionGet));
  kernel_reg->AddKernel("tfrt_gpu.function.launch",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(GpuFunctionLaunch));
}

}  // namespace gpu
}  // namespace tfrt
