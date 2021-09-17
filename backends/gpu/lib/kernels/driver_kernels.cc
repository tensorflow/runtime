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

#include "llvm/Support/Error.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/kernels/kernels_detail.h"
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

// tfrt_gpu.context.primary returns the primary gpu context for the given
// device.
static Expected<GpuContext> GpuContextPrimary(wrapper::Device device) {
  auto context = wrapper::DevicePrimaryCtxRetain(device);
  if (!context) return context.takeError();
  return GpuContext(std::move(*context));
}

// tfrt_gpu.context.create creates a gpu context for the given device.
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

// tfrt_gpu.stream.synchronize sets the output chain ready when all work
// previously enqueued on the stream are completed.
static AsyncValueRef<Chain> GpuStreamSynchronize(
    Argument<GpuStream> stream, const ExecutionContext& exec_ctx) {
  return EnqueueBlockingWork(
      exec_ctx, [stream = stream.ValueRef()]() -> Expected<Chain> {
        if (auto error = wrapper::StreamSynchronize(stream->get()))
          return std::move(error);
        return Chain();
      });
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

// tfrt_gpu.mem.copy copies memory between host or device.
static Error GpuMemCopy(RemainingArguments args) {
  if (args.size() != 4)
    return MakeStringError("Expected 4 arguments, got ", args.size());

  const GpuStream& stream = args[2]->get<GpuStream>();
  auto current = wrapper::CtxSetCurrent(stream.context());
  if (!current) return current.takeError();

  auto get_size = [](AsyncValue* arg) {
    if (arg->IsType<GpuBuffer>()) return arg->get<GpuBuffer>().size();
    assert(arg->IsType<RCReference<HostBuffer>>());
    return arg->get<RCReference<HostBuffer>>()->size();
  };

  size_t dst_size = get_size(args[0]);
  size_t src_size = get_size(args[1]);

  if (dst_size != src_size)
    return MakeStringError("Buffer sizes don't match: ", dst_size,
                           " (dst size) vs ", src_size, " (src size)");

  auto get_ptr = [&](AsyncValue* arg) {
    if (arg->IsType<GpuBuffer>()) return arg->get<GpuBuffer>().pointer();
    assert(arg->IsType<RCReference<HostBuffer>>());
    auto* ptr = arg->get<RCReference<HostBuffer>>()->data();
    return GpuPointer(ptr, current->platform());
  };

  return wrapper::MemcpyAsync(*current, get_ptr(args[0]), get_ptr(args[1]),
                              dst_size, stream.get());
}

static Expected<GpuBuffer> GpuMemRegister(
    Argument<GpuContext> context, Argument<RCReference<HostBuffer>> buffer) {
  auto current = wrapper::CtxSetCurrent(context->get());
  if (!current) return current.takeError();

  auto size = (*buffer)->size();
  auto flags = wrapper::MemHostRegisterFlags::PORTABLE |
               wrapper::MemHostRegisterFlags::DEVICEMAP;
  auto memory =
      wrapper::MemHostRegister(*current, (*buffer)->data(), size, flags);
  if (!memory) return memory.takeError();
  auto pointer = memory->get();

  // The allocator unregisters the pointer on destruction and then releases
  // the references to the context and the host buffer.
  using Allocator = GpuOneShotAllocator<
      std::tuple<RCReference<HostBuffer>, AsyncValueRef<GpuContext>,
                 wrapper::RegisteredMemory<void>>>;
  auto allocator = MakeAvailableAsyncValueRef<Allocator>(
      pointer, std::make_tuple(buffer->CopyRef(), context.ValueRef(),
                               std::move(*memory)));
  return GpuBuffer::Allocate(std::move(allocator), size);
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
  if (buffer->size() != shape.GetNumElements() * GetHostSize(GetDType<T>())) {
    return MakeStringError(
        "tfrt_gpu.tensor.make failed: buffer_size (", buffer->size(),
        ") is not equal to the number of elements in shape (", shape,
        ") times element size (", GetHostSize(GetDType<T>()), ")");
  }
  return DenseGpuTensor(shape, GetDType<T>(), std::move(buffer.ValueRef()));
}

// tfrt_gpu.tensor.print_metadata prints `tensor`'s metadata.
static void GpuTensorPrintMetadata(const DenseGpuTensor& tensor) {
  (tfrt::outs() << tensor << "\n").flush();
}

// Loads a GPU module from `data`, or `exec_ctx` if `data` is empty.
// `key` is used to uniquely identify the modules within `context`.
static Expected<GpuModule> GpuModuleLoad(
    Argument<GpuContext> context,
    // Note: Attributes must be in alphabetical order (see b/140896071).
    StringAttribute data, Attribute<uint64_t> key,
    const ExecutionContext& exec_ctx) {
  string_view data_str = data.get();
  if (data_str.empty()) {
    const GpuModuleMap* gpu_module_map =
        exec_ctx.request_ctx()->GetDataIfExists<GpuModuleMap>();
    if (gpu_module_map == nullptr) {
      return MakeStringError(
          "No GpuModuleMap resource found in the request context.");
    }
    TFRT_ASSIGN_OR_RETURN(data_str, gpu_module_map->GetModule(key.get()));
  }
  auto module = context->LoadModule(key.get(), data_str);
  if (!module) return module.takeError();
  return GpuModule(context.ValueRef(), *module);
}

static Expected<GpuBuffer> GpuModuleGetGlobal(Argument<GpuModule> module,
                                              StringAttribute name) {
  auto global = wrapper::ModuleGetGlobal(module->get(), name.str().c_str());
  if (!global) return global.takeError();
  using Allocator = GpuOneShotAllocator<AsyncValueRef<GpuModule>>;
  auto allocator =
      MakeAvailableAsyncValueRef<Allocator>(global->base, module.ValueRef());
  return GpuBuffer::Allocate(std::move(allocator), global->size_bytes);
}

static Expected<GpuFunction> GpuModuleFunction(const GpuModule& module,
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
  union KernelArg {
    void* ptr;
    int32_t i;
    float f;
    double d;
  };
  llvm::SmallVector<KernelArg, 16> arg_values;
  arg_values.reserve(args.size());
  for (const auto& arg : args.values()) {
    KernelArg kernel_arg;
    if (arg->IsType<GpuBuffer>()) {
      auto pointer = arg->get<GpuBuffer>().pointer();
      kernel_arg.ptr = pointer.raw();
    } else if (arg->IsType<int32_t>()) {
      kernel_arg.i = arg->get<int32_t>();
    } else if (arg->IsType<float>()) {
      kernel_arg.f = arg->get<float>();
    } else if (arg->IsType<double>()) {
      kernel_arg.d = arg->get<double>();
    } else {
      return MakeStringError("Unsupported argument type");
    }
    arg_values.push_back(kernel_arg);
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
  kernel_reg->AddKernel("tfrt_gpu.context.primary",
                        TFRT_KERNEL(GpuContextPrimary));
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
  kernel_reg->AddKernel("tfrt_gpu.mem.copy",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(GpuMemCopy));
  kernel_reg->AddKernel("tfrt_gpu.mem.register", TFRT_KERNEL(GpuMemRegister));
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

  kernel_reg->AddKernel("tfrt_gpu.module.load", TFRT_KERNEL(GpuModuleLoad));
  kernel_reg->AddKernel("tfrt_gpu.module.get_global",
                        TFRT_KERNEL(GpuModuleGetGlobal));
  kernel_reg->AddKernel("tfrt_gpu.module.function",
                        TFRT_KERNEL(GpuModuleFunction));
  kernel_reg->AddKernel("tfrt_gpu.function.launch",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(GpuFunctionLaunch));
}

}  // namespace gpu
}  // namespace tfrt
