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
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include "llvm/Support/Error.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/kernels/kernels_detail.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value.h"
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
  return stream.context().CopyRef();
}

// tfrt_gpu.stream.wait makes a stream wait on an event.
static Error GpuStreamWait(const GpuStream& stream, const GpuEvent& event) {
  return wrapper::StreamWaitEvent(stream.get(), event.get());
}

// tfrt_gpu.stream.synchronize sets the output chain ready when all work
// (including callbacks) previously enqueued on the stream is completed.
static AsyncValueRef<Chain> GpuStreamSynchronize(
    Argument<GpuStream> stream, const ExecutionContext& exec_ctx) {
  return EnqueueBlockingWork(
      exec_ctx.host(),
      DestroyCapturesOnInvoke(
          [stream = stream.ValueRef()]() -> Expected<Chain> {
            if (auto error = wrapper::StreamSynchronize(stream->get()))
              return std::move(error);
            auto none_pending = stream->context()->MaybeInvokeCallbacks();
            if (auto error = none_pending.takeError()) return std::move(error);
            return Chain();
          }));
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
// reached, i.e. all work (including callbacks) scheduled prior to the last call
// to tfrt_gpu.event.record has been completed.
static AsyncValueRef<Chain> GpuEventSynchronize(
    Argument<GpuEvent> event, const ExecutionContext& exec_ctx) {
  // Check if event has already completed and we can skip enqueuing work.
  auto ready = wrapper::EventQuery(event->get());
  if (!ready) return MakeErrorAsyncValueRef(StrCat(ready.takeError()));
  if (*ready) {
    if (auto error = event->context()->MaybeInvokeCallbacks().takeError())
      return MakeErrorAsyncValueRef(StrCat(std::move(error)));
    return GetReadyChain();
  }
  return EnqueueBlockingWork(
      exec_ctx.host(),
      DestroyCapturesOnInvoke([event = event.ValueRef()]() -> Expected<Chain> {
        if (auto error = wrapper::EventSynchronize(event->get()))
          return std::move(error);
        if (auto error = event->context()->MaybeInvokeCallbacks().takeError())
          return std::move(error);
        return Chain();
      }));
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

// tfrt_gpu.mem.set fills memory with a 32bit scalar value.
static Error GpuMemset(const GpuBuffer& buffer, AsyncValue* untyped_value,
                       const GpuStream& stream) {
  if (!(untyped_value->IsType<uint32_t>() || untyped_value->IsType<int32_t>() ||
        untyped_value->IsType<float>())) {
    return MakeStringError("Expected 32 bit value.");
  }

  auto current = wrapper::CtxSetCurrent(stream.context()->get());
  if (!current) return current.takeError();

  union {
    uint32_t u;
    int32_t i;
    float f;
  } typed_value;

  if (untyped_value->IsType<uint32_t>())
    typed_value.u = untyped_value->get<uint32_t>();
  else if (untyped_value->IsType<int32_t>())
    typed_value.i = untyped_value->get<int32_t>();
  else if (untyped_value->IsType<float>())
    typed_value.f = untyped_value->get<float>();
  size_t count = buffer.size() / sizeof(uint32_t);

  return wrapper::MemsetD32Async(*current, buffer.pointer(), typed_value.u,
                                 count, stream.get());
}

// tfrt_gpu.mem.copy copies memory between host or device.
static Error GpuMemCopy(RemainingArguments args,
                        const ExecutionContext& exec_ctx) {
  if (args.size() != 4)
    return MakeStringError("Expected 4 arguments, got ", args.size());

  const GpuStream& stream = args[2]->get<GpuStream>();
  auto current = wrapper::CtxSetCurrent(stream.context()->get());
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

  auto dst_ptr = get_ptr(args[0]);
  auto src_ptr = get_ptr(args[1]);
  if (dst_ptr == src_ptr) {
    // Source and destination are the same; skip copy.
    return Error::success();
  }

  if (auto error = wrapper::MemcpyAsync(*current, dst_ptr, src_ptr, dst_size,
                                        stream.get()))
    return error;

  // Hold on to ref-counts of src and dst until the async memcpy completes.
  return GpuContext::AddEventualCallback(
      *current, stream, [dst = FormRef(args[0]), src = FormRef(args[1])] {},
      exec_ctx.host());
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
      pointer,
      std::make_tuple(*buffer, context.ValueRef(), std::move(*memory)));
  return GpuBuffer::Allocate(std::move(allocator), size);
}

static Expected<GpuBuffer> GpuMemView(Argument<GpuBuffer> buffer,
                                      uint64_t offset, uint64_t size) {
  if (buffer->size() < offset + size) {
    return MakeStringError("buffer size (", buffer->size(),
                           ") is smaller than offset (", offset,
                           ") plus size (", size, ")");
  }
  // The allocator releases the buffer reference on destruction.
  using Allocator = GpuOneShotAllocator<AsyncValueRef<GpuBuffer>>;
  auto allocator = MakeAvailableAsyncValueRef<Allocator>(
      wrapper::Pointer<char>(buffer->pointer()) + offset, buffer.ValueRef());
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
static Expected<GpuModule> GpuModuleLoad(Argument<GpuContext> context,
                                         StringAttribute data) {
  string_view blob = data.get();

  if (blob.empty() || blob.back() != 0)
    return MakeStringError("data attribute must be null-terminated");

  auto current = wrapper::CtxSetCurrent(context->get());
  if (!current) return current.takeError();

#ifdef NDEBUG
  auto module = wrapper::ModuleLoadData(*current, blob.data());
  if (!module) return module.takeError();
#else
  std::string info_log;
  std::string error_log;

  wrapper::ModuleLoadOptions options{&info_log, &error_log, 1};
  auto module = wrapper::ModuleLoadDataEx(*current, blob.data(), options);
  if (!info_log.empty()) TFRT_LOG_INFO << "GPU JIT info log: " << info_log;

  if (!module) {
    return llvm::joinErrors(module.takeError(),
                            MakeStringError("GPU JIT error log: ", error_log));
  }
#endif

  return GpuModule(context.ValueRef(), std::move(*module));
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

static Expected<GpuFunction> GpuModuleGetFunction(Argument<GpuModule> module,
                                                  StringAttribute name) {
  auto function = wrapper::ModuleGetFunction(module->get(), name.str().c_str());
  if (!function) return function.takeError();
  return GpuFunction(module.ValueRef(), function.get());
}

static Error GpuFunctionLaunch(const GpuStream& stream,
                               const GpuFunction& function, uint32_t grid_dim_x,
                               uint32_t grid_dim_y, uint32_t grid_dim_z,
                               uint32_t block_dim_x, uint32_t block_dim_y,
                               uint32_t block_dim_z,
                               uint32_t shared_memory_size_bytes, Chain,
                               RemainingArguments args) {
  auto current = wrapper::CtxSetCurrent(stream.context()->get());
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

  return wrapper::LaunchKernel(*current, function.get(), grid_dim_x, grid_dim_y,
                               grid_dim_z, block_dim_x, block_dim_y,
                               block_dim_z, shared_memory_size_bytes,
                               stream.get(), arg_pointers, {});
}

static void GpuAlias(AsyncValue* value, Chain, RemainingResults results) {
  results.values().front() = FormRef(value);
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
  kernel_reg->AddKernel("tfrt_gpu.mem.set",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(GpuMemset));
  kernel_reg->AddKernel("tfrt_gpu.mem.register", TFRT_KERNEL(GpuMemRegister));
  kernel_reg->AddKernel("tfrt_gpu.mem.view", TFRT_KERNEL(GpuMemView));
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
  kernel_reg->AddKernel("tfrt_gpu.module.get_function",
                        TFRT_KERNEL(GpuModuleGetFunction));
  kernel_reg->AddKernel("tfrt_gpu.function.launch",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(GpuFunctionLaunch));

  kernel_reg->AddKernel("tfrt_gpu.alias", TFRT_KERNEL(GpuAlias));
}

}  // namespace gpu
}  // namespace tfrt
