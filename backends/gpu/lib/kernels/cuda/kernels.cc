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

//===- kernels.cc - CUDA runtime interface --------------------------------===//
//
// This file defines the C++ functions that implement the kernels provided by
// the TFRT CUDA runtime.

#include <tuple>

#include "llvm/Support/Error.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/gpu/event_manager.h"
#include "tfrt/gpu/memory/caching_gpu_allocator.h"
#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/module_table.h"
#include "tfrt/gpu/stream/cuda_wrapper.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace cuda {

constexpr string_view kModuleTableResource = "cuda_module_table_resource";

// tfrt_cuda.init initializes CUDA driver.
static Error CudaInitSync() {
  return gpu::stream::Init(gpu::stream::Platform::CUDA);
}
static Expected<Chain> CudaInitAsync(Chain chain) {
  if (auto error = CudaInitSync()) return std::move(error);
  return chain;
}

// tfrt_cuda.device.get returns the CUDA Device at the given index.
static Expected<gpu::stream::Device> CudaDeviceGet(int32_t ordinal) {
  return gpu::stream::DeviceGet(gpu::stream::Platform::CUDA, ordinal);
}

// tfrt_cuda.stream.create creates a new stream that does not implicitly
// synchronize with stream 0.
static Expected<gpu::stream::OwningStream> CudaStreamCreate(
    gpu::stream::Context context) {
  auto current = gpu::stream::CtxSetCurrent(context);
  if (!current) return current.takeError();
  return gpu::stream::StreamCreate(*current,
                                   gpu::stream::StreamFlags::NON_BLOCKING);
}

// tfrt_cuda.stream.synchronize waits until all stream's tasks are completed.
//
// Result: Sets the output chain when all tasks submitted on a stream are
// completed. This kernel will block the caller thread.
static Error CudaStreamSynchronizeSync(
    const gpu::stream::OwningStream& stream) {
  return gpu::stream::StreamSynchronize(stream.get());
}
static void CudaStreamSynchronizeAsync(
    Argument<gpu::stream::OwningStream> stream, Chain in_chain,
    Result<Chain> out_chain, const ExecutionContext& exec_ctx) {
  auto result = out_chain.Allocate();
  bool enqueued = EnqueueBlockingWork(
      exec_ctx, [result = result.CopyRef(), stream = stream.ValueRef(),
                 in_chain = in_chain]() mutable {
        if (auto error = CudaStreamSynchronizeSync(*stream))
          return result.SetError(StrCat(error));
        result.emplace(in_chain);
      });
  if (!enqueued) return result.SetError("Failed to enqueue blocking work.");
}

// tfrt_cuda.event.create creates a new cuda event.
//
// Result: new cuda event.
static Expected<gpu::stream::OwningEvent> CudaEventCreate(
    gpu::stream::Context context) {
  auto current = gpu::stream::CtxSetCurrent(context);
  if (!current) return current.takeError();
  return gpu::stream::EventCreate(*current,
                                  gpu::stream::EventFlags::DISABLE_TIMING);
}

// tfrt_cuda.event.create creates a new cuda event.
//
// Result: new cuda event.
static Error CudaEventRecordSync(const gpu::stream::OwningEvent& event,
                                 const gpu::stream::OwningStream& stream) {
  return gpu::stream::EventRecord(event.get(), stream.get());
}
static Expected<Chain> CudaEventRecordAsync(
    const gpu::stream::OwningEvent& event,
    const gpu::stream::OwningStream& stream, Chain chain) {
  if (auto error = CudaEventRecordSync(event, stream)) return std::move(error);
  return chain;
}

// tfrt_cuda.event.synchronize sets the output chain when the event has been
// reached, i.e. all work scheduled prior to the last call to
// tfrt_cuda.event.record has been completed.
static Error CudaEventSynchronizeSync(const gpu::stream::OwningEvent& event) {
  return gpu::stream::EventSynchronize(event.get());
}
static void CudaEventSynchronizeAsync(Argument<gpu::stream::OwningEvent> event,
                                      Chain in_chain, Result<Chain> out_chain,
                                      const ExecutionContext& exec_ctx) {
  auto result = out_chain.Allocate();
  // Check if event has already completed and we can skip enqueuing work.
  auto ready = gpu::stream::EventQuery(event->get());
  if (!ready) return result.SetError(StrCat(ready.takeError()));
  if (*ready) return result.emplace(in_chain);
  bool enqueued = EnqueueBlockingWork(
      exec_ctx, [result = result.CopyRef(), event = event.ValueRef(),
                 in_chain = in_chain]() mutable {
        if (auto error = CudaEventSynchronizeSync(*event))
          return result.SetError(StrCat(error));
        result.emplace(in_chain);
      });
  if (!enqueued) return result.SetError("Failed to enqueue blocking work.");
}

// tfrt_cuda.allocator.create creates a new allocator.
//
// Result: new allocator.
static std::unique_ptr<gpu::GpuAllocator> CudaAllocatorCreate(
    gpu::stream::Context) {
  return std::make_unique<gpu::CachingGpuAllocator>();
}

// tfrt_cuda.allocator.destroy destroys an allocator.
static void CudaAllocatorDestroySync(
    Argument<std::unique_ptr<gpu::GpuAllocator>> allocator) {
  allocator->reset();
}
static Chain CudaAllocatorDestroyAsync(
    Argument<std::unique_ptr<gpu::GpuAllocator>> allocator, Chain chain) {
  CudaAllocatorDestroySync(std::move(allocator));
  return chain;
}

// tfrt_cuda.mem.allocate allocates a new CUDA buffer.
static Expected<RCReference<gpu::GpuBuffer>> CudaMemAllocateSync(
    const std::unique_ptr<gpu::GpuAllocator>& allocator,
    const gpu::stream::OwningStream& stream, int64_t size) {
  return allocator->Allocate(size, stream.get());
}
static Expected<std::tuple<RCReference<gpu::GpuBuffer>, Chain>>
CudaMemAllocateAsync(const std::unique_ptr<gpu::GpuAllocator>& allocator,
                     const gpu::stream::OwningStream& stream, int64_t size,
                     Chain chain) {
  auto alloc = CudaMemAllocateSync(allocator, stream, size);
  if (!alloc) return alloc.takeError();
  return std::make_tuple(std::move(*alloc), chain);
}

// tfrt_cuda.mem.print_metadata prints `buffer`'s metadata.
static void CudaMemPrintMetadataSync(
    const RCReference<gpu::GpuBuffer>& buffer) {
  // The check for buffer validity is not done intentionally. Printing invalid
  // buffers can be useful for debugging.
  (tfrt::outs() << *buffer << "\n").flush();
}
static Chain CudaMemPrintMetadataAsync(
    const RCReference<gpu::GpuBuffer>& buffer, Chain chain) {
  CudaMemPrintMetadataSync(buffer);
  return chain;
}

// tfrt_cuda.tensor.make makes a tensor from the given shape and buffer.
// It is specialized for each supported DType.
template <typename T>
static Expected<gpu::DenseGpuTensor> CudaTensorMakeSync(
    RCReference<gpu::GpuBuffer> buffer, TensorShape shape) {
  if (!buffer->IsValid()) {
    return MakeStringError(
        "Cannot make tensor from invalid (moved from?) buffer");
  }
  if (buffer->size() != shape.GetNumElements() * GetDType<T>().GetHostSize()) {
    return MakeStringError(
        "tfrt_cuda.tensor.make failed: buffer_size (", buffer->size(),
        ") is not equal to the number of elements in shape (", shape,
        ") times element size (", GetDType<T>().GetHostSize(), ")");
  }
  return gpu::DenseGpuTensor(shape, GetDType<T>(), std::move(buffer));
}
template <typename T>
static Expected<std::tuple<gpu::DenseGpuTensor, Chain>> CudaTensorMakeAsync(
    Argument<RCReference<gpu::GpuBuffer>> buffer, TensorShape shape,
    Chain chain) {
  auto tensor = CudaTensorMakeSync<T>(std::move(*buffer), shape);
  if (!tensor) return tensor.takeError();
  return std::make_tuple(std::move(*tensor), chain);
}

// tfrt_cuda.tensor.print_metadata prints `tensor`'s metadata.
static void CudaTensorPrintMetadataSync(const gpu::DenseGpuTensor& tensor) {
  (tfrt::outs() << tensor << "\n").flush();
}
static Chain CudaTensorPrintMetadataAsync(const gpu::DenseGpuTensor& tensor,
                                          Chain chain) {
  CudaTensorPrintMetadataSync(tensor);
  return chain;
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

// tfrt_cuda.mem.copy_host_to_device copies memory from host to device.
static Error CudaMemcpyHtoDSync(gpu::stream::Context context,
                                const RCReference<gpu::GpuBuffer>& dst,
                                const RCReference<HostBuffer>& src,
                                int64_t bytes_count,
                                const gpu::stream::OwningStream& stream) {
  if (auto error = CheckMemcpySizes(dst->size(), src->size(), bytes_count))
    return error;
  auto current = gpu::stream::CtxSetCurrent(context);
  if (!current) return current.takeError();
  return gpu::stream::MemcpyAsync(
      *current, dst->pointer(),
      gpu::stream::Pointer<const void>(src->data(), context.platform()),
      bytes_count, stream.get());
}
static Expected<Chain> CudaMemcpyHtoDAsync(
    gpu::stream::Context context, const RCReference<gpu::GpuBuffer>& dst,
    const RCReference<HostBuffer>& src, int64_t bytes_count,
    const gpu::stream::OwningStream& stream, Chain chain) {
  if (auto error = CudaMemcpyHtoDSync(context, dst, src, bytes_count, stream))
    return std::move(error);
  return chain;
}

// tfrt_cuda.mem.copy_host_to_device copies memory from host to device.
static Error CudaMemcpyDtoHSync(gpu::stream::Context context,
                                const RCReference<HostBuffer>& dst,
                                const RCReference<gpu::GpuBuffer>& src,
                                int64_t bytes_count,
                                const gpu::stream::OwningStream& stream) {
  if (auto error = CheckMemcpySizes(dst->size(), src->size(), bytes_count))
    return error;
  auto current = gpu::stream::CtxSetCurrent(context);
  if (!current) return current.takeError();
  return gpu::stream::MemcpyAsync(
      *current, gpu::stream::Pointer<void>(dst->data(), context.platform()),
      src->pointer(), bytes_count, stream.get());
}
static Expected<Chain> CudaMemcpyDtoHAsync(
    gpu::stream::Context context, const RCReference<HostBuffer>& dst,
    const RCReference<gpu::GpuBuffer>& src, int64_t bytes_count,
    const gpu::stream::OwningStream& stream, Chain chain) {
  if (auto error = CudaMemcpyDtoHSync(context, dst, src, bytes_count, stream))
    return std::move(error);
  return chain;
}

static Error CudaModuleLoadStaticSync(gpu::stream::Context context,
                                      // Note: Attributes must be in
                                      // alphabetical order (see b/140896071).
                                      ArrayAttribute<int32_t> funcs_per_module,
                                      AggregateAttr functions,
                                      AggregateAttr modules,
                                      const ExecutionContext& exec_ctx) {
  auto current = gpu::stream::CtxSetCurrent(context);
  if (!current) return current.takeError();

  auto device = gpu::stream::CtxGetDevice(*current);
  if (!device) return device.takeError();

  auto multi_device_module_table =
      exec_ctx.resource_context()
          ->GetOrCreateResource<std::unique_ptr<gpu::MultiDeviceModuleTable>>(
              kModuleTableResource, gpu::MultiDeviceModuleTable::Create());
  if (multi_device_module_table->get()->GetTable(*device).hasValue()) {
    return MakeStringError(
        "Unable to load module table. Table has already been created for "
        "device ",
        *device);
  }
  auto parsed_spec =
      gpu::ParseModuleTableSpec(modules, funcs_per_module, functions);
  if (!parsed_spec) return parsed_spec.takeError();

  auto module_table = gpu::ModuleTable::Create(*current, *parsed_spec);
  if (!module_table) return module_table.takeError();

  return multi_device_module_table->get()->AddTable(*device,
                                                    std::move(*module_table));
}
static Expected<Chain> CudaModuleLoadStaticAsync(
    gpu::stream::Context context, Chain chain,
    // Note: Attributes must be in
    // alphabetical order (see b/140896071).
    ArrayAttribute<int32_t> funcs_per_module, AggregateAttr functions,
    AggregateAttr modules, const ExecutionContext& exec_ctx) {
  if (auto error = CudaModuleLoadStaticSync(context, funcs_per_module,
                                            functions, modules, exec_ctx))
    return std::move(error);
  return chain;
}

static Error CudaLaunchSync(gpu::stream::Context context, uint32_t grid_dim_x,
                            uint32_t grid_dim_y, uint32_t grid_dim_z,
                            uint32_t block_dim_x, uint32_t block_dim_y,
                            uint32_t block_dim_z,
                            uint32_t shared_memory_size_bytes,
                            const gpu::stream::OwningStream& stream,
                            RemainingArguments args,
                            Attribute<gpu::ModuleFuncHandle> function_handle,
                            const ExecutionContext& exec_ctx) {
  auto current = gpu::stream::CtxSetCurrent(context);
  if (!current) return current.takeError();

  auto device = gpu::stream::CtxGetDevice(*current);
  if (!device) return device.takeError();

  auto multi_device_module_table_or =
      exec_ctx.resource_context()
          ->GetResource<std::unique_ptr<gpu::MultiDeviceModuleTable>>(
              kModuleTableResource);
  if (!multi_device_module_table_or.hasValue()) {
    return MakeStringError(
        "CUDA module table has not been initialized for any device");
  }
  const gpu::MultiDeviceModuleTable& multi_device_module_table =
      *(*multi_device_module_table_or.getValue());
  const auto module_table = multi_device_module_table.GetTable(*device);
  if (!module_table) {
    return MakeStringError(
        "CUDA module table has not been initialized for device ", *device);
  }

  // Kernel params are a vector of pointers to the kernel args, so we must first
  // materialize the kernel arg values.
  llvm::SmallVector<uintptr_t, 16> arg_values;
  arg_values.reserve(args.size());
  for (const auto& arg : args.values()) {
    if (arg->IsType<RCReference<gpu::GpuBuffer>>()) {
      auto pointer = arg->get<RCReference<gpu::GpuBuffer>>()->pointer();
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

  gpu::stream::Function func_ptr =
      (*module_table)->GetFunction(function_handle.get());
  return gpu::stream::LaunchKernel(
      *current, func_ptr, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
      block_dim_y, block_dim_z, shared_memory_size_bytes, stream.get(),
      arg_pointers, llvm::ArrayRef<void*>{});
}
static Expected<Chain> CudaLaunchAsync(
    gpu::stream::Context context, uint32_t grid_dim_x, uint32_t grid_dim_y,
    uint32_t grid_dim_z, uint32_t block_dim_x, uint32_t block_dim_y,
    uint32_t block_dim_z, uint32_t shared_memory_size_bytes,
    const gpu::stream::OwningStream& stream, Chain chain,
    RemainingArguments args, Attribute<gpu::ModuleFuncHandle> function_handle,
    const ExecutionContext& exec_ctx) {
  if (auto error = CudaLaunchSync(context, grid_dim_x, grid_dim_y, grid_dim_z,
                                  block_dim_x, block_dim_y, block_dim_z,
                                  shared_memory_size_bytes, stream, args,
                                  function_handle, exec_ctx))
    return std::move(error);
  return chain;
}

void RegisterCudaKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_cuda.init", TFRT_KERNEL(CudaInitAsync));
  kernel_reg->AddKernel("tfrt_cuda.device.get", TFRT_KERNEL(CudaDeviceGet));
  kernel_reg->AddKernel("tfrt_cuda.stream.create",
                        TFRT_KERNEL(CudaStreamCreate));
  kernel_reg->AddKernel("tfrt_cuda.stream.synchronize",
                        TFRT_KERNEL(CudaStreamSynchronizeAsync));

  kernel_reg->AddKernel("tfrt_cuda.event.create", TFRT_KERNEL(CudaEventCreate));
  kernel_reg->AddKernel("tfrt_cuda.event.record",
                        TFRT_KERNEL(CudaEventRecordAsync));
  kernel_reg->AddKernel("tfrt_cuda.event.synchronize",
                        TFRT_KERNEL(CudaEventSynchronizeAsync));

  kernel_reg->AddKernel("tfrt_cuda.allocator.create",
                        TFRT_KERNEL(CudaAllocatorCreate));
  kernel_reg->AddKernel("tfrt_cuda.allocator.destroy",
                        TFRT_KERNEL(CudaAllocatorDestroyAsync));

  kernel_reg->AddKernel("tfrt_cuda.mem.allocate",
                        TFRT_KERNEL(CudaMemAllocateAsync));
  kernel_reg->AddKernel("tfrt_cuda.mem.print_metadata",
                        TFRT_KERNEL(CudaMemPrintMetadataAsync));

  kernel_reg->AddKernel("tfrt_cuda.tensor.make.i8",
                        TFRT_KERNEL(CudaTensorMakeAsync<int8_t>));
  kernel_reg->AddKernel("tfrt_cuda.tensor.make.i32",
                        TFRT_KERNEL(CudaTensorMakeAsync<int32_t>));
  kernel_reg->AddKernel("tfrt_cuda.tensor.make.i64",
                        TFRT_KERNEL(CudaTensorMakeAsync<int64_t>));
  kernel_reg->AddKernel("tfrt_cuda.tensor.make.f32",
                        TFRT_KERNEL(CudaTensorMakeAsync<float>));
  kernel_reg->AddKernel("tfrt_cuda.tensor.make.f64",
                        TFRT_KERNEL(CudaTensorMakeAsync<double>));

  kernel_reg->AddKernel("tfrt_cuda.tensor.print_metadata",
                        TFRT_KERNEL(CudaTensorPrintMetadataAsync));
  kernel_reg->AddKernel("tfrt_cuda.mem.copy_host_to_device",
                        TFRT_KERNEL(CudaMemcpyHtoDAsync));
  kernel_reg->AddKernel("tfrt_cuda.mem.copy_device_to_host",
                        TFRT_KERNEL(CudaMemcpyDtoHAsync));

  kernel_reg->AddKernel("tfrt_cuda.module.load_static",
                        TFRT_KERNEL(CudaModuleLoadStaticAsync));
  kernel_reg->AddKernel("tfrt_cuda.launch", TFRT_KERNEL(CudaLaunchAsync));
}

}  // namespace cuda
}  // namespace tfrt
