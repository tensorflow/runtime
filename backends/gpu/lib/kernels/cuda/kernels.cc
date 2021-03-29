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
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/gpu/event_manager.h"
#include "tfrt/gpu/memory/caching_gpu_allocator.h"
#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/module_table.h"
#include "tfrt/gpu/stream/cuda_wrapper.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace cuda {

constexpr string_view kModuleTableResource = "cuda_module_table_resource";

// Overloaded helpers for the REPORT_ERROR macro. Allows the macro to use either
// strings or llvm::Errors.
static void ReportErrorInternal(KernelErrorHandler error_handler,
                                string_view error_message, string_view file,
                                int line) {
  return error_handler.ReportError(file, ':', line, ' ', error_message);
}

static void ReportErrorInternal(KernelErrorHandler error_handler,
                                llvm::Error error, string_view file, int line) {
  llvm::handleAllErrors(std::move(error), [&](const llvm::ErrorInfoBase& info) {
    ReportErrorInternal(error_handler, info.message(), file, line);
  });
}

// Convert 'error' to string and report to 'error_handler'.
#define REPORT_ERROR(error_handler, error) \
  ReportErrorInternal(error_handler, error, __FILE__, __LINE__)

// tfrt_cuda.init initializes CUDA driver.
static void CudaInit(Argument<Chain> in_chain, Result<Chain> out_chain,
                     KernelErrorHandler handler) {
  llvm::Error result = gpu::stream::Init(gpu::stream::Platform::CUDA);
  if (result) return REPORT_ERROR(handler, std::move(result));
  out_chain.Set(in_chain);
}

// tfrt_cuda.device.get returns the CUDA Device at the given index.
static void CudaDeviceGet(Argument<int32_t> ordinal, Argument<Chain> in_chain,
                          Result<gpu::stream::Device> out_device,
                          Result<Chain> out_chain, KernelErrorHandler handler) {
  auto device = gpu::stream::DeviceGet(gpu::stream::Platform::CUDA, *ordinal);
  if (!device) return REPORT_ERROR(handler, device.takeError());
  out_device.Emplace(device.get());
  out_chain.Set(in_chain);
}

// tfrt_cuda.stream.create creates a new stream that does not implicitly
// synchronize with stream 0.
static void CudaStreamCreate(Argument<gpu::stream::Context> context,
                             Argument<Chain> in_chain,
                             Result<gpu::stream::OwningStream> out_stream,
                             Result<Chain> out_chain,
                             KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  auto stream = gpu::stream::StreamCreate(
      *current, gpu::stream::StreamFlags::NON_BLOCKING);
  if (!stream) return REPORT_ERROR(handler, stream.takeError());
  out_stream.Emplace(std::move(*stream));
  out_chain.Set(in_chain);
}

// tfrt_cuda.stream.synchronize waits until all stream's tasks are completed.
//
// Result: Sets the output chain when all tasks submitted on a stream are
// completed. This kernel will block the caller thread.
static void CudaStreamSynchronize(Argument<gpu::stream::OwningStream> stream,
                                  Argument<Chain> in_chain,
                                  Result<Chain> out_chain,
                                  KernelErrorHandler handler) {
  llvm::Error error = gpu::stream::StreamSynchronize(stream->get());
  if (error) return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

// tfrt_cuda.event.create creates a new cuda event.
//
// Result: new cuda event.
static void CudaEventCreate(Argument<gpu::stream::Context> context,
                            Result<RCReference<gpu::RcEvent>> out_event,
                            KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  auto event = gpu::stream::EventCreate(
      *current, gpu::stream::EventFlags::DISABLE_TIMING);
  if (!event) return REPORT_ERROR(handler, event.takeError());
  out_event.Emplace(TakeRef(new gpu::RcEvent(std::move(*event))));
}

// tfrt_cuda.event.create creates a new cuda event.
//
// Result: new cuda event.
static void CudaEventRecord(Argument<RCReference<gpu::RcEvent>> event,
                            Argument<gpu::stream::OwningStream> stream,
                            Argument<Chain> in_chain, Result<Chain> out_chain,
                            KernelErrorHandler handler) {
  llvm::Error record_error =
      gpu::stream::EventRecord(event->get()->get(), stream->get());
  if (record_error) return REPORT_ERROR(handler, std::move(record_error));
  out_chain.Set(in_chain);
}

// tfrt_cuda.event.poll polls for event being reached.
//
// Result: Sets the output chain when the event has been reached, i.e.
// all work scheduled prior to the last call to tfrt_cuda.event.record has been
// completed
static void CudaEventPoll(Argument<RCReference<gpu::RcEvent>> event,
                          Argument<Chain> in_chain, Result<Chain> out_chain,
                          AsyncKernelFrame* in_frame) {
  // TODO(b/146084342): Implement this with an efficient EventMgr.
  llvm::Error error = gpu::stream::EventSynchronize(event->get()->get());
  if (error)
    return REPORT_ERROR(KernelErrorHandler(in_frame), std::move(error));
  out_chain.Set(in_chain);
}

// tfrt_cuda.allocator.create creates a new allocator.
//
// Result: new allocator.
static void CudaAllocatorCreate(
    Argument<gpu::stream::Context> context, Argument<Chain> in_chain,
    Result<std::unique_ptr<gpu::GpuAllocator>> out_allocator,
    Result<Chain> out_chain) {
  out_allocator.Emplace(std::make_unique<gpu::CachingGpuAllocator>());
  out_chain.Set(in_chain);
}

// tfrt_cuda.allocator.destroy destroys an allocator.
static void CudaAllocatorDestroy(
    Argument<std::unique_ptr<gpu::GpuAllocator>> allocator,
    Argument<Chain> in_chain, Result<Chain> out_chain) {
  allocator->reset();
  out_chain.Set(in_chain);
}

// tfrt_cuda.mem.allocate allocates a new CUDA buffer.
static void CudaMemAllocate(
    Argument<std::unique_ptr<gpu::GpuAllocator>> allocator,
    Argument<gpu::stream::OwningStream> stream, Argument<int64_t> size,
    Argument<Chain> in_chain, Result<RCReference<gpu::GpuBuffer>> out_buffer,
    Result<Chain> out_chain, KernelErrorHandler handler) {
  auto buffer = (*allocator)->Allocate(*size, stream->get());
  if (!buffer) return REPORT_ERROR(handler, buffer.takeError());
  out_buffer.Emplace(std::move(*buffer));
  out_chain.Set(in_chain);
}

// tfrt_cuda.mem.print_metadata prints `buffer`'s metadata.
static void CudaMemPrintMetadata(Argument<RCReference<gpu::GpuBuffer>> buffer,
                                 Argument<Chain> in_chain,
                                 Result<Chain> out_chain) {
  // The check for buffer validity is not done intentionally. Printing invalid
  // buffers can be useful for debugging.
  tfrt::outs() << *buffer.get() << "\n";
  tfrt::outs().flush();
  out_chain.Set(in_chain);
}

// tfrt_cuda.tensor.make makes a tensor from the given shape and buffer.
// It is specialized for each supported DType.
template <typename T>
static void CudaTensorMake(Argument<RCReference<gpu::GpuBuffer>> buffer,
                           Argument<TensorShape> shape,
                           Argument<Chain> in_chain,
                           Result<gpu::DenseGpuTensor> tensor,
                           Result<Chain> out_chain,
                           KernelErrorHandler handler) {
  if (!buffer.get()->IsValid()) {
    REPORT_ERROR(handler,
                 "Cannot make tensor from invalid (moved from?) buffer");
    return;
  }
  if (buffer.get()->size() !=
      shape->GetNumElements() * GetDType<T>().GetHostSize()) {
    std::string error_msg;
    llvm::raw_string_ostream ss(error_msg);
    ss << "tfrt_cuda.tensor.make failed: buffer_size (" << buffer.get()->size()
       << ") is not equal to the number of elements in shape (" << *shape
       << ") times element size (" << GetDType<T>().GetHostSize() << ")";
    REPORT_ERROR(handler, ss.str());
    return;
  }
  tensor.Emplace(*shape, GetDType<T>(), std::move(*buffer));
  out_chain.Set(in_chain);
}

// tfrt_cuda.tensor.print_metadata prints `tensor`'s metadata.
static void CudaTensorPrintMetadata(Argument<gpu::DenseGpuTensor> tensor,
                                    Argument<Chain> in_chain,
                                    Result<Chain> out_chain,
                                    KernelErrorHandler handler) {
  tfrt::outs() << *tensor << "\n";
  tfrt::outs().flush();
  out_chain.Set(in_chain);
}

// tfrt_cuda.mem.copy_host_to_device copies memory from host to device.
static void CudaMemcpyHtoD(Argument<gpu::stream::Context> context,
                           Argument<RCReference<gpu::GpuBuffer>> dst,
                           Argument<RCReference<HostBuffer>> src,
                           Argument<int64_t> bytes_count,
                           Argument<gpu::stream::OwningStream> stream,
                           Argument<Chain> in_chain, Result<Chain> out_chain,
                           KernelErrorHandler handler) {
  if (dst.get()->size() < *bytes_count) {
    REPORT_ERROR(handler,
                 tfrt::StrCat("tfrt_cuda.mem.copy_host_to_device failed: "
                              "destination buffer size (",
                              dst.get()->size(),
                              ") is less than number of bytes to copy (",
                              *bytes_count, ")"));
    return;
  }
  if (src.get()->size() < *bytes_count) {
    REPORT_ERROR(
        handler,
        tfrt::StrCat(
            "tfrt_cuda.mem.copy_host_to_device failed: source buffer size (",
            src.get()->size(), ") is less than number of bytes to copy (",
            *bytes_count, ")"));
    return;
  }
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  llvm::Error error =
      MemcpyAsync(*current, dst.get()->pointer(),
                  gpu::stream::Pointer<const void>(src.get()->data(),
                                                   gpu::stream::Platform::CUDA),
                  *bytes_count, stream->get());
  if (error) return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

// tfrt_cuda.mem.copy_device_to_host copies memory from device to host.
static void CudaMemcpyDtoH(Argument<gpu::stream::Context> context,
                           Argument<RCReference<HostBuffer>> dst,
                           Argument<RCReference<gpu::GpuBuffer>> src,
                           Argument<int64_t> bytes_count,
                           Argument<gpu::stream::OwningStream> stream,
                           Argument<Chain> in_chain, Result<Chain> out_chain,
                           KernelErrorHandler handler) {
  if (dst.get()->size() < *bytes_count) {
    REPORT_ERROR(handler,
                 tfrt::StrCat("tfrt_cuda.mem.copy_device_to_host failed: "
                              "destination buffer size (",
                              dst.get()->size(),
                              ") is less than number of bytes to copy (",
                              *bytes_count, ")"));
    return;
  }
  if (src.get()->size() < *bytes_count) {
    REPORT_ERROR(
        handler,
        tfrt::StrCat(
            "tfrt_cuda.mem.copy_device_to_host failed: source buffer size (",
            src.get()->size(), ") is less than number of bytes to copy (",
            *bytes_count, ")"));
    return;
  }
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());
  llvm::Error error =
      MemcpyAsync(*current,
                  gpu::stream::Pointer<void>(dst.get()->data(),
                                             gpu::stream::Platform::CUDA),

                  src.get()->pointer(), *bytes_count, stream->get());
  if (error) return REPORT_ERROR(handler, std::move(error));
  out_chain.Set(in_chain);
}

static void CudaModuleLoadStatic(Argument<gpu::stream::Context> context,
                                 Argument<Chain> in_chain,
                                 Result<Chain> out_chain,
                                 // Note: Attributes must be in alphabetical
                                 // order (see b/140896071).
                                 ArrayAttribute<int32_t> funcs_per_module,
                                 AggregateAttr functions, AggregateAttr modules,
                                 const ExecutionContext& exec_ctx,
                                 KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());

  auto device = gpu::stream::CtxGetDevice(*current);
  if (!device) return REPORT_ERROR(handler, device.takeError());

  auto multi_device_module_table =
      exec_ctx.resource_context()
          ->GetOrCreateResource<std::unique_ptr<gpu::MultiDeviceModuleTable>>(
              kModuleTableResource, gpu::MultiDeviceModuleTable::Create());
  if (multi_device_module_table->get()->GetTable(*device).hasValue()) {
    return REPORT_ERROR(handler,
                        StrCat("Unable to load CUDA module table. Table has "
                               "already been created for device ",
                               device->id(current->platform())));
  }
  auto parsed_spec =
      gpu::ParseModuleTableSpec(modules, funcs_per_module, functions);
  if (!parsed_spec) return REPORT_ERROR(handler, parsed_spec.takeError());

  auto module_table = gpu::ModuleTable::Create(*current, *parsed_spec);
  if (!module_table) return REPORT_ERROR(handler, module_table.takeError());

  if (auto error = multi_device_module_table->get()->AddTable(
          *device, std::move(*module_table))) {
    return REPORT_ERROR(handler, std::move(error));
  }

  out_chain.Set(in_chain);
}

static void CudaLaunch(
    Argument<Chain> in_chain, Argument<gpu::stream::Context> context,
    Argument<uint32_t> grid_dim_x, Argument<uint32_t> grid_dim_y,
    Argument<uint32_t> grid_dim_z, Argument<uint32_t> block_dim_x,
    Argument<uint32_t> block_dim_y, Argument<uint32_t> block_dim_z,
    Argument<uint32_t> shared_memory_size_bytes,
    Argument<gpu::stream::OwningStream> stream, RemainingArguments args,
    Result<Chain> out_chain, Attribute<gpu::ModuleFuncHandle> function_handle,
    const ExecutionContext& exec_ctx, KernelErrorHandler handler) {
  auto current = gpu::stream::CtxSetCurrent(*context);
  if (!current) return REPORT_ERROR(handler, current.takeError());

  auto device = gpu::stream::CtxGetDevice(*current);
  if (!device) return REPORT_ERROR(handler, device.takeError());

  const int device_ordinal = device->id(current->platform());

  auto multi_device_module_table_or =
      exec_ctx.resource_context()
          ->GetResource<std::unique_ptr<gpu::MultiDeviceModuleTable>>(
              kModuleTableResource);
  if (!multi_device_module_table_or.hasValue()) {
    REPORT_ERROR(handler,
                 "CUDA module table has not been initialized for any device");
    return;
  }
  const gpu::MultiDeviceModuleTable& multi_device_module_table =
      *(*multi_device_module_table_or.getValue());
  const auto module_table = multi_device_module_table.GetTable(*device);
  if (!module_table) {
    REPORT_ERROR(
        handler,
        StrCat("CUDA module table has not been initialized for device ",
               device_ordinal));
    return;
  }

  // Kernel params are a vector of pointers to the kernel args, so we must first
  // materialize the kernel arg values.
  llvm::SmallVector<uintptr_t, 16> arg_values;
  arg_values.reserve(args.size());
  for (const auto& arg : args.values()) {
    if (arg->IsType<RCReference<gpu::GpuBuffer>>()) {
      arg_values.push_back(reinterpret_cast<uintptr_t>(
          arg->get<RCReference<gpu::GpuBuffer>>()->pointer().raw()));
    } else if (arg->IsType<int32_t>()) {
      arg_values.push_back(arg->get<int32_t>());
    } else {
      REPORT_ERROR(handler,
                   "Unsupported argument type provided to tfrt_cuda.launch.");
      return;
    }
  }

  // Add required layer of indirection for kernel params.
  // TODO(idan): Consider using packed params interface.
  llvm::SmallVector<void*, 16> arg_pointers;
  arg_pointers.reserve(args.size());
  for (auto& arg_value : arg_values) {
    arg_pointers.push_back(&arg_value);
  }

  gpu::stream::Function func_ptr =
      (*module_table)->GetFunction(function_handle.get());
  llvm::Error error = gpu::stream::LaunchKernel(
      *current, func_ptr, grid_dim_x.get(), grid_dim_y.get(), grid_dim_z.get(),
      block_dim_x.get(), block_dim_y.get(), block_dim_z.get(),
      shared_memory_size_bytes.get(), stream->get(), arg_pointers,
      llvm::ArrayRef<void*>{});

  if (error) return REPORT_ERROR(handler, std::move(error));

  out_chain.Set(in_chain);
}

void RegisterCudaKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_cuda.init", TFRT_KERNEL(CudaInit));
  kernel_reg->AddKernel("tfrt_cuda.device.get", TFRT_KERNEL(CudaDeviceGet));
  kernel_reg->AddKernel("tfrt_cuda.stream.create",
                        TFRT_KERNEL(CudaStreamCreate));
  kernel_reg->AddKernel("tfrt_cuda.stream.synchronize",
                        TFRT_KERNEL(CudaStreamSynchronize));

  kernel_reg->AddKernel("tfrt_cuda.event.create", TFRT_KERNEL(CudaEventCreate));
  kernel_reg->AddKernel("tfrt_cuda.event.record", TFRT_KERNEL(CudaEventRecord));
  kernel_reg->AddKernel("tfrt_cuda.event.poll", TFRT_KERNEL(CudaEventPoll));

  kernel_reg->AddKernel("tfrt_cuda.allocator.create",
                        TFRT_KERNEL(CudaAllocatorCreate));
  kernel_reg->AddKernel("tfrt_cuda.allocator.destroy",
                        TFRT_KERNEL(CudaAllocatorDestroy));

  kernel_reg->AddKernel("tfrt_cuda.mem.allocate", TFRT_KERNEL(CudaMemAllocate));
  kernel_reg->AddKernel("tfrt_cuda.mem.print_metadata",
                        TFRT_KERNEL(CudaMemPrintMetadata));

  kernel_reg->AddKernel("tfrt_cuda.tensor.make.i8",
                        TFRT_KERNEL(CudaTensorMake<int8_t>));
  kernel_reg->AddKernel("tfrt_cuda.tensor.make.i32",
                        TFRT_KERNEL(CudaTensorMake<int32_t>));
  kernel_reg->AddKernel("tfrt_cuda.tensor.make.i64",
                        TFRT_KERNEL(CudaTensorMake<int64_t>));
  kernel_reg->AddKernel("tfrt_cuda.tensor.make.f32",
                        TFRT_KERNEL(CudaTensorMake<float>));
  kernel_reg->AddKernel("tfrt_cuda.tensor.make.f64",
                        TFRT_KERNEL(CudaTensorMake<double>));

  kernel_reg->AddKernel("tfrt_cuda.tensor.print_metadata",
                        TFRT_KERNEL(CudaTensorPrintMetadata));
  kernel_reg->AddKernel("tfrt_cuda.mem.copy_host_to_device",
                        TFRT_KERNEL(CudaMemcpyHtoD));
  kernel_reg->AddKernel("tfrt_cuda.mem.copy_device_to_host",
                        TFRT_KERNEL(CudaMemcpyDtoH));

  kernel_reg->AddKernel("tfrt_cuda.module.load_static",
                        TFRT_KERNEL(CudaModuleLoadStatic));
  kernel_reg->AddKernel("tfrt_cuda.launch", TFRT_KERNEL(CudaLaunch));
}

}  // namespace cuda
}  // namespace tfrt
