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

//===- gpu_op_handler.cc --------------------------------------------------===//
//
// This file implements the GpuOpHandler.
//
//===----------------------------------------------------------------------===//

#include "tfrt/gpu/core_runtime/gpu_op_handler.h"

#include "eigen_support.h"
#include "gpu_op_registry_impl.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/dispatch_utils.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/gpu/core_runtime/gpu_config.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/tensor_util.h"
#include "tfrt/gpu/device/device_util.h"
#include "tfrt/gpu/memory/bfc_gpu_allocator.h"
#include "tfrt/gpu/stream/blas_wrapper.h"
#include "tfrt/gpu/stream/cublas_wrapper.h"
#include "tfrt/gpu/stream/dnn_wrapper.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/device.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/host_tensor.h"

#define DEBUG_TYPE "tfrt-gpu-op-op_handler"

namespace tfrt {
class AsyncValue;
class Chain;
class GpuOpRegistry;
class Tensor;
using gpu::stream::CtxSetCurrent;
using gpu::stream::CurrentContext;
using gpu::stream::DeviceGet;
using gpu::stream::OwningEvent;
using gpu::stream::Platform;
using gpu::stream::StreamFlags;

class GpuOpHandler : public OpHandler {
 public:
  explicit GpuOpHandler(CoreRuntime* runtime, int gpu_ordinal,
                        OpHandler* fallback, GpuOpRegistry op_registry,
                        RCReference<Device> device);

  Expected<CoreRuntimeOp> MakeOp(string_view op_name) override;

  GpuDispatchContext MakeGpuDispatchContext();

  AsyncValueRef<HostTensor> CopyDeviceTensorToHost(
      const ExecutionContext& exec_ctx, const Tensor& tensor) override;

  AsyncValueRef<Tensor> CopyHostTensorToDevice(
      const DenseHostTensor& tensor) override;

 private:
  llvm::Error Initialize();

  int gpu_ordinal_;
  // TODO(sanjoy): we need to figure out how the lifetimes of these objects
  // interact with the lifetime of the GPU op handler.

  gpu::stream::Device device_;

  // NB! The declaration order here is important.  We want to destroy context_
  // last.
  // If `owned_context_` is null, `context_` points to a non-owning context.
  // Otherwise, `context_` is set to be `owned_context_.get()`.
  gpu::stream::OwningContext owned_context_;
  gpu::stream::Context context_;
  // If `owned_stream` is null, `stream_` points to a non-owning stream.
  // Otherwise, `stream_` is set to be `owned_stream_.get()`.
  gpu::stream::OwningStream owned_stream_;
  gpu::stream::Stream stream_;
  gpu::stream::OwningBlasHandle blas_handle_;
  gpu::stream::OwningDnnHandle dnn_handle_;

  // NB! The declaration order here is important. The eigen_gpu_device_
  // references eigen_stream_interface_, which references stream_.
  gpu::OwningEigenStreamInterface eigen_stream_interface_;
  gpu::OwningEigenGpuDevice eigen_gpu_device_;

  std::unique_ptr<gpu::GpuAllocator> allocator_;
  const GpuOpRegistry op_registry_;

  RCReference<Device> tfrt_device_;

  friend llvm::Expected<OpHandler*> CreateGpuOpHandler(CoreRuntime* runtime,
                                                       int gpu_ordinal,
                                                       OpHandler* fallback);

  // TODO(b/157120084): Remove after op_handler DSL is deprecated.
  friend llvm::Expected<std::unique_ptr<OpHandler>> GPUOpHandlerFactory(
      CoreRuntime* runtime, OpHandler* fallback);
};

namespace {
struct GpuOpHandlerTraits {
  using InputTensorTy = AsyncValue;
  using OpEntryTy = GpuOpEntry;
  using OpHandlerInfoTy = GpuOpHandler*;

  static bool MaybeConvertTensor(const GpuOpEntry& op_entry,
                                 GpuOpHandler* gpu_op_handler,
                                 const Tensor& arg_tensor,
                                 const ExecutionContext& exec_ctx,
                                 RCReference<AsyncValue>* converted) {
    return false;
  }

  static void Dispatch(const GpuOpEntry& op_entry, GpuOpHandler* gpu_op_handler,
                       ArrayRef<AsyncValue*> inputs, const OpAttrsRef& attrs,
                       ArrayRef<TensorMetadata> result_mds,
                       MutableArrayRef<RCReference<AsyncValue>> results,
                       AsyncValueRef<Chain>* chain,
                       const ExecutionContext& exec_ctx) {
    GpuDispatchContext dctx = gpu_op_handler->MakeGpuDispatchContext();
    op_entry.dispatch_fn(exec_ctx, &dctx, inputs, attrs, result_mds, results,
                         chain);
  }
};
}  // namespace

llvm::Expected<std::unique_ptr<OpHandler>> GPUOpHandlerFactory(
    CoreRuntime* runtime, OpHandler* fallback) {
  if (llvm::Error result = gpu::stream::Init(gpu::stream::Platform::CUDA))
    return std::move(result);

  GpuOpRegistry op_registry;
  tfrt::RegisterStaticGpuOps(&op_registry);
  // TODO(xldrx): Add multi gpu support.
  auto device = gpu::CreateGpuDevice(0, runtime->GetHostContext());
  auto op_handler = std::make_unique<GpuOpHandler>(
      runtime, 0, fallback, std::move(op_registry), std::move(device));

  if (auto error = op_handler->Initialize()) {
    return std::move(error);
  }

  return op_handler;
}

llvm::Expected<OpHandler*> CreateGpuOpHandler(CoreRuntime* runtime,
                                              int gpu_ordinal,
                                              OpHandler* fallback) {
  if (llvm::Error result = gpu::stream::Init(gpu::stream::Platform::CUDA))
    return std::move(result);

  GpuOpRegistry op_registry;
  tfrt::RegisterStaticGpuOps(&op_registry);
  auto device = gpu::CreateGpuDevice(gpu_ordinal, runtime->GetHostContext());
  auto gpu_op_handler =
      std::make_unique<GpuOpHandler>(runtime, gpu_ordinal, fallback,
                                     std::move(op_registry), std::move(device));

  if (auto error = gpu_op_handler->Initialize()) {
    return std::move(error);
  }

  auto gpu_op_handler_ptr = gpu_op_handler.get();
  runtime->TakeOpHandler(std::move(gpu_op_handler));
  return gpu_op_handler_ptr;

  return gpu_op_handler_ptr;
}

GpuOpHandler::GpuOpHandler(CoreRuntime* runtime, int gpu_ordinal,
                           OpHandler* fallback, GpuOpRegistry op_registry,
                           RCReference<Device> device)
    : OpHandler("gpu", runtime, fallback),
      gpu_ordinal_(gpu_ordinal),
      op_registry_(std::move(op_registry)),
      tfrt_device_(std::move(device)) {}

llvm::Error GpuOpHandler::Initialize() {
  // TODO(zhangqiaorjc): Generalize to multi-GPU.
  TFRT_ASSIGN_OR_RETURN(device_, DeviceGet(Platform::CUDA, gpu_ordinal_));
  llvm::Optional<CurrentContext> current_context;

  // Use external GPU resources if they are available.
  if (auto gpu_resources = gpu::GetTfrtGpuResources(device_)) {
    // Set a non-owning context.
    context_ = gpu_resources->gpu_context;
    TFRT_ASSIGN_OR_RETURN(auto current, CtxSetCurrent(context_));
    current_context.emplace(current);

    allocator_ = std::unique_ptr<gpu::GpuAllocator>(
        gpu_resources->allocator_factory(context_));

    stream_ = gpu_resources->stream;
  } else {
    TFRT_ASSIGN_OR_RETURN(owned_context_, DevicePrimaryCtxRetain(device_));
    context_ = owned_context_.get();
    TFRT_ASSIGN_OR_RETURN(auto current, CtxSetCurrent(context_));
    current_context.emplace(current);

    TFRT_ASSIGN_OR_RETURN(owned_stream_,
                          StreamCreate(current, StreamFlags::DEFAULT));
    stream_ = owned_stream_.get();

    allocator_ =
        std::unique_ptr<gpu::GpuAllocator>(new gpu::BfcGpuAllocator(current));
  }

  eigen_stream_interface_ = gpu::CreateEigenStreamInterface(stream_);
  eigen_gpu_device_ = gpu::CreateEigenGpuDevice(eigen_stream_interface_.get());

  // TODO(iga): Only log errors during BLAS handle creation?
  TFRT_ASSIGN_OR_RETURN(blas_handle_, BlasCreate(*current_context));
  if (auto error = gpu::stream::BlasSetStream(blas_handle_.get(), stream_))
    return error;
  if (auto error = gpu::stream::CublasSetMathMode(
          static_cast<cublasHandle_t>(blas_handle_.get()),
          CUBLAS_TENSOR_OP_MATH))
    return error;

  TFRT_ASSIGN_OR_RETURN(dnn_handle_, gpu::stream::DnnCreate(*current_context));
  if (auto error = gpu::stream::DnnSetStream(dnn_handle_.get(), stream_))
    return error;

  return Error::success();
}

GpuDispatchContext GpuOpHandler::MakeGpuDispatchContext() {
  // FIXME(sanjoy): Add proper error handling.
  llvm::ExitOnError die_if_error;

  CurrentContext current_context = die_if_error(CtxSetCurrent(context_));
  return GpuDispatchContext{stream_,
                            allocator_.get(),
                            eigen_gpu_device_.get(),
                            blas_handle_.get(),
                            dnn_handle_.get(),
                            current_context};
}

AsyncValueRef<HostTensor> GpuOpHandler::CopyDeviceTensorToHost(
    const ExecutionContext& exec_ctx, const Tensor& tensor) {
  auto* host = GetRuntime()->GetHostContext();
  if (auto* gpu_tensor = dyn_cast<gpu::DenseGpuTensor>(&tensor)) {
    GpuDispatchContext dctx = MakeGpuDispatchContext();
    return gpu::CopyDenseGpuTensorToHost(&dctx, *gpu_tensor, host);
  } else {
    return GetFallback()->CopyDeviceTensorToHost(exec_ctx, tensor);
  }
}

AsyncValueRef<Tensor> GpuOpHandler::CopyHostTensorToDevice(
    const DenseHostTensor& tensor) {
  GpuDispatchContext dctx = MakeGpuDispatchContext();
  auto* host = GetRuntime()->GetHostContext();
  Expected<gpu::DenseGpuTensor> gpu_tensor =
      gpu::CopyDenseHostTensorToGpu(&dctx, tensor, host);

  if (gpu_tensor) {
    return host->MakeAvailableAsyncValueRef<gpu::DenseGpuTensor>(
        std::move(*gpu_tensor));
  } else {
    return host->MakeErrorAsyncValueRef(StrCat(gpu_tensor.takeError()));
  }
}
Expected<CoreRuntimeOp> GpuOpHandler::MakeOp(string_view op_name) {
  auto* op_entry = op_registry_.impl_->LookupOpEntry(op_name);
  // If this operation is unknown by gpu device, then we try to run it on
  // fallback device.
  if (op_entry->dispatch_fn == nullptr) return GetFallback()->MakeOp(op_name);
  // TODO(b/149044322): Add side-effect flag in op registry.

  return CoreRuntimeOp(
      [op_entry, this](const OpInvocation& invocation) {
        return ExecuteOnOpHandler<GpuOpHandlerTraits>(
            /*update_chain=*/false, invocation, this->tfrt_device_.CopyRef(),
            *op_entry, this);
      },
      /*is_fallback=*/false);
}

}  // namespace tfrt
