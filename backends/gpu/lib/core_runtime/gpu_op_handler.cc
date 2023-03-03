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

// This file implements the GpuOpHandler.

#include "tfrt/gpu/core_runtime/gpu_op_handler.h"

#include "gpu_op_registry_impl.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/dispatch_utils.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/device/device.h"
#include "tfrt/gpu/device/device_util.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/device.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/conversion_registry.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/host_tensor.h"

#define DEBUG_TYPE "tfrt-gpu-op-op_handler"

namespace tfrt {
class Tensor;

namespace gpu {
class GpuOpRegistry;

class GpuOpHandler : public OpHandler {
 public:
  explicit GpuOpHandler(CoreRuntime* runtime, OpHandler* fallback,
                        GpuOpRegistry op_registry,
                        RCReference<GpuDevice> device);

  Expected<CoreRuntimeOp> MakeOp(string_view op_name) override;

  Expected<GpuDispatchContext> MakeGpuDispatchContext();

  RCReference<Device> GetDeviceRef() { return device_; }

 private:
  const GpuOpRegistry op_registry_;

  RCReference<GpuDevice> device_;

  friend llvm::Expected<OpHandler*> CreateGpuOpHandler(
      CoreRuntime* runtime, RCReference<Device> device, OpHandler* fallback);
};

namespace {
struct GpuOpHandlerTraits {
  using InputTensorTy = AsyncValue;
  using OpEntryTy = GpuOpEntry;
  using OpHandlerInfoTy = GpuOpHandler*;

  static void Dispatch(const GpuOpEntry& op_entry, GpuOpHandler* gpu_op_handler,
                       ArrayRef<AsyncValue*> inputs, const OpAttrsRef& attrs,
                       ArrayRef<TensorMetadata> result_mds,
                       MutableArrayRef<RCReference<AsyncValue>> results,
                       AsyncValueRef<Chain>* chain,
                       const ExecutionContext& exec_ctx) {
    llvm::Expected<GpuDispatchContext> dctx =
        gpu_op_handler->MakeGpuDispatchContext();
    if (!dctx) {
      if (chain && *chain)
        chain->SetError(absl::InternalError(toString(dctx.takeError())));
      return;
    }

    op_entry.dispatch_fn(exec_ctx, &dctx.get(), inputs, attrs, result_mds,
                         results, chain);
  }

  // TODO(fishx): Remove this method.
  static Variant<RCReference<Device>, AsyncValueRef<RCReference<Device>>>
  GetResultDevice(GpuOpHandler* gpu_op_handler,
                  const AsyncValueRef<Tensor>& result_tensor_av,
                  const ExecutionContext& exec_ctx) {
    return gpu_op_handler->GetDeviceRef();
  }

  // TODO(b/168609399): design a proper way to obtain device for result tensors.
  static Variant<RCReference<Device>, AsyncValueRef<RCReference<Device>>>
  GetResultDevice(const GpuOpEntry& op_entry, GpuOpHandler* gpu_op_handler,
                  const AsyncValueRef<Tensor>& result_tensor_av, int index,
                  const ExecutionContext& exec_ctx) {
    return gpu_op_handler->GetDeviceRef();
  }
};
}  // namespace

llvm::Expected<OpHandler*> CreateGpuOpHandler(CoreRuntime* runtime,
                                              RCReference<GpuDevice> device,
                                              OpHandler* fallback) {
  GpuOpRegistry op_registry;
  RegisterStaticGpuOps(&op_registry);
  auto gpu_op_handler = std::make_unique<GpuOpHandler>(
      runtime, fallback, std::move(op_registry), std::move(device));

  auto gpu_op_handler_ptr = gpu_op_handler.get();
  runtime->TakeOpHandler(std::move(gpu_op_handler));
  return gpu_op_handler_ptr;
}

GpuOpHandler::GpuOpHandler(CoreRuntime* runtime, OpHandler* fallback,
                           GpuOpRegistry op_registry,
                           RCReference<GpuDevice> device)
    : OpHandler("gpu", runtime, fallback),
      op_registry_(std::move(op_registry)),
      device_(std::move(device)) {}

Expected<GpuDispatchContext> GpuOpHandler::MakeGpuDispatchContext() {
  return GpuDispatchContext::Create(device_.get());
}

Expected<CoreRuntimeOp> GpuOpHandler::MakeOp(string_view op_name) {
  auto* op_entry = op_registry_.impl_->LookupOpEntry(op_name);
  // If this operation is unknown by gpu OpHandler, then we try to run it on
  // fallback OpHandler.
  if (op_entry->dispatch_fn == nullptr) return GetFallback()->MakeOp(op_name);
  // TODO(b/149044322): Add side-effect flag in op registry.

  return CoreRuntimeOp(
      [op_entry, this](const OpInvocation& invocation) {
        // GPU OpHandler should associate a GPU device.
        assert(this->device_);

        return (void)ExecuteOnOpHandler<GpuOpHandlerTraits>(
            /*update_chain=*/false, invocation, *op_entry, this);
      },
      /*is_fallback=*/false, /*device=*/device_,
      /*arg_tensor_type=*/GetStaticTensorType("DenseGpu"));
}

}  // namespace gpu
}  // namespace tfrt
