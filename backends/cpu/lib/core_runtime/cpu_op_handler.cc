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

//===- cpu_op_handler.cc --------------------------------------------------===//
//
// This file implements the CpuOpHandler.
//
//===----------------------------------------------------------------------===//

#include "tfrt/cpu/core_runtime/cpu_op_handler.h"

#include "cpu_op_registry_impl.h"
#include "llvm/Support/Compiler.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/dispatch_utils.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/device.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/tensor/coo_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/host_tensor.h"
#include "tfrt/tensor/scalar_host_tensor.h"
#include "tfrt/tensor/string_host_tensor.h"
#include "tfrt/tensor/tensor_type_registration.h"

#define DEBUG_TYPE "tfrt-cpu-op-op_handler"

namespace tfrt {
class CpuOpHandler : public OpHandler {
 public:
  ~CpuOpHandler() override {}

  Expected<CoreRuntimeOp> MakeOp(string_view op_name) override;

  // For CpuOpHandler, the argument `tensor` needs to be a HostTensor. This
  // function returns a HostTensor (DenseHostTensor or StringHostTensor), which
  // contains a copy of the underlying data.
  AsyncValueRef<HostTensor> CopyDeviceTensorToHost(
      const ExecutionContext& exec_ctx, const Tensor& tensor) override;

  // This function returns a DenseHostTensor that contains a copy of the
  // underlying buffer of the argument `tensor`.
  AsyncValueRef<Tensor> CopyHostTensorToDevice(
      const DenseHostTensor& tensor) override;

  RCReference<Device> GetDeviceRef() { return device_.CopyRef(); }

 private:
  const CpuOpRegistry op_registry_;
  RCReference<Device> device_;

  friend llvm::Expected<OpHandler*> CreateCpuOpHandler(
      CoreRuntime* runtime, RCReference<Device> device, OpHandler* fallback);

  // TODO(b/157120084): Remove after op_handler DSL is deprecated.
  friend llvm::Expected<std::unique_ptr<OpHandler>> CpuOpHandlerFactory(
      CoreRuntime* runtime, OpHandler* fallback);

  explicit CpuOpHandler(CoreRuntime* runtime, OpHandler* fallback,
                        CpuOpRegistry op_registry, RCReference<Device> device)
      : OpHandler("cpu", runtime, fallback),
        op_registry_(std::move(op_registry)),
        device_(std::move(device)) {}
};

namespace {

// If the specified tensor needs conversion to be compatible with CpuOpEntry,
// then return the allowed format mask.
uint32_t TensorNeedsConversion(const Tensor& t, const CpuOpEntry& entry) {
  // DenseHostTensor is always supported.
  // TODO(chenyin): remove allowed_formats. ConvertToHostTensor should only need
  // tensor type.
  uint32_t allowed_formats =
      1 << static_cast<uint32_t>(Tensor::Subclass::DenseHost);
  uint32_t allowed_types =
      uint32_t{1} << static_cast<uint32_t>(DenseHostTensor::kTensorType.id());

  if (entry.flags & CpuOpFlags::AllowsScalar) {
    allowed_formats |= 1 << static_cast<uint32_t>(Tensor::Subclass::ScalarHost);
    allowed_types |= uint32_t{1} << static_cast<uint32_t>(
                         AnyScalarHostTensor::kTensorType.id());
  }

  if (entry.flags & CpuOpFlags::AllowsString) {
    allowed_formats |= 1 << static_cast<uint32_t>(Tensor::Subclass::StringHost);
    allowed_types |= uint32_t{1} << static_cast<uint32_t>(
                         StringHostTensor::kTensorType.id());
  }

  if (entry.flags & CpuOpFlags::AllowsCoo) {
    allowed_formats |= 1 << static_cast<uint32_t>(Tensor::Subclass::CooHost);
    allowed_types |= uint32_t{1}
                     << static_cast<uint32_t>(CooHostTensor::kTensorType.id());
  }

  // Note: TFLite tensors are deprecated and this path will be removed.
  if (entry.flags & CpuOpFlags::AllowsTfLite) {
    allowed_formats |= 1 << static_cast<uint32_t>(Tensor::Subclass::TFLiteHost);
    allowed_types |= uint32_t{1} << static_cast<uint32_t>(
                         GetStaticTensorType("TFLiteHost").id());
    allowed_types |= uint32_t{1} << static_cast<uint32_t>(
                         GetStaticTensorType("TFLiteStringHost").id());
  }

  // If the tensor is already in a supported format, then we're done.
  if (allowed_types & uint32_t{1}
                          << static_cast<uint32_t>(t.tensor_type().id()))
    return 0;

  // Otherwise return the mask of supported formats so a conversion can be
  // performed.
  return allowed_formats;
}

struct CpuOpHandlerTraits {
  using InputTensorTy = AsyncValue;
  using OpEntryTy = CpuOpEntry;
  using OpHandlerInfoTy = CpuOpHandler*;

  static bool MaybeConvertTensor(const CpuOpEntry& op_entry,
                                 CpuOpHandler* cpu_op_handler,
                                 const Tensor& arg_tensor,
                                 const ExecutionContext& exec_ctx,
                                 RCReference<AsyncValue>* converted) {
    if (auto allowed_formats = TensorNeedsConversion(arg_tensor, op_entry)) {
      *converted =
          arg_tensor.ConvertToHostTensor(exec_ctx.host(), allowed_formats);
      return true;
    }
    return false;
  }

  static void Dispatch(const CpuOpEntry& op_entry, CpuOpHandler* cpu_op_handler,
                       ArrayRef<AsyncValue*> inputs, const OpAttrsRef& attrs,
                       ArrayRef<TensorMetadata> result_mds,
                       MutableArrayRef<RCReference<AsyncValue>> results,
                       AsyncValueRef<Chain>* chain,
                       const ExecutionContext& exec_ctx) {
    op_entry.dispatch_fn(exec_ctx, inputs, attrs, result_mds, results, chain);
  }

  static Expected<RCReference<Device>> GetResultDevice(
      CpuOpHandler* cpu_op_handler, AsyncValueRef<Tensor> result_tensor_av_ref,
      const ExecutionContext& exec_ctx) {
    return cpu_op_handler->GetDeviceRef();
  }
};

}  // namespace

// TODO(b/157120084): Remove after op_handler DSL is deprecated.
llvm::Expected<std::unique_ptr<OpHandler>> CpuOpHandlerFactory(
    CoreRuntime* runtime, OpHandler* fallback) {
  CpuOpRegistry op_registry;
  RegisterStaticCpuOps(&op_registry);
  return std::unique_ptr<OpHandler>(new CpuOpHandler(
      runtime, fallback, std::move(op_registry),
      runtime->GetHostContext()->GetDeviceManager()->GetDeviceRef<CpuDevice>(
          "CPU:0")));
}

llvm::Expected<OpHandler*> CreateCpuOpHandler(CoreRuntime* runtime,
                                              RCReference<Device> device,
                                              OpHandler* fallback) {
  if (!runtime) {
    return MakeStringError("Invalid Runtime");
  }
  CpuOpRegistry op_registry;
  RegisterStaticCpuOps(&op_registry);
  auto cpu_op_handler = std::unique_ptr<CpuOpHandler>(new CpuOpHandler(
      runtime, fallback, std::move(op_registry), std::move(device)));
  auto cpu_op_handler_ptr = cpu_op_handler.get();
  runtime->TakeOpHandler(std::move(cpu_op_handler));
  return cpu_op_handler_ptr;
}

AsyncValueRef<HostTensor> CpuOpHandler::CopyDeviceTensorToHost(
    const ExecutionContext& exec_ctx, const Tensor& tensor) {
  if (tensor.IsHostTensor()) {
    // If tensor is a host tensor, we call Tensor::ConvertToHostTensor
    // to make a copy of the tensor here, because the source and result buffers
    // are logically independent.
    auto& dst_tensor_type = tensor.IsTensorType(StringHostTensor::kTensorType)
                                ? StringHostTensor::kTensorType
                                : DenseHostTensor::kTensorType;
    return ConvertTensorOnHost(tensor, dst_tensor_type,
                               GetRuntime()->GetHostContext());
  }

  // Otherwise, this copy is meant for the fallback device.
  return GetFallback()->CopyDeviceTensorToHost(exec_ctx, tensor);
}

AsyncValueRef<Tensor> CpuOpHandler::CopyHostTensorToDevice(
    const DenseHostTensor& tensor) {
  // We call ConvertTensorOnHost to make a copy of the tensor here,
  // because the source and result buffers are logically independent.
  return ConvertTensorOnHost(tensor, DenseHostTensor::kTensorType,
                             GetRuntime()->GetHostContext());
}

//===----------------------------------------------------------------------===//
// Op Dispatch Implementation
//===----------------------------------------------------------------------===//

Expected<CoreRuntimeOp> CpuOpHandler::MakeOp(string_view op_name) {
  auto* op_entry = op_registry_.impl_->LookupOpEntry(op_name);

  // If this operation is unknown by cpu OpHandler, then we try to run it on
  // fallback OpHandler.
  if (op_entry->dispatch_fn == nullptr) return GetFallback()->MakeOp(op_name);

  // NOTE(fishx): To avoid introducing an extra heap allocation, we need to
  // ensure that the size of captured variable is smaller than 3 pointers.
  return CoreRuntimeOp(
      [op_entry, this](const OpInvocation& invocation) {
        // CPU OpHandler should associate a CPU device.
        assert(this->device_);
        bool update_chain = !(op_entry->flags & CpuOpFlags::NoSideEffects);

        // TODO(fishx): ExecuteOnOpHandler should return void.
        ExecuteOnOpHandler<CpuOpHandlerTraits>(update_chain, invocation,
                                               *op_entry, this);
      },
      /*is_fallback=*/false, /*device=*/device_.CopyRef(),
      /*arg_tensor_type=*/DenseHostTensor::kTensorType);
}

}  // namespace tfrt
