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

// This file implements the CpuOpHandler.

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
#include "tfrt/support/logging.h"
#include "tfrt/tensor/coo_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/host_tensor.h"
#include "tfrt/tensor/scalar_host_tensor.h"
#include "tfrt/tensor/string_host_tensor.h"
#include "tfrt/tensor/tensor_type_registration.h"

#define DEBUG_TYPE "tfrt-cpu-op-op_handler"

namespace tfrt {

namespace {

// If the specified tensor needs conversion to be compatible with CpuOpEntry,
// then return the target tensor type. Otherwise, return the original tensor
// type.
TensorType ArgumentTensorType(const Tensor& t, const CpuOpFlags& flags) {
  // DenseHostTensor is always supported.
  auto result = DenseHostTensor::kTensorType;

  if (flags & CpuOpFlags::AllowsScalar) {
    auto type = AnyScalarHostTensor::kTensorType;
    if (t.IsTensorType(type)) return type;
  }

  if (flags & CpuOpFlags::AllowsCoo) {
    auto type = CooHostTensor::kTensorType;
    if (t.IsTensorType(type)) return type;
  }

  if (flags & CpuOpFlags::AllowsString) {
    auto type = StringHostTensor::kTensorType;
    if (t.IsTensorType(type)) return type;
    if (t.dtype() == DType::String && result == DenseHostTensor::kTensorType)
      result = type;
  }

  // Note: TFLite tensors are deprecated and this path will be removed.
  if (flags & CpuOpFlags::AllowsTfLite) {
    auto type = t.dtype() == DType::String
                    ? GetStaticTensorType("TFLiteStringHost")
                    : GetStaticTensorType("TFLiteHost");
    if (t.IsTensorType(type)) return type;
    if (result == DenseHostTensor::kTensorType) result = type;
  }

  if (flags & CpuOpFlags::AllowsTfRuntimeFallback) {
    auto type = GetStaticTensorType("RuntimeFallback");
    if (t.IsTensorType(type)) return type;
    if (result == DenseHostTensor::kTensorType) result = type;
  }

  return result;
}

TensorHandle MaybeConvertArgument(const ExecutionContext& exec_ctx,
                                  const CpuOpFlags& flags,
                                  CpuOpHandler* op_handler, TensorHandle arg) {
  if (arg.IsError()) return arg;
  RCReference<Device> device = op_handler->GetDeviceRef();

  if (arg.GetAsyncTensor()->IsAvailable() && arg.IsDeviceAvailable()) {
    // We does not support implicit tensor conversion across device.
    if (device.get() != arg.GetAvailableDevice().get()) {
      // TODO(b/172847467): Return error tensor here instead of a slient
      // warning. This cannot be done currently as it will break existing GPU
      // tests.
      TFRT_LOG(WARNING) << "Cannot implictly convert from device "
                        << arg.GetAvailableDevice()->name() << " to "
                        << device->name();
      return arg;
    }
    auto& tensor = arg.GetAsyncTensor()->get<Tensor>();
    auto target_type = ArgumentTensorType(tensor, flags);
    if (target_type == tensor.tensor_type()) return arg;
    if (!op_handler->AllowImplicitConversion(tensor.tensor_type(),
                                             target_type)) {
      return TensorHandle(EmitErrorAsync(
          exec_ctx,
          tfrt::StrCat("Cannot implictly convert ", tensor.tensor_type().name(),
                       " to ", target_type.name())));
    }
    return arg.TransferTo(exec_ctx, op_handler->GetDeviceRef(), target_type);
  } else {
    RCReference<IndirectAsyncValue> result_ind_av =
        MakeIndirectAsyncValue(exec_ctx.host());
    SmallVector<AsyncValue*, 2> async_values;
    async_values.push_back(arg.GetAsyncTensor());
    if (!arg.IsDeviceAvailable()) {
      async_values.push_back(arg.GetAsyncDevice().GetAsyncValue());
    }
    RunWhenReady(async_values, [exec_ctx, arg = arg.CopyRef(), result_ind_av,
                                device, op_handler, flags]() mutable {
      if (arg.IsDeviceError()) {
        result_ind_av->ForwardTo(arg.GetAsyncDevice().CopyRCRef());
        return;
      }
      // We does not support implicit tensor conversion across
      // device.
      if (device.get() != arg.GetAvailableDevice().get()) {
        // TODO(b/172847467): Return error tensor here instead of a
        // slient warning. This cannot be done currently as it will
        // break existing GPU tests.
        TFRT_LOG(WARNING) << "Cannot implict convert from device "
                          << arg.GetAvailableDevice()->name() << " to "
                          << device->name();
        result_ind_av->ForwardTo(FormRef(arg.GetAsyncTensor()));
        return;
      }
      if (arg.GetAsyncTensor()->IsError()) {
        result_ind_av->ForwardTo(FormRef(arg.GetAsyncTensor()));
        return;
      }
      auto& arg_tensor = arg.GetAsyncTensor()->get<Tensor>();
      auto target_type = ArgumentTensorType(arg_tensor, flags);
      if (target_type == arg_tensor.tensor_type()) {
        result_ind_av->ForwardTo(FormRef(arg.GetAsyncTensor()));
        return;
      }
      if (!op_handler->AllowImplicitConversion(arg_tensor.tensor_type(),
                                               target_type)) {
        result_ind_av->ForwardTo(EmitErrorAsync(
            exec_ctx, tfrt::StrCat("Cannot implictly convert ",
                                   arg_tensor.tensor_type().name(), " to ",
                                   target_type.name())));
        return;
      }
      result_ind_av->ForwardTo(
          ConvertTensor(exec_ctx, arg_tensor, *device, *device, target_type));
    });

    if (arg.IsMetadataAvailable()) {
      return TensorHandle(std::move(device), arg.GetAvailableMetadata(),
                          AsyncValueRef<Tensor>(std::move(result_ind_av)));
    } else {
      return TensorHandle(std::move(device), arg.GetAsyncMetadata().CopyRef(),
                          AsyncValueRef<Tensor>(std::move(result_ind_av)));
    }
  }
}

struct CpuOpHandlerTraits {
  using InputTensorTy = AsyncValue;
  using OpEntryTy = CpuOpEntry;
  using OpHandlerInfoTy = CpuOpHandler*;

  static void Dispatch(const CpuOpEntry& op_entry, CpuOpHandler* cpu_op_handler,
                       ArrayRef<AsyncValue*> inputs, const OpAttrsRef& attrs,
                       ArrayRef<TensorMetadata> result_mds,
                       MutableArrayRef<RCReference<AsyncValue>> results,
                       AsyncValueRef<Chain>* chain,
                       const ExecutionContext& exec_ctx) {
    op_entry.dispatch_fn(exec_ctx, inputs, attrs, result_mds, results, chain);
  }

  // TODO(fishx): Remove this method.
  static Variant<RCReference<Device>, AsyncValueRef<RCReference<Device>>>
  GetResultDevice(CpuOpHandler* cpu_op_handler,
                  const AsyncValueRef<Tensor>& result_tensor_av,
                  const ExecutionContext& exec_ctx) {
    return cpu_op_handler->GetDeviceRef();
  }

  static Variant<RCReference<Device>, AsyncValueRef<RCReference<Device>>>
  GetResultDevice(const CpuOpEntry& op_entry, CpuOpHandler* cpu_op_handler,
                  const AsyncValueRef<Tensor>& result_tensor_av, int index,
                  const ExecutionContext& exec_ctx) {
    return cpu_op_handler->GetDeviceRef();
  }
};

}  // namespace

llvm::Expected<CpuOpHandler*> CreateCpuOpHandler(CoreRuntime* runtime,
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

  cpu_op_handler_ptr->AddImplicitConversion(AnyScalarHostTensor::kTensorType,
                                            DenseHostTensor::kTensorType);
  cpu_op_handler_ptr->AddImplicitConversion(CooHostTensor::kTensorType,
                                            DenseHostTensor::kTensorType);

  return cpu_op_handler_ptr;
}

const char* const CpuOpHandler::kName = "cpu";

CpuOpHandler::CpuOpHandler(CoreRuntime* runtime, OpHandler* fallback,
                           CpuOpRegistry op_registry,
                           RCReference<Device> device)
    : OpHandler(CpuOpHandler::kName, runtime, fallback),
      op_registry_(std::move(op_registry)),
      device_(std::move(device)) {}

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

        // Convert the argument tensors if needed.
        for (auto& argument : invocation.arguments) {
          argument = MaybeConvertArgument(invocation.exec_ctx, op_entry->flags,
                                          this, std::move(argument));
        }

        // TODO(fishx): ExecuteOnOpHandler should return void.
        ExecuteOnOpHandler<CpuOpHandlerTraits>(update_chain, invocation,
                                               *op_entry, this);
      },
      /*is_fallback=*/false, /*device=*/device_,
      /*arg_tensor_type=*/DenseHostTensor::kTensorType);
}

void CpuOpHandler::AddImplicitConversion(TensorType src, TensorType dst) {
  allowed_conversions.insert({src, dst});
}
bool CpuOpHandler::AllowImplicitConversion(TensorType src, TensorType dst) {
  return allowed_conversions.contains({src, dst});
}

}  // namespace tfrt
