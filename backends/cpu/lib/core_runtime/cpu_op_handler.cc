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

#include "cpu_op_handler.h"  // NOLINT

#include "cpu_op_registry_impl.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/dispatch_utils.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_handler_factory.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/host_tensor.h"

#define DEBUG_TYPE "tfrt-cpu-op-op_handler"

namespace tfrt {

namespace {

// If the specified tensor needs conversion to be compatible with CpuOpEntry,
// then return the allowed format mask.
uint32_t TensorNeedsConversion(const Tensor& t, const CpuOpEntry& entry) {
  // DenseHostTensor is always supported.
  uint32_t allowed_formats =
      1 << static_cast<uint32_t>(Tensor::Subclass::DenseHost);

  if (entry.flags & CpuOpFlags::AllowsScalar)
    allowed_formats |= 1 << static_cast<uint32_t>(Tensor::Subclass::ScalarHost);

  if (entry.flags & CpuOpFlags::AllowsCoo)
    allowed_formats |= 1 << static_cast<uint32_t>(Tensor::Subclass::CooHost);

  if (entry.flags & CpuOpFlags::AllowsTfLite)
    allowed_formats |= 1 << static_cast<uint32_t>(Tensor::Subclass::TFLiteHost);

  // If the tensor is already in a supported format, then we're done.
  if (allowed_formats & 1 << static_cast<uint32_t>(t.subclass())) return 0;

  // Otherwise return the mask of supported formats so a conversion can be
  // performed.
  return allowed_formats;
}

struct CpuOpHandlerTraits {
  using InputTensorTy = AsyncValue;
  using OpEntryTy = CpuOpEntry;

  static bool MaybeConvertTensor(const CpuOpEntry& op_entry,
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

  static void Dispatch(const CpuOpEntry& op_entry, ArrayRef<AsyncValue*> inputs,
                       const OpAttrsRef& attrs,
                       ArrayRef<TensorMetadata> result_mds,
                       MutableArrayRef<RCReference<AsyncValue>> results,
                       AsyncValueRef<Chain>* chain,
                       const ExecutionContext& exec_ctx) {
    op_entry.dispatch_fn(exec_ctx, inputs, attrs, result_mds, results, chain);
  }
};

}  // namespace

llvm::Expected<std::unique_ptr<CpuOpHandler>> CpuOpHandler::Create(
    CoreRuntime* runtime, OpHandler* fallback) {
  CpuOpRegistry op_registry;
  tfrt::RegisterStaticCpuOps(&op_registry);
  return std::make_unique<CpuOpHandler>(runtime, fallback,
                                        std::move(op_registry));
}

CpuOpHandler::CpuOpHandler(CoreRuntime* runtime, OpHandler* fallback,
                           CpuOpRegistry op_registry)
    : OpHandler("cpu", runtime, fallback),
      op_registry_(std::move(op_registry)) {}

CpuOpHandler::~CpuOpHandler() {}

AsyncValueRef<HostTensor> CpuOpHandler::CopyDeviceTensorToHost(
    const Tensor& tensor) {
  if (tensor.IsHostTensor()) {
    // If tensor is a host tensor, we call Tensor::ConvertToHostTensor
    // to make a copy of the tensor here, because the source and result buffers
    // are logically independent.
    auto host = GetRuntime()->GetHostContext();
    uint32_t allowed_formats =
        1 << static_cast<uint32_t>(Tensor::Subclass::DenseHost) |
        1 << static_cast<uint32_t>(Tensor::Subclass::StringHost);
    auto host_tensor = tensor.ConvertToHostTensor(host, allowed_formats);
    return AsyncValueRef<DenseHostTensor>(host_tensor.ReleaseRCRef());
  }

  // Otherwise, this copy is meant for the fallback device.
  return GetFallback()->CopyDeviceTensorToHost(tensor);
}

AsyncValueRef<Tensor> CpuOpHandler::CopyHostTensorToDevice(
    const DenseHostTensor& tensor) {
  // We call Tensor::ConvertToHostTensor to make a copy of the tensor here,
  // because the source and result buffers are logically independent.
  auto host = GetRuntime()->GetHostContext();
  uint32_t allowed_formats =
      1 << static_cast<uint32_t>(Tensor::Subclass::DenseHost);
  auto host_tensor = tensor.ConvertToHostTensor(host, allowed_formats);
  return AsyncValueRef<DenseHostTensor>(host_tensor.ReleaseRCRef());
}

//===----------------------------------------------------------------------===//
// Op Dispatch Implementation
//===----------------------------------------------------------------------===//

Expected<CoreRuntimeOp> CpuOpHandler::MakeOp(string_view op_name) {
  auto* op_entry = op_registry_.impl_->LookupOpEntry(op_name);

  // If this operation is unknown by cpu device, then we try to run it on
  // fallback device.
  if (op_entry->dispatch_fn == nullptr) return GetFallback()->MakeOp(op_name);

  // NOTE(fishx): To avoid introducing an extra heap allocation, we need to
  // ensure that the size of captured variable is smaller than 3 pointers.
  return CoreRuntimeOp([op_entry](const OpInvocation& invocation) {
    bool update_chain = !(op_entry->flags & CpuOpFlags::NoSideEffects);
    // TODO(fishx): ExecuteOnOpHandler should return void.
    ExecuteOnOpHandler<CpuOpHandlerTraits>(update_chain, invocation, *op_entry);
  });
}

}  // namespace tfrt
