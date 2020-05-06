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

//===- composite_op_handler.cc
//-------------------------------------------------===//
//
// This file implements the CompositeOpHandler.
//
//===----------------------------------------------------------------------===//

#include "composite_op_handler.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/dispatch_utils.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/host_tensor.h"

#define DEBUG_TYPE "tfrt-composite-op-handler"

namespace tfrt {

llvm::Expected<std::unique_ptr<CompositeOpHandler>> CompositeOpHandler::Create(
    CoreRuntime* runtime, OpHandler* fallback) {
  return std::make_unique<CompositeOpHandler>(runtime);
}

CompositeOpHandler::CompositeOpHandler(CoreRuntime* runtime)
    : OpHandler("composite_op", runtime, nullptr) {}

AsyncValueRef<DenseHostTensor> CompositeOpHandler::CopyDeviceTensorToHost(
    const Tensor& tensor) {
  return GetFallback()->CopyDeviceTensorToHost(tensor);
}

AsyncValueRef<Tensor> CompositeOpHandler::CopyHostTensorToDevice(
    const DenseHostTensor& tensor) {
  return GetFallback()->CopyHostTensorToDevice(tensor);
}

CompositeOpHandler::~CompositeOpHandler() {}

bool CompositeOpHandler::RegisterCompositeOp(string_view name,
                                             RCReference<Function> fn) {
  assert(!name.empty() && "name cannot be empty");
  auto& entry = composite_op_mappings_[name];
  if (entry.dispatch_fn) return false;
  entry.dispatch_fn = std::move(fn);
  return true;
}

struct FunctionOpHandlerTraits {
  using InputTensorTy = AsyncValue;
  using OpEntryTy = FunctionOpEntry;

  static bool MaybeConvertTensor(const FunctionOpEntry& op_entry,
                                 const Tensor& arg_tensor,
                                 const ExecutionContext& exec_ctx,
                                 RCReference<AsyncValue>* converted) {
    return false;
  }

  static void Dispatch(const FunctionOpEntry& op_entry,
                       ArrayRef<AsyncValue*> inputs, const OpAttrsRef& attrs,
                       ArrayRef<TensorMetadata> result_mds,
                       MutableArrayRef<RCReference<AsyncValue>> results,
                       AsyncValueRef<Chain>* chain,
                       const ExecutionContext& exec_ctx) {
    op_entry.dispatch_fn->Execute(inputs, results, exec_ctx.host());
  }
};

//===----------------------------------------------------------------------===//
// Op Dispatch Implementation
//===----------------------------------------------------------------------===//

Expected<CoreRuntimeOp> CompositeOpHandler::MakeOp(string_view op_name) {
  auto composite_op_it = composite_op_mappings_.find(op_name);

  // If this operation is unknown by this device, then we fail to execute
  // without emitting an error.
  if (composite_op_it == composite_op_mappings_.end())
    return MakeStringError(op_name, " was not found in CompositeOpHandler");

  return CoreRuntimeOp(
      [metadata_fn = composite_op_it->second.metadata_fn,
       dispatch_fn = composite_op_it->second.dispatch_fn.CopyRef()](
          const OpInvocation& invocation) {
        FunctionOpEntry op_entry{.metadata_fn = metadata_fn,
                                 .dispatch_fn = dispatch_fn.CopyRef()};
        return ExecuteOnOpHandler<FunctionOpHandlerTraits>(
            /*update_chain=*/false, invocation, std::move(op_entry));
      });
}
}  // namespace tfrt
