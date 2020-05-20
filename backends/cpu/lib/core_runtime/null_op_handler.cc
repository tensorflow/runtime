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

//===- null_op_handler.cc -------------------------------------------------===//
//
// This file implements the NullOpHandler, which always fails on op execution.
// It is the default fallback op handler.
//
//===----------------------------------------------------------------------===//

#include "null_op_handler.h"  // NOLINT

#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/op_handler_factory.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/error_util.h"

namespace tfrt {

llvm::Expected<std::unique_ptr<NullOpHandler>> NullOpHandler::Create(
    CoreRuntime* runtime, OpHandler* fallback) {
  return std::unique_ptr<NullOpHandler>(new NullOpHandler(runtime));
}

NullOpHandler::NullOpHandler(CoreRuntime* runtime)
    : OpHandler("null", runtime, nullptr) {}

NullOpHandler::~NullOpHandler() {}

AsyncValueRef<HostTensor> NullOpHandler::CopyDeviceTensorToHost(
    const Tensor& tensor) {
  assert(false && "NullOpHandler::CopyDeviceTensorToHost should not be called");
  abort();
}

AsyncValueRef<Tensor> NullOpHandler::CopyHostTensorToDevice(
    const DenseHostTensor& tensor) {
  assert(false && "NullOpHandler::CopyHostTensorToDevice should not be called");
  abort();
}

Expected<CoreRuntimeOp> NullOpHandler::MakeOp(string_view op_name) {
  return MakeStringError(op_name, " was not supported by NullOpHandler.");
}

}  // namespace tfrt
