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

//===- async_value_ref.cc - RCReference<AsyncValue> wrapper -----*- C++ -*-===//
//
// This file implements AsyncValueRef.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/async_value_ref.h"

#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {

RCReference<IndirectAsyncValue> MakeIndirectAsyncValue(HostContext* host) {
  return TakeRef(host->Construct<IndirectAsyncValue>(host));
}

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            string_view message) {
  auto diag = EmitError(exec_ctx, message);
  return MakeErrorAsyncValueRef(exec_ctx.host(), std::move(diag));
}

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            llvm::Error error) {
  return EmitErrorAsync(exec_ctx, StrCat(error));
}

RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(
    HostContext* host, DecodedDiagnostic&& diagnostic) {
  // Create an AsyncValue for this error condition.
  auto* error_value =
      HostContextConstruct<ErrorAsyncValue>(host, host, std::move(diagnostic));

  return TakeRef(error_value);
}

RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(HostContext* host,
                                                    string_view message) {
  return MakeErrorAsyncValueRef(host, DecodedDiagnostic(message));
}

}  // namespace tfrt
