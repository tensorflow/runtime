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

// RCReference<AsyncValue> wrapper
//
// This file implements AsyncValueRef.

#include "tfrt/host_context/async_value_ref.h"

#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/execution_context.h"

namespace tfrt {

RCReference<IndirectAsyncValue> MakeIndirectAsyncValue() {
  return TakeRef(internal::SimpleConstruct<IndirectAsyncValue>());
}

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            string_view message) {
  return EmitErrorAsync(exec_ctx, message, ErrorCode::kUnknown);
}

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            string_view message,
                                            ErrorCode code) {
  auto diag = EmitError(exec_ctx, message, code);
  return MakeErrorAsyncValueRef(std::move(diag));
}

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            llvm::Error error) {
  return EmitErrorAsync(exec_ctx, StrCat(error));
}

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            llvm::Error error, ErrorCode code) {
  return EmitErrorAsync(exec_ctx, StrCat(error), code);
}

RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(
    DecodedDiagnostic diagnostic) {
  // Create an AsyncValue for this error condition.
  auto* error_value =
      internal::SimpleConstruct<ErrorAsyncValue>(std::move(diagnostic));

  return TakeRef(error_value);
}

RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(string_view message) {
  return MakeErrorAsyncValueRef(DecodedDiagnostic(message));
}

}  // namespace tfrt
