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

#include <string_view>
#include <utility>

#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/execution_context.h"

namespace tfrt {

RCReference<IndirectAsyncValue> MakeIndirectAsyncValue() {
  return TakeRef(internal::SimpleConstruct<IndirectAsyncValue>());
}

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            absl::Status status) {
  return MakeErrorAsyncValueRef(EmitError(exec_ctx, status).status);
}

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            std::string_view message) {
  return EmitErrorAsync(exec_ctx, absl::InternalError(message));
}

RCReference<ErrorAsyncValue> EmitErrorAsync(const ExecutionContext& exec_ctx,
                                            Error error) {
  return EmitErrorAsync(exec_ctx,
                        absl::InternalError(toString(std::move(error))));
}

RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(absl::Status status) {
  auto* error_value =
      internal::SimpleConstruct<ErrorAsyncValue>(std::move(status));
  return TakeRef(error_value);
}

RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(std::string_view message) {
  return MakeErrorAsyncValueRef(absl::InternalError(message));
}

}  // namespace tfrt
