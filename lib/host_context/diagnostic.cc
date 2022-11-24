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

//===- diagnostic.cc - Decoded diagnostic abstraction ---------------------===//
//
// This file implements the decoded diagnostic abstraction.

#include "tfrt/host_context/diagnostic.h"

#include <string_view>
#include <utility>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"

namespace tfrt {

raw_ostream& operator<<(raw_ostream& os, const DecodedDiagnostic& diag) {
  if (diag.location) {
    os << diag.location.value() << ": ";
  } else {
    os << "UnknownLocation: ";
  }
  return os << diag.status.message();
}

DecodedDiagnostic EmitError(const ExecutionContext& exec_ctx,
                            absl::Status status) {
  auto decoded_loc = exec_ctx.location().Decode();
  auto diag = DecodedDiagnostic(decoded_loc, std::move(status));

  HostContext* host = exec_ctx.host();
  host->EmitError(diag);

  return diag;
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

}  // namespace tfrt
