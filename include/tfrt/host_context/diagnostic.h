/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Decoded diagnostic abstraction
//
// This file declares DecodedDiagnostic.

#ifndef TFRT_HOST_CONTEXT_DIAGNOSTIC_H_
#define TFRT_HOST_CONTEXT_DIAGNOSTIC_H_

#include <utility>

#include "tfrt/host_context/location.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/string_util.h"

namespace tfrt {

class ExecutionContext;

// This is a simple representation of a decoded diagnostic.
class DecodedDiagnostic {
 public:
  // TODO(b/169618466): carry error code in llvm::Error.
  explicit DecodedDiagnostic(const Error& error);
  explicit DecodedDiagnostic(string_view message) : message(message) {}
  DecodedDiagnostic(string_view message, ErrorCode code)
      : message(message), code(code) {}
  DecodedDiagnostic(DecodedLocation location, string_view message)
      : location(std::move(location)), message(message) {}
  DecodedDiagnostic(DecodedLocation location, string_view message,
                    ErrorCode code)
      : location(std::move(location)), message(message), code(code) {}

  llvm::Optional<DecodedLocation> location;
  std::string message;
  ErrorCode code{ErrorCode::kOK};
};

raw_ostream& operator<<(raw_ostream& os, const DecodedDiagnostic& diagnostic);

DecodedDiagnostic EmitError(const ExecutionContext& exec_ctx,
                            string_view message);

DecodedDiagnostic EmitError(const ExecutionContext& exec_ctx,
                            string_view message, ErrorCode code);

template <typename... Args>
DecodedDiagnostic EmitError(const ExecutionContext& exec_ctx, Args&&... args) {
  return EmitError(exec_ctx, string_view{StrCat(std::forward<Args>(args)...)});
}

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_DIAGNOSTIC_H_
