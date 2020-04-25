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

//===- diagnostic.h - Decoded diagnostic abstraction ------------*- C++ -*-===//
//
// This file declares DecodedDiagnostic.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_DIAGNOSTIC_H_
#define TFRT_HOST_CONTEXT_DIAGNOSTIC_H_

#include "tfrt/host_context/location.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// This is a simple representation of a decoded diagnostic.
struct DecodedDiagnostic {
  explicit DecodedDiagnostic(const Error& error);
  explicit DecodedDiagnostic(string_view message) : message(message) {}
  DecodedDiagnostic(DecodedLocation location, string_view message)
      : location(std::move(location)), message(message) {}

  llvm::Optional<DecodedLocation> location;
  std::string message;
};

raw_ostream& operator<<(raw_ostream& os, const DecodedDiagnostic& diagnostic);

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_DIAGNOSTIC_H_
