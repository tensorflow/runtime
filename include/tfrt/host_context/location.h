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

//===- location.h -----------------------------------------------*- C++ -*-===//
//
// This file declares some routines for working with diagnostics: it declares
// the Location class, which is the primary encoded form of a location, as well
// as the LocationHandler class which decodes them.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_LOCATION_H_
#define TFRT_HOST_CONTEXT_LOCATION_H_

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class DecodedDiagnostic;
class HostContext;
class LocationHandler;
class ErrorAsyncValue;

// This is a simple representation of a source location. The
// filename/line/column are all optional.
struct DecodedLocation {
  std::string filename;
  int line = -1;
  int column = -1;
};

// This is an opaque location token that is passed to kernel implementations,
// and is used to report errors. It should remain a simple POD type.
class Location {
 public:
  Location() = default;
  Location(const LocationHandler *handler, intptr_t data)
      : data(data), handler_(handler) {}

  // Location has an optional semantic. A default constructed evaluates to
  // false.
  explicit operator bool() const { return handler_ != nullptr; }

  // Return the host for a location.  This will return null if the Location is
  // default constructed or otherwise null.
  HostContext *GetHost() const;

  DecodedLocation Decode() const;

  // Emit an error due to a dynamic condition that didn't work out right, such
  // as a shape error.
  //
  // For consistency, the error message should start with a lower case letter
  // and not end with a period.
  DecodedDiagnostic EmitError(string_view message) const;

  // Emit an error due to a dynamic condition that didn't work out right,
  // such as a shape error, and return an AsyncValue corresponding to the
  // result.
  //
  // For consistency, the error message should start with a lower case letter
  // and not end with a period.
  //
  RCReference<ErrorAsyncValue> EmitErrorAsync(string_view message) const;

  RCReference<ErrorAsyncValue> EmitErrorAsync(llvm::Error error) const;

  // Opaque implementation details of this location, only interpretable by the
  // location handler.
  intptr_t data = 0;

 private:
  friend class LocationHandler;
  const LocationHandler *handler_ = nullptr;
};

// This is a virtual base class used by things that create locations.
class LocationHandler {
 public:
  HostContext *GetHost() const { return host_; }

  virtual DecodedLocation DecodeLocation(Location loc) const = 0;

  virtual DecodedDiagnostic EmitError(Location loc, string_view message) const;

 protected:
  explicit LocationHandler(HostContext *host) : host_(host) {}

  // This intentionally does not have a virtual destructor, since the object
  // should never be destroyed through this.
  ~LocationHandler() {}

 private:
  virtual void VtableAnchor();

  /// This is the host context that owns this LocationHandler.
  HostContext *host_;
};

// Return the host for a location.  This will return null if the Location is
// default constructed or otherwise null.
inline HostContext *Location::GetHost() const {
  return handler_ ? handler_->GetHost() : nullptr;
}

inline DecodedLocation Location::Decode() const {
  assert(handler_);
  return handler_->DecodeLocation(*this);
}

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_LOCATION_H_
