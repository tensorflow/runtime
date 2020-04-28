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

//===- execution_context.h --------------------------------------*- C++ -*-===//
//
// This file declares ExecutionContext.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_EXECUTION_CONTEXT_H_
#define TFRT_HOST_CONTEXT_EXECUTION_CONTEXT_H_

#include "tfrt/host_context/location.h"

namespace tfrt {

class HostContext;

// ExecutionContext holds the context information for kernel and op execution,
// which currently includes the memory allocator, thread pool (memory allocator
// and thread pool are part of HostContext), and the location information. In
// the future, we plan to include other contextual information, such as client
// request id and request priority, and the request cancellation support, in the
// ExecutionContext as well.
//
// ExecutionContext is passed widely in the code base, as most code requires
// some of the facilities provided by ExecutionContext, e.g. memory allocation,
// dispatching async tasks, or reporting errors.

class ExecutionContext {
 public:
  explicit ExecutionContext(HostContext* host) : host_{host} {}

  Location location() const { return location_; }
  HostContext* host() const { return host_; }

  void set_location(Location location) { location_ = location; }

 private:
  Location location_;
  HostContext* host_ = nullptr;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_EXECUTION_CONTEXT_H_
