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
#include "tfrt/support/ref_count.h"

namespace tfrt {

class HostContext;
class ErrorAsyncValue;

// A request refers to either a BEFFunction execution or an op execution.
// RequestContext holds per request information, such as the cancellation status
// and request priority. A RequestContext object is reference counted and is
// passed around during the execution of a request. This allows us to support
// per-request actions, such as canceling all pending ops for a request and
// assigning all tasks of a request to a particular priority.

class RequestContext : public ReferenceCounted<RequestContext> {
 public:
  static RCReference<RequestContext> Create(HostContext* host) {
    return TakeRef(new RequestContext(host));
  }
  ~RequestContext();

  bool IsCancelled() const { return GetCancelAsyncValue(); }
  void Cancel();
  HostContext* host() const { return host_; }

  // If the request has been canceled, return an ErrorAsyncValue for
  // the cancellation. Otherwise, return nullptr.
  ErrorAsyncValue* GetCancelAsyncValue() const {
    return cancel_value_.load(std::memory_order_acquire);
  }

 private:
  explicit RequestContext(HostContext* host) : host_{host} {}

  HostContext* const host_ = nullptr;
  std::atomic<ErrorAsyncValue*> cancel_value_{nullptr};
};

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
  explicit ExecutionContext(RCReference<RequestContext> req_ctx,
                            Location location = {})
      : request_ctx_{std::move(req_ctx)}, location_{location} {}

  ExecutionContext(const ExecutionContext& exec_ctx)
      : request_ctx_{exec_ctx.request_ctx_.CopyRef()},
        location_{exec_ctx.location()} {}

  ExecutionContext(ExecutionContext&& exec_ctx)
      : request_ctx_{std::move(exec_ctx.request_ctx_)},
        location_{exec_ctx.location()} {}

  Location location() const { return location_; }
  HostContext* host() const { return request_ctx_->host(); }
  bool IsCancelled() const { return request_ctx_->IsCancelled(); }
  ErrorAsyncValue* GetCancelAsyncValue() const {
    return request_ctx_->GetCancelAsyncValue();
  }

  void set_location(Location location) { location_ = location; }

  RequestContext* request_ctx() const { return request_ctx_.get(); }

 private:
  RCReference<RequestContext> request_ctx_;
  Location location_;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_EXECUTION_CONTEXT_H_
