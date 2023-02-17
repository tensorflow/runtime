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

// This file implements ExecutionContext.

#include "tfrt/host_context/execution_context.h"

#include <utility>

#include "llvm/Support/Error.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {

CancellationContext::~CancellationContext() {
  if (auto cancel_value = GetCancelAsyncValue()) {
    cancel_value->DropRef();
  }
}

void CancellationContext::Cancel() {
  // Create an AsyncValue in error state for cancel.
  auto* error_value =
      MakeErrorAsyncValueRef(absl::CancelledError("Cancelled")).release();

  ErrorAsyncValue* expected_value = nullptr;
  // Use memory_order_release for the success case so that error_value is
  // visible to other threads when they load with memory_order_acquire. For the
  // failure case, we do not care about expected_value, so we can use
  // memory_order_relaxed.
  if (!cancel_value_.compare_exchange_strong(expected_value, error_value,
                                             std::memory_order_release,
                                             std::memory_order_relaxed)) {
    error_value->DropRef();
  }
}

void RequestContext::Cancel() { cancellation_->Cancel(); }

Expected<RCReference<RequestContext>> RequestContextBuilder::build() && {
  return TakeRef(new RequestContext(host_, resource_context_,
                                    std::move(context_data_), id_));
};

ExecutionContext::ExecutionContext(RCReference<RequestContext> req_ctx,
                                   Location location)
    : request_ctx_{std::move(req_ctx)},
      work_queue_(&host()->work_queue()),
      location_{location} {}

}  // namespace tfrt
