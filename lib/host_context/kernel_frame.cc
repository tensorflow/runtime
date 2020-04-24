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

//===- kernel_frame.cc - Information for kernel invocation ----------------===//
//
// This file implements KernelFrame which captures argument, result, and other
// related information provided to kernels on kernel invocation.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/kernel_frame.h"

#include "tfrt/support/ref_count.h"

namespace tfrt {

void KernelFrame::ReportError(string_view msg) {
  auto diag = location_.EmitError(msg);

  bool has_set_error = false;

  RCReference<AsyncValue> error_value;

  // Set any unavailable ConcreteAsyncValue to error and use that as error_value
  // for other results. We prefer reusing ConcreteAsyncValue instead of
  // allocating a new AsyncValue in error state for unset results or indirect
  // results.
  for (auto& result : GetResults()) {
    if (result && result->IsUnavailable()) {
      result->SetError(diag);
      has_set_error = true;

      if (!error_value) {
        result->AddRef();
        error_value.reset(result);
      }
    }
  }

  // If no error_value is set, create one.
  if (!error_value) {
    auto diag_copy = diag;
    error_value = host_->MakeErrorAsyncValueRef(std::move(diag_copy));
  }

  // Set unset results to error AsyncValue.
  for (auto& result : GetResults()) {
    if (!result) {
      // Must AddRef on each iteration.
      result = error_value.CopyRef().release();
      has_set_error = true;
    } else if (result->IsUnavailable()) {
      result->SetError(diag);
      has_set_error = true;
    }
  }
  assert(has_set_error && "ReportError must set at least one error");
}

}  // namespace tfrt
