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

// This file implements the NullOpHandler, which always fails on op execution.
// It is the default fallback op handler.

#include "tfrt/cpu/core_runtime/null_op_handler.h"

#include <memory>

#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/support/error_util.h"

namespace tfrt {

class NullOpHandler : public OpHandler {
 public:
  ~NullOpHandler() override {}

  Expected<CoreRuntimeOp> MakeOp(string_view op_name) override {
    return MakeStringError(op_name, " was not supported by NullOpHandler.");
  }

  friend llvm::Expected<OpHandler*> CreateNullOpHandler(CoreRuntime* runtime);

 private:
  explicit NullOpHandler(CoreRuntime* runtime)
      : OpHandler("null", runtime, nullptr) {}
};

llvm::Expected<OpHandler*> CreateNullOpHandler(CoreRuntime* runtime) {
  if (!runtime) {
    return MakeStringError("Invalid Runtime");
  }
  auto null_op_handler =
      std::unique_ptr<NullOpHandler>(new NullOpHandler(runtime));
  auto null_op_handler_ptr = null_op_handler.get();
  runtime->TakeOpHandler(std::move(null_op_handler));
  return null_op_handler_ptr;
}

}  // namespace tfrt
