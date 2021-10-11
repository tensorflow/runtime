/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

#ifndef TFRT_UTILS_MLIR_RUNNER_UTIL_H_
#define TFRT_UTILS_MLIR_RUNNER_UTIL_H_

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/MLIRContext.h"
#include "tfrt/bef/bef_buffer.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/rc_array.h"

namespace tfrt {
namespace testing {

// This class is a utility class that provides support for users to specify an
// MLIR function, supply inputs and then have it compiled and run through TFRT.
// This class is primarily meant for use in unit tests and benchmarking.
class TfrtMlirRunner {
 public:
  class Builder {
   public:
    Builder();

    // Sets the MLIR function string and returns the object to chain setters.
    // Does not perform validation, will be validated when Compile is called.
    Builder& set_mlir_input(string_view mlir_input) {
      assert(!mlir_input.empty() && "MLIR input must not be empty.");
      mlir_input_ = mlir_input.str();
      return *this;
    }

    // Sets the MLIR function name that will be compiled and run, returns the
    // object to chain setters.
    Builder& set_mlir_fn_name(string_view fn_name) {
      assert(!fn_name.empty() && "Function name must not be empty.");
      fn_name_ = fn_name.str();
      return *this;
    }

    // Sets the `mlir_context` that should be used for compiling the MLIR code.
    // `mlir_context` must outlive TfrtMlirRunner.
    Builder& set_mlir_context(mlir::MLIRContext* mlir_context) {
      assert(mlir_context && "MLIR context must not be null.");
      mlir_context_ = mlir_context;
      return *this;
    }

    // Adds an input for the MLIR function. The inputs are wrapped in
    // AsyncValues before being passed in to the compiled function. The order of
    // the set calls must correspond exactly to the order in which the MLIR
    // function expects inputs.
    template <typename T, typename... Args>
    Builder& add_input(Args&&... args) {
      auto async_value_ref =
          tfrt::MakeAvailableAsyncValueRef<T>(std::forward<Args>(args)...);
      inputs_.push_back(std::move(async_value_ref));
      return *this;
    }

    // Compiles the MLIR function to BEF and returns a TfrtMlirRunner
    // object that can be used to Run the MLIR function of interest on TFRT and
    // extract outputs. Assert fails if any of mlir_input, fn_name, mlir_context
    // are not set.
    TfrtMlirRunner Compile();

   private:
    std::string mlir_input_;
    std::string fn_name_;
    std::vector<RCReference<AsyncValue>> inputs_;
    mlir::MLIRContext* mlir_context_ = nullptr;
    std::unique_ptr<HostContext> host_context_ = nullptr;
  };

  // Runs the MLIR function on TFRT and returns the outputs.
  llvm::SmallVector<RCReference<AsyncValue>> Run();

 private:
  // Use TfrtMlirRunner::Builder to get a TfrtMlirRunner object.
  TfrtMlirRunner(const std::string& fn_name, BefBuffer bef_buffer,
                 std::vector<RCReference<AsyncValue>> inputs);

  std::string fn_name_;
  std::vector<RCReference<AsyncValue>> inputs_;
  std::vector<tfrt::AsyncValue*> input_ptrs_;
  BefBuffer bef_buffer_;
  std::unique_ptr<HostContext> host_context_ = nullptr;
  RCReference<tfrt::BEFFile> bef_file_;
  const tfrt::Function* func_;
  std::unique_ptr<tfrt::ResourceContext> resource_context_ = nullptr;
  ExecutionContext execution_context_;
};

}  // namespace testing
}  // namespace tfrt

#endif  // TFRT_UTILS_MLIR_RUNNER_UTIL_H_
