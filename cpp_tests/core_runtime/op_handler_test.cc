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

//===- op_attrs_test.cc ---------------------------------------------------===//
//
// This file has unit tests for creating and registering OpHandlers.
//
//===----------------------------------------------------------------------===//

#include "tfrt/core_runtime/op_handler.h"

#include <memory>

#include "gtest/gtest.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace {

class DummyOpHandler : public OpHandler {
 public:
  DummyOpHandler(string_view op_handler_name, CoreRuntime* runtime)
      : OpHandler(op_handler_name, runtime, /*fallback=*/nullptr) {}

  Expected<CoreRuntimeOp> MakeOp(string_view op_name) override {
    return MakeStringError(op_name, " is not supported.");
  }
  AsyncValueRef<HostTensor> CopyDeviceTensorToHost(
      const ExecutionContext& exec_ctx, const Tensor& tensor) override {
    llvm_unreachable("not implemented");
  }
  AsyncValueRef<Tensor> CopyHostTensorToDevice(
      const DenseHostTensor& tensor) override {
    llvm_unreachable("not implemented");
  }
};

static std::unique_ptr<CoreRuntime> CreateCoreRuntime() {
  auto diag_handler = [](const DecodedDiagnostic& diag) {
    llvm::errs() << "Encountered runtime error: " << diag.message << "\n";
  };
  Expected<std::unique_ptr<CoreRuntime>> corert =
      CoreRuntime::Create(diag_handler, tfrt::CreateMallocAllocator(),
                          tfrt::CreateMultiThreadedWorkQueue(
                              /*num_threads=*/4, /*num_blocking_threads=*/64),
                          /*op_handler_chains=*/{"cpu"});

  assert(corert);
  return std::move(*corert);
}

TEST(OpHandlerTest, Registration) {
  auto chain_name = "CustomOpHandlerChain0";
  auto op_handler_name = "DummyOpHandler0";
  auto core_runtime = CreateCoreRuntime();

  // Before
  ASSERT_FALSE(core_runtime->GetOpHandler(chain_name));

  auto op_handler =
      std::make_unique<DummyOpHandler>(op_handler_name, core_runtime.get());
  // After DummyOpHandler construction
  ASSERT_FALSE(core_runtime->GetOpHandler(chain_name));

  auto chain_root = op_handler.get();
  core_runtime->TakeOpHandler(std::move(op_handler));
  // After DummyOpHandler ownership takeover
  ASSERT_FALSE(core_runtime->GetOpHandler(chain_name));

  core_runtime->RegisterOpHandler(chain_name, chain_root);
  // After Chain Registration
  ASSERT_EQ(core_runtime->GetOpHandler(chain_name), chain_root);
  ASSERT_FALSE(core_runtime->GetOpHandler(op_handler_name));
}
}  // namespace
}  // namespace tfrt
