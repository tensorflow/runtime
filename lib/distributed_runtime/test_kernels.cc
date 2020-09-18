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

//===- test_kernels.cc ----------------------------------------------------===//
//
// This file implements kernels for testing distributed kernels.
//
//===----------------------------------------------------------------------===//

#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/distributed_kernels.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/request_handler.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {

namespace {
// TestRequestHandler: A wrapper for RequestHandler to:
// - buffer the incoming requests
// - provide functionality to trigger request processing one by one by
//   process_next_request kernel.
// - delegate execution to RequestHandler
class TestRequestHandler : public FabricCommunicatorRequestHandler {
 public:
  explicit TestRequestHandler(DistributedContext* context)
      : handler_(context) {}
  ~TestRequestHandler() final {}

  void HandleRemoteRegister(const RemoteRegisterInvocation& request) final {
    handler_.HandleRemoteRegister(request);
  }

  void HandleRemoteExecute(const RemoteExecuteInvocation& request) final {
    mutex_lock lock(invocations_mutex_);
    // Need to deep copy the request since RPC would be marked as done and
    // hence, the underlying data request refers to will be deleted.
    string_store_.push(request.program_name.str());
    RemoteExecuteInvocation request_copy;
    request_copy.program_name = string_store_.back();
    request_copy.inputs = request.inputs;
    request_copy.outputs = request.outputs;
    invocations_.push(request_copy);
    cond_.notify_one();
  }

  void ProcessNextRequest() {
    RemoteExecuteInvocation invocation;
    {
      mutex_lock lock(invocations_mutex_);
      cond_.wait(lock, [this]() { return !this->invocations_.empty(); });
      invocation = invocations_.front();
      invocations_.pop();
    }
    handler_.HandleRemoteExecute(invocation);
  }

 private:
  mutex invocations_mutex_;
  tfrt::condition_variable cond_;
  std::queue<RemoteExecuteInvocation> invocations_;
  std::queue<std::string> string_store_;
  RequestHandler handler_;
};

// Process the next request in the DistributedContext's TestRequestHandler
// queue.
void TestProcessNextRequest(Argument<DistributedContext> dist_context,
                            Result<Chain> chain,
                            const ExecutionContext& exec_ctx) {
  AsyncValueRef<Chain> out = chain.Allocate();
  TestRequestHandler* handler =
      static_cast<TestRequestHandler*>(dist_context->GetRequestHandler());
  exec_ctx.host()->EnqueueWork([handler, out = out.CopyRef()] {
    handler->ProcessNextRequest();
    out.emplace();
  });
}

AsyncValueRef<DistributedContext> TestCreateDistributedContext(
    const DistributedContextConfiguration& configuration,
    const ExecutionContext& exec_ctx) {
  AsyncValueRef<DistributedContext> value =
      MakeAvailableAsyncValueRef<DistributedContext>(
          exec_ctx.host(), exec_ctx.host(), configuration);
  auto handler = std::make_unique<TestRequestHandler>(&value.get());
  value->Init(std::move(handler));
  return value;
}

}  // namespace

void RegisterDistributedTestKernels(KernelRegistry* registry) {
  registry->AddKernel("dist.test_process_next_request",
                      TFRT_KERNEL(TestProcessNextRequest));
  registry->AddKernel("dist.test_create_distributed_context",
                      TFRT_KERNEL(TestCreateDistributedContext));
}

}  // namespace tfrt
