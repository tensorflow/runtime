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
  TestRequestHandler() {}
  ~TestRequestHandler() final {}

  void HandleRemoteRegister(const RemoteRegisterInvocation& request) final {
    handler_.HandleRemoteRegister(request);
  }

  void HandleRemoteExecute(const RemoteExecuteInvocation& request) final {
    mutex_lock lock(invocations_mutex_);
    invocations_.push(request);
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
  RequestHandler handler_;
};

void TestSetupFromStringWithHandler(
    Argument<HostId> id, Result<DistributedContext> dist_context,
    std::unique_ptr<FabricCommunicatorRequestHandler> handler,
    const ExecutionContext& exec_ctx) {
  HostConfiguration host_config{{"localhost:50000", "localhost:50001",
                                 "localhost:50002", "localhost:50003"},
                                id.get()};
  FabricCommunicatorConfiguration fabric_config{"grpc_communicator",
                                                host_config};
  CollectiveGroup group1{/*name=*/"group1", /*members=*/{0, 1, 2, 3}};
  DistributedContextConfiguration dist_context_config{fabric_config, {group1}};
  AsyncValueRef<DistributedContext> context =
      MakeAvailableAsyncValueRef<DistributedContext>(
          exec_ctx.host(), exec_ctx.host(), std::move(handler),
          dist_context_config);
  dist_context.Set(context.CopyRef());
}

void TestSetupFromString(Argument<HostId> id,
                         Result<DistributedContext> dist_context,
                         const ExecutionContext& exec_ctx) {
  TestSetupFromStringWithHandler(id, dist_context, nullptr, exec_ctx);
}

void TestSetupWithRequestHandlerFromString(
    Argument<HostId> id, Result<DistributedContext> dist_context,
    const ExecutionContext& exec_ctx) {
  auto handler = std::make_unique<TestRequestHandler>();
  TestSetupFromStringWithHandler(id, dist_context, std::move(handler),
                                 exec_ctx);
}

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

}  // namespace

void RegisterDistributedTestKernels(KernelRegistry* registry) {
  registry->AddKernel("dist.test_setup_from_string",
                      TFRT_KERNEL(TestSetupFromString));
  registry->AddKernel("dist.test_setup_with_request_handler_from_string",
                      TFRT_KERNEL(TestSetupWithRequestHandlerFromString));
  registry->AddKernel("dist.test_process_next_request",
                      TFRT_KERNEL(TestProcessNextRequest));
}

}  // namespace tfrt
