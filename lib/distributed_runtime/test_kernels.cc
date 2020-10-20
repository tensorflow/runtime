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

#include "llvm/Support/raw_ostream.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/distributed_kernels.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/remote_execute.h"
#include "tfrt/distributed_runtime/request_handler.h"
#include "tfrt/host_context/async_dispatch.h"
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
  explicit TestRequestHandler(AsyncValueRef<DistributedContext> context)
      : handler_(context.CopyRef()) {}
  ~TestRequestHandler() final {}

  Error HandleRemoteRegister(const RemoteRegisterInvocation& request) final {
    return handler_.HandleRemoteRegister(request);
  }

  void HandleRemoteExecute(const RemoteExecuteInvocation& request,
                           RemoteExecuteCallbackFn done) final {
    mutex_lock lock(invocations_mutex_);
    invocations_.push({request, std::move(done)});
    cond_.notify_one();
  }

  using CallbackFn = llvm::unique_function<void()>;
  void ProcessNextRequest(CallbackFn fn) {
    std::unique_ptr<InvocationPair> invocation_pair;
    {
      mutex_lock lock(invocations_mutex_);
      cond_.wait(lock, [this]() { return !this->invocations_.empty(); });
      invocation_pair = std::make_unique<InvocationPair>();
      invocation_pair->invocation = invocations_.front().invocation;
      invocation_pair->callback_fn =
          std::move(invocations_.front().callback_fn);
      invocations_.pop();
    }
    auto& invocation = invocation_pair->invocation;
    handler_.HandleRemoteExecute(
        invocation,
        [invocation_pair = std::move(invocation_pair), fn = std::move(fn)](
            std::unique_ptr<RemoteExecuteInvocationResult> result) mutable {
          invocation_pair->callback_fn(std::move(result));
          fn();
        });
  }

 private:
  struct InvocationPair {
    RemoteExecuteInvocation invocation;
    RemoteExecuteCallbackFn callback_fn;
  };
  mutex invocations_mutex_;
  tfrt::condition_variable cond_;
  std::queue<InvocationPair> invocations_;
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
  EnqueueWork(exec_ctx, [handler, out = out.CopyRef()] {
    handler->ProcessNextRequest([out = out.CopyRef()] { out.emplace(); });
  });
}

AsyncValueRef<DistributedContext> TestCreateDistributedContext(
    const DistributedContextConfiguration& configuration,
    const ExecutionContext& exec_ctx) {
  AsyncValueRef<DistributedContext> value =
      MakeAvailableAsyncValueRef<DistributedContext>(
          exec_ctx.host(), exec_ctx.host(), configuration);
  auto handler = std::make_unique<TestRequestHandler>(value.CopyRef());
  value->Init(std::move(handler));
  return value;
}

void TestPrintRemoteObjectId(const RemoteObjectId& id,
                             const ExecutionContext& exec_ctx) {
  tfrt::outs() << id << "\n";
}

void TestPrintRemoteExecuteSpec(const RemoteExecuteSpec& id,
                                const ExecutionContext& exec_ctx) {
  tfrt::outs() << "RemoteExecuteSpec:[ ";
  for (const auto& device : id.output_devices) {
    tfrt::outs() << device->name() << " ";
  }
  tfrt::outs() << "]\n";
}

}  // namespace

void RegisterDistributedTestKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_dist.test_process_next_request",
                      TFRT_KERNEL(TestProcessNextRequest));
  registry->AddKernel("tfrt_dist.test_create_distributed_context",
                      TFRT_KERNEL(TestCreateDistributedContext));
  registry->AddKernel("tfrt_dist.test_print_remote_object_id",
                      TFRT_KERNEL(TestPrintRemoteObjectId));
  registry->AddKernel("tfrt_dist.test_print_remote_execute_spec",
                      TFRT_KERNEL(TestPrintRemoteExecuteSpec));
}

}  // namespace tfrt
