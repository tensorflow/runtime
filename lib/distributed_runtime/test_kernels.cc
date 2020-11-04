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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "tfrt/compiler/compiler_pass.h"
#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/distributed_kernels.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/proto/remote_message.pb.h"
#include "tfrt/distributed_runtime/remote_execute.h"
#include "tfrt/distributed_runtime/remote_object.h"
#include "tfrt/distributed_runtime/request_handler.h"
#include "tfrt/distributed_runtime/request_handler_impl.h"
#include "tfrt/distributed_runtime/server_context.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/init_tfrt_dialects.h"

namespace tfrt {

namespace {
// TestRequestHandler: A wrapper for RequestHandler to:
// - buffer the incoming requests
// - provide functionality to trigger request processing one by one by
//   process_next_request kernel.
// - delegate execution to RequestHandler
class TestRequestHandler : public RequestHandlerInterface {
 public:
  explicit TestRequestHandler(ServerContext* server_context)
      : handler_(NewRequestHandler(server_context)) {}
  ~TestRequestHandler() final {}

  Error HandleSendData(const SendDataRequest* request,
                       SendDataResponse* response) final {
    return handler_->HandleSendData(request, response);
  }

  void HandleRegisterFunction(const RegisterFunctionRequest* request,
                              RegisterFunctionResponse* response,
                              CallbackFn done) final {
    return handler_->HandleRegisterFunction(request, response, std::move(done));
  }

  void HandleDeleteRemoteObjects(const DeleteRemoteObjectsRequest* request,
                                 DeleteRemoteObjectsResponse* response,
                                 CallbackFn done) final {
    return handler_->HandleDeleteRemoteObjects(request, response,
                                               std::move(done));
  }

  void HandleRemoteExecute(const RemoteExecuteRequest* request,
                           RemoteExecuteResponse* response,
                           CallbackFn done) final {
    mutex_lock lock(invocations_mutex_);
    Invocation invocation{request, response, std::move(done)};
    invocations_.push(std::move(invocation));
    cond_.notify_one();
  }

  void ProcessNextRequest(llvm::unique_function<void()> fn) {
    auto invocation = std::make_shared<Invocation>();
    {
      mutex_lock lock(invocations_mutex_);
      cond_.wait(lock, [this]() { return !this->invocations_.empty(); });
      *invocation = std::move(invocations_.front());
      invocations_.pop();
    }
    handler_->HandleRemoteExecute(
        invocation->request, invocation->response,
        [invocation, fn = std::move(fn)](Error e) mutable {
          invocation->callback_fn(std::move(e));
          fn();
        });
  }

 private:
  struct Invocation {
    const RemoteExecuteRequest* request;
    RemoteExecuteResponse* response;
    CallbackFn callback_fn;
  };
  mutex invocations_mutex_;
  tfrt::condition_variable cond_;
  std::queue<Invocation> invocations_;
  std::unique_ptr<RequestHandlerInterface> handler_;
};

class TestServerContext : public ServerContext {
 public:
  TestServerContext(HostContext* host_context,
                    ServerContextConfiguration configuration)
      : ServerContext(host_context, configuration) {
    ResetRequestHandler(std::make_unique<TestRequestHandler>(this));
  }
};

// Process the next request in the DistributedContext's TestRequestHandler
// queue.
void TestProcessNextRequest(Argument<DistributedContext> dist_context,
                            Result<Chain> chain,
                            const ExecutionContext& exec_ctx) {
  AsyncValueRef<Chain> out = chain.Allocate();
  TestRequestHandler* handler = static_cast<TestRequestHandler*>(
      dist_context->GetServerContext()->GetRequestHandler());
  EnqueueWork(exec_ctx, [handler, out = out.CopyRef()] {
    handler->ProcessNextRequest([out = out.CopyRef()] { out.emplace(); });
  });
}

AsyncValueRef<DistributedContext> TestCreateDistributedContext(
    const DistributedContextConfiguration& configuration,
    const ExecutionContext& exec_ctx) {
  const HostId id = configuration.cluster_config.id;
  const auto& server_address =
      configuration.cluster_config.addresses[id].address;
  FabricCommunicatorConfiguration fabric_config{"grpc_communicator",
                                                server_address};
  ServerContextConfiguration server_config{fabric_config};
  ServerContext* server = new TestServerContext(exec_ctx.host(), server_config);

  // Create distributed context with context id 0.
  // TODO(haoyuzhang): take context_id as op input and allow multiple
  // distributed contexts to be created in the same server context.
  Error e = server->CreateDistributedContext(0, configuration);
  if (e) {
    return MakeErrorAsyncValueRef(exec_ctx.host(),
                                  DecodedDiagnostic(std::move(e)));
  }
  return server->GetDistributedContextAsyncValue(0);
}

void TestCloseDistributedContext(Argument<DistributedContext> dist_context,
                                 Argument<Chain> in_chain,
                                 Result<Chain> out_chain,
                                 const ExecutionContext& exec_ctx) {
  auto out_chain_indirect = out_chain.Allocate();
  ServerContext* server = dist_context->GetServerContext();
  // TODO(haoyuzhang): allow individual distributed contexts to be closed.
  server->ShutDown();
  delete server;
  out_chain_indirect.emplace();
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

class FakeCompilerPass : public CompilerPass {
 public:
  ~FakeCompilerPass() override {}
  FakeCompilerPass(string_view compiled_program,
                   const llvm::SmallVector<std::string, 4>& output_devices)
      : compiled_program_(compiled_program), output_devices_(output_devices) {
    RegisterTFRTDialects(context_.getDialectRegistry());
    context_.allowUnregisteredDialects();
  }

  llvm::Expected<CompilationOutput> Compile(
      mlir::ModuleOp module) const override {
    CompilationOutput output;
    output.module = mlir::parseSourceString(compiled_program_, &context_);
    output.output_devices = output_devices_;

    return std::move(output);
  }

 private:
  mutable mlir::MLIRContext context_;
  std::string compiled_program_;
  llvm::SmallVector<std::string, 4> output_devices_;
};

void TestRegisterFakeCompilerPass(RemainingArguments inputs,
                                  StringAttribute compiled_program,
                                  StringAttribute compiler_pass_name,
                                  const ExecutionContext& exec_ctx) {
  llvm::SmallVector<std::string, 4> output_devices;
  for (int i = 0; i < inputs.size(); ++i) {
    output_devices.push_back(inputs[i]->get<std::string>());
  }
  RegisterCompilerPass(
      compiler_pass_name.str(),
      new FakeCompilerPass(compiled_program.get(), output_devices));
}

}  // namespace

void RegisterDistributedTestKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_dist.test_process_next_request",
                      TFRT_KERNEL(TestProcessNextRequest));
  registry->AddKernel("tfrt_dist.test_create_distributed_context",
                      TFRT_KERNEL(TestCreateDistributedContext));
  registry->AddKernel("tfrt_dist.test_close_distributed_context",
                      TFRT_KERNEL(TestCloseDistributedContext));
  registry->AddKernel("tfrt_dist.test_print_remote_object_id",
                      TFRT_KERNEL(TestPrintRemoteObjectId));
  registry->AddKernel("tfrt_dist.test_print_remote_execute_spec",
                      TFRT_KERNEL(TestPrintRemoteExecuteSpec));
  registry->AddKernel("tfrt_dist.test_register_fake_compiler_pass",
                      TFRT_KERNEL(TestRegisterFakeCompilerPass));
}

}  // namespace tfrt
