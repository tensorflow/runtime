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

// This file implements kernels for testing distributed kernels.

#include "llvm/Support/raw_ostream.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "tfrt/compiler/compiler_pass.h"
#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/distributed_kernels.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/proto/remote_message.pb.h"
#include "tfrt/distributed_runtime/remote_chain_manager.h"
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
class TestRequestHandler final : public RequestHandlerInterface {
 public:
  explicit TestRequestHandler(ServerContext* server_context)
      : handler_(NewRequestHandler(server_context)) {}
  ~TestRequestHandler() final {}

#define TEST_HANDLE_METHOD(method)                                         \
  void Handle##method(const method##Request* request,                      \
                      method##Response* response, CallbackFn done) final { \
    handler_->Handle##method(request, response, std::move(done));          \
  }

  TEST_HANDLE_METHOD(GetDevices);
  TEST_HANDLE_METHOD(CreateContext);
  TEST_HANDLE_METHOD(CloseContext);
  TEST_HANDLE_METHOD(SendData);
  TEST_HANDLE_METHOD(SendReadyChains);
  TEST_HANDLE_METHOD(RegisterFunction);
  TEST_HANDLE_METHOD(DeleteRemoteObjects);
  TEST_HANDLE_METHOD(RemoteExecuteOp);
  TEST_HANDLE_METHOD(KeepAlive);

#undef TEST_HANDLE_METHOD

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

// Return N DistributedContext, one for each server
void TestCreateDistributedContext(RemainingArguments configurations,
                                  RemainingResults distributed_contexts,
                                  const ExecutionContext& exec_ctx) {
  llvm::SmallVector<ServerContext*, 4> servers;
  // Create remote servers
  for (int i = 0; i < configurations.size(); ++i) {
    const DistributedContextConfiguration& configuration =
        configurations[i]->get<DistributedContextConfiguration>();
    string_view server_address;
    for (const auto& job_config : configuration.cluster_config().jobs()) {
      if (job_config.name() == configuration.job_name()) {
        server_address = job_config.tasks().at(configuration.task_id());
        break;
      }
    }
    FabricCommunicatorConfiguration fabric_config{"grpc_communicator",
                                                  server_address.str()};
    ServerContextConfiguration server_config{fabric_config};
    servers.push_back(new TestServerContext(exec_ctx.host(), server_config));
  }
  // Create distributed context with context id 0 at the first server
  const DistributedContextConfiguration& configuration =
      configurations[0]->get<DistributedContextConfiguration>();
  Expected<DistributedContext*> dist_context_or_error =
      servers[0]->CreateDistributedContext(0, configuration);
  if (!dist_context_or_error) {
    Error error = dist_context_or_error.takeError();
    for (int i = 0; i < servers.size(); ++i) {
      distributed_contexts[i] =
          MakeErrorAsyncValueRef(exec_ctx.host(), DecodedDiagnostic(error));
    }
    return;
  }
  llvm::SmallVector<RCReference<IndirectAsyncValue>, 4> outputs;
  for (int i = 0; i < servers.size(); ++i) {
    outputs.push_back(distributed_contexts.AllocateIndirectResultAt(i));
  }
  DistributedContext* dist_context = dist_context_or_error.get();
  // Get device info on remote servers.
  dist_context->GetRemoteDevices(
      [dist_context, servers,
       outputs = std::move(outputs)](Error error) mutable {
        if (error) {
          for (int i = 0; i < outputs.size(); i++) {
            outputs[i]->SetError(DecodedDiagnostic(error));
          }
          return;
        }
        // Create other distributed context for remote servers.
        dist_context->CreateRemoteContexts(
            DistributedContext::RemoteInitMode::SINGLE_CLIENT,
            [servers, outputs = std::move(outputs)](Error error) mutable {
              for (int i = 0; i < outputs.size(); ++i) {
                if (error) {
                  outputs[i]->SetError(DecodedDiagnostic(error));
                } else {
                  AsyncValueRef<DistributedContext> c =
                      servers[i]->GetDistributedContextAsyncValue(0);
                  outputs[i]->ForwardTo(
                      servers[i]->GetDistributedContextAsyncValue(0));
                }
              }
            });
      });
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

AsyncValueRef<RemoteChainManager> TestCreateRemoteChainManager(
    Argument<DistributedContext> dist_context,
    const ExecutionContext& exec_ctx) {
  return MakeAvailableAsyncValueRef<RemoteChainManager>(exec_ctx.host(),
                                                        &dist_context.get());
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
      : compiled_program_(compiled_program), output_devices_(output_devices) {}

  mlir::OwningOpRef<mlir::ModuleOp> ParseMlirProgram(
      string_view program, mlir::MLIRContext* context) const override {
    mlir::DialectRegistry registry;
    tfrt::RegisterTFRTDialects(registry);
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect>();
    context->appendDialectRegistry(registry);
    context->allowUnregisteredDialects();

    TFRT_DLOG(INFO) << "Parsing: " << program;

    return mlir::parseSourceString(program, context);
  }

  llvm::Expected<CompilationOutput> Compile(
      mlir::ModuleOp module, mlir::MLIRContext* context) const override {
    mlir::DialectRegistry registry;
    RegisterTFRTDialects(registry);
    context->appendDialectRegistry(registry);
    context->allowUnregisteredDialects();

    CompilationOutput output;
    output.module = mlir::parseSourceString(compiled_program_, context);
    output.output_devices = output_devices_;

    return std::move(output);
  }

 private:
  mutable mlir::MLIRContext context_;
  std::string compiled_program_;
  llvm::SmallVector<std::string, 4> output_devices_;
};

AsyncValueRef<Chain> TestRegisterFakeCompilerPass(
    RemainingArguments inputs, StringAttribute compiled_program,
    StringAttribute compiler_pass_name, const ExecutionContext& exec_ctx) {
  llvm::SmallVector<std::string, 4> output_devices;
  for (int i = 0; i < inputs.size(); ++i) {
    output_devices.push_back(inputs[i]->get<std::string>());
  }
  RegisterCompilerPass(
      compiler_pass_name.str(),
      new FakeCompilerPass(compiled_program.get(), output_devices));
  return GetReadyChain();
}

Expected<RCReference<Device>> TestGetRemoteDevice(
    DistributedContext* dist_ctx, StringAttribute device_name,
    const ExecutionContext& exec_ctx) {
  auto remote_device = dist_ctx->GetRemoteDeviceManager()->GetDeviceRef<Device>(
      device_name.get());
  if (!remote_device) {
    return MakeStringError("cannot find remote device ", device_name.get());
  }
  return std::move(remote_device);
}

}  // namespace

void RegisterDistributedTestKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_dist.test_process_next_request",
                      TFRT_KERNEL(TestProcessNextRequest));
  registry->AddKernel("tfrt_dist.test_create_distributed_context",
                      TFRT_KERNEL(TestCreateDistributedContext));
  registry->AddKernel("tfrt_dist.test_close_distributed_context",
                      TFRT_KERNEL(TestCloseDistributedContext));
  registry->AddKernel("tfrt_dist.test_create_remote_chain_manager",
                      TFRT_KERNEL(TestCreateRemoteChainManager));
  registry->AddKernel("tfrt_dist.test_print_remote_object_id",
                      TFRT_KERNEL(TestPrintRemoteObjectId));
  registry->AddKernel("tfrt_dist.test_print_remote_execute_spec",
                      TFRT_KERNEL(TestPrintRemoteExecuteSpec));
  registry->AddKernel("tfrt_dist.test_register_fake_compiler_pass",
                      TFRT_KERNEL(TestRegisterFakeCompilerPass));
  registry->AddKernel("tfrt_dist.test_get_remote_device",
                      TFRT_KERNEL(TestGetRemoteDevice));
}

}  // namespace tfrt
