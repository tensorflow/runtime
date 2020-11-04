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

//===- request_handler_impl.cc - Request Handler ---------*- C++ -*--------===//
//
// This file contains implementation of RequestHandler class.
//
//===----------------------------------------------------------------------===//
#include "tfrt/distributed_runtime/request_handler_impl.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "tfrt/bef_converter/mlir_src_to_bef.h"
#include "tfrt/bef_converter/mlir_to_bef.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/compiler/compiler_pass.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/function_cache.h"
#include "tfrt/distributed_runtime/proto/remote_message.pb.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/distributed_runtime/request_handler.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {
namespace {
const char* kCompilerPassName = "partition_tf_dialect";

class RequestHandler : public RequestHandlerInterface {
 public:
  explicit RequestHandler(ServerContext* server_context)
      : server_context_(server_context) {}
  ~RequestHandler() override{};

  Error HandleSendData(const SendDataRequest* request,
                       SendDataResponse* response) final;

  void HandleRegisterFunction(const RegisterFunctionRequest* request,
                              RegisterFunctionResponse* response,
                              CallbackFn done) final;

  void HandleRemoteExecute(const RemoteExecuteRequest* request,
                           RemoteExecuteResponse* response,
                           CallbackFn done) final;

  void HandleDeleteRemoteObjects(const DeleteRemoteObjectsRequest* request,
                                 DeleteRemoteObjectsResponse* response,
                                 CallbackFn done) final;

 private:
  HostContext* host_ctx() { return server_context_->GetHostContext(); }

  ServerContext* server_context_;
};

Error RequestHandler::HandleSendData(const SendDataRequest* request,
                                     SendDataResponse* response) {
  auto expected = server_context_->GetDistributedContext(request->context_id());
  if (!expected) return expected.takeError();
  DistributedContext* dist_context = expected.get();

  InstanceKey key = request->instance_key();
  // TODO(ayushd): avoid string copy
  auto* payload = new std::string(request->payload());
  dist_context->GetCallbackRegistry()->SetValue(
      key, std::unique_ptr<std::string>(payload));
  return Error::success();
}

void RequestHandler::HandleRegisterFunction(
    const RegisterFunctionRequest* request, RegisterFunctionResponse* response,
    CallbackFn done) {
  auto expected = server_context_->GetDistributedContext(request->context_id());
  if (!expected) done(expected.takeError());
  DistributedContext* dist_context = expected.get();

  BEFBuffer bef_buffer;
  if (request->need_compilation()) {
    // Create MLIR module from the request.
    mlir::MLIRContext context;
    context.allowUnregisteredDialects();
    auto module = mlir::parseSourceString(request->program(), &context);

    const CompilerPass* pass = GetCompilerPass(kCompilerPassName);
    if (pass == nullptr) {
      TFRT_LOG(ERROR) << "Not implemented";
      done(llvm::make_error<NotFoundErrorInfo>(
          StrCat("Compiler pass not found for program: ",
                 request->program_name()),
          dist_context->GetTaskName()));
      return;
    }
    llvm::Expected<CompilerPass::CompilationOutput> output_or =
        pass->Compile(module.get());
    if (!output_or) {
      done(llvm::make_error<CompilationFailedErrorInfo>(
          StrCat("Failed to convert MLIR to BEF: ", request->program_name()),
          dist_context->GetTaskName()));
      return;
    }

    CompilerPass::CompilationOutput output = std::move(output_or.get());
    bef_buffer = ConvertMLIRToBEF(output.module.get(),
                                  /* disable_optional_sections = */ true);
    for (const auto& output_device : output.output_devices) {
      response->add_output_device(output_device);
    }
  } else {
    bef_buffer = ConvertMLIRSrcToBEF(request->program(),
                                     /* disable_optional_sections = */ true);
  }
  if (bef_buffer.empty()) {
    done(llvm::make_error<MalformattedMlirFileErrorInfo>(
        StrCat("Failed to convert MLIR to BEF: ", request->program_name()),
        dist_context->GetTaskName()));
    return;
  }
  FunctionCache* function_cache = dist_context->GetFunctionCache();
  done(
      function_cache->Register(request->program_name(), std::move(bef_buffer)));
}

void RequestHandler::HandleRemoteExecute(const RemoteExecuteRequest* request,
                                         RemoteExecuteResponse* response,
                                         CallbackFn done) {
  auto expected = server_context_->GetDistributedContext(request->context_id());
  if (!expected) {
    done(expected.takeError());
    return;
  }
  DistributedContext* dist_context = expected.get();

  FunctionCache* function_cache = dist_context->GetFunctionCache();
  auto cached_bef = function_cache->Prepare(request->program_name());
  RCReference<BEFFile>& bef_file = cached_bef.bef_file;
  if (bef_file.get() == nullptr) {
    done(llvm::make_error<InvalidArgumentErrorInfo>(
        StrCat("Can't find program: [", request->program_name(), "]")));
    return;
  }

  const Function* fn = bef_file->GetFunction(request->program_name());
  if (fn == nullptr) {
    done(llvm::make_error<InvalidArgumentErrorInfo>(
        StrCat("Failed to get program from BEFFile with name ",
               request->program_name(), ".")));
    return;
  }
  if (fn->result_types().size() != request->output_size()) {
    done(llvm::make_error<InvalidArgumentErrorInfo>(
        StrCat("Result size mismatch: fn #result: ", fn->result_types().size(),
               " Received #outputs: ", request->output_size())));
    return;
  }

  // TODO(bramandia): Propagate RequestContext from the request.
  ResourceContext resource_context;
  RCReference<tfrt::RequestContext> req_ctx =
      RequestContext::Create(host_ctx(), &resource_context);

  tfrt::ExecutionContext exec_ctx{std::move(req_ctx)};

  RemoteObjectManager* manager = dist_context->GetRemoteObjectManager();
  SmallVector<AsyncValue*, 4> arguments;
  SmallVector<RCReference<AsyncValue>, 4> arguments_ref;
  arguments.reserve(fn->argument_types().size());
  arguments_ref.reserve(fn->argument_types().size());
  // Allow the first argument to be `DistributedContext`.
  if (cached_bef.require_distributed_context) {
    AsyncValue* dist_context_arg =
        server_context_->GetDistributedContextAsyncValue(request->context_id())
            .GetAsyncValue();
    arguments.push_back(dist_context_arg);
  }
  if (cached_bef.require_preallocated_outputs) {
    for (int i = 0; i < request->output_size(); ++i) {
      auto& id = request->output(i).id();
      RCReference<Device> device =
          host_ctx()->GetDeviceManager()->GetDeviceRef<Device>(id.device());
      if (device.get() == nullptr) {
        done(llvm::make_error<DeviceNotFoundErrorInfo>(
            StrCat("Can't find device: ", id.device())));
        return;
      }
      RCReference<AsyncValue> remote_object_id =
          MakeAvailableAsyncValueRef<RemoteObjectId>(
              host_ctx(), id.prefix_id(), id.local_id(), device.CopyRef());
      arguments_ref.push_back(remote_object_id.CopyRef());
      arguments.push_back(remote_object_id.get());
    }
  }
  if (fn->argument_types().size() != arguments.size() + request->input_size()) {
    done(llvm::make_error<InvalidArgumentErrorInfo>(
        StrCat("Argument size mismatch: fn #arg: ", fn->argument_types().size(),
               " Received #inputs: ", request->input_size())));
    return;
  }
  for (int i = 0; i < request->input_size(); ++i) {
    auto& id = request->input(i);

    RCReference<Device> device =
        host_ctx()->GetDeviceManager()->GetDeviceRef<Device>(id.device());
    if (device.get() == nullptr) {
      done(llvm::make_error<DeviceNotFoundErrorInfo>(
          StrCat("Can't find device: ", id.device())));
      return;
    }
    RemoteObjectId input_id(id.prefix_id(), id.local_id(), device.CopyRef());
    RCReference<AsyncValue> val = manager->GetRemoteObject(input_id);
    arguments_ref.push_back(val.CopyRef());
    arguments.push_back(val.get());
  }
  auto results = std::make_unique<SmallVector<RCReference<AsyncValue>, 4>>();
  results->resize(fn->result_types().size());

  fn->Execute(exec_ctx, arguments, *results);
  for (int i = 0; i < request->output_size(); ++i) {
    auto& id = request->output(i).id();
    RCReference<Device> device =
        host_ctx()->GetDeviceManager()->GetDeviceRef<Device>(id.device());
    if (device.get() == nullptr) {
      done(llvm::make_error<DeviceNotFoundErrorInfo>(
          StrCat("Can't find device: ", id.device())));
      return;
    }
    // TODO(bramandia): Do not store the output in the map if the device is not
    // a local device.
    RemoteObjectId output_id(id.prefix_id(), id.local_id(), device.CopyRef());
    manager->SetRemoteObject(output_id, (*results)[i].CopyRef());
  }

  // get the pointer of results before being moved on the lambda capture.
  auto result_ref = results.get();
  // Request will live as long as done is not called yet.
  RunWhenReady(*result_ref, [fn, done = std::move(done), request, response,
                             results = std::move(results),
                             arguments = std::move(arguments),
                             arguments_ref =
                                 std::move(arguments_ref)]() mutable {
    for (int i = 0; i < request->output_size(); ++i) {
      if (request->output(i).need_metadata()) {
        if (fn->result_types()[i].GetName() == "!t.tensor") {
          std::string serialized =
              SerializeTensorMetadata((*results)[i]->get<Tensor>().metadata());
          response->add_metadata(serialized);
        } else if (fn->result_types()[i].GetName() == "!corert.tensorhandle") {
          std::string serialized = SerializeTensorMetadata(
              (*results)[i]->get<TensorHandle>().GetAvailableMetadata());
          response->add_metadata(serialized);
        } else {
          done(llvm::make_error<InvalidArgumentErrorInfo>(
              StrCat("Invalid type ", fn->result_types()[i].GetName())));
          return;
        }
      }
    }
    done(Error::success());
  });
}

void RequestHandler::HandleDeleteRemoteObjects(
    const DeleteRemoteObjectsRequest* request,
    DeleteRemoteObjectsResponse* response, CallbackFn done) {
  auto expected = server_context_->GetDistributedContext(request->context_id());
  if (!expected) {
    done(expected.takeError());
    return;
  }
  DistributedContext* dist_context = expected.get();

  llvm::SmallVector<RemoteObjectId, 4> ids;
  for (const RemoteObjectIdProto& id : request->input()) {
    RCReference<Device> device =
        host_ctx()->GetDeviceManager()->GetDeviceRef<Device>(id.device());
    if (device.get() == nullptr) {
      done(llvm::make_error<DeviceNotFoundErrorInfo>(
          StrCat("Can't find device: ", id.device())));
      return;
    }
    ids.emplace_back(id.prefix_id(), id.local_id(), device.CopyRef());
  }
  RemoteObjectManager* manager = dist_context->GetRemoteObjectManager();
  done(manager->DeleteRemoteObjects(ids));
}
}  // namespace

std::unique_ptr<RequestHandlerInterface> NewRequestHandler(
    ServerContext* server_context) {
  return std::make_unique<RequestHandler>(server_context);
}
}  // namespace tfrt
