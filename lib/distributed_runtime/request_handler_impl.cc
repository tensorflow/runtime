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
#include "tfrt/distributed_runtime/request_handler_impl.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/bef_converter/mlir_src_to_bef.h"
#include "tfrt/bef_converter/mlir_to_bef.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/compiler/compiler_pass.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/core_runtime_op.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/distributed_init_helper.h"
#include "tfrt/distributed_runtime/function_cache.h"
#include "tfrt/distributed_runtime/proto/remote_message.pb.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/distributed_runtime/request_handler.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/device.h"
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
const char* kCompilerPassName = "tfrt";
void ToProto(const RemoteObjectId& id, RemoteObjectIdProto* proto) {
  proto->set_prefix_id(id.prefix_id);
  proto->set_local_id(id.local_id);
  proto->set_device(id.device->name().str());
}

class RequestHandler : public RequestHandlerInterface {
 public:
  explicit RequestHandler(ServerContext* server_context)
      : server_context_(server_context) {}
  ~RequestHandler() override{};

  void HandleGetDevices(const GetDevicesRequest* request,
                        GetDevicesResponse* response, CallbackFn done) final;

  void HandleCreateContext(const CreateContextRequest* request,
                           CreateContextResponse* response,
                           CallbackFn done) final;

  void HandleCloseContext(const CloseContextRequest* request,
                          CloseContextResponse* response,
                          CallbackFn done) final;

  void HandleSendReadyChains(const SendReadyChainsRequest* request,
                             SendReadyChainsResponse* response,
                             CallbackFn done) final;

  void HandleSendData(const SendDataRequest* request,
                      SendDataResponse* response, CallbackFn done) final;

  void HandleRegisterFunction(const RegisterFunctionRequest* request,
                              RegisterFunctionResponse* response,
                              CallbackFn done) final;

  void HandleRemoteExecute(const RemoteExecuteRequest* request,
                           RemoteExecuteResponse* response,
                           CallbackFn done) final;

  void HandleRemoteExecuteOp(const RemoteExecuteOpRequest* request,
                             RemoteExecuteOpResponse* response,
                             CallbackFn done) final;

  void HandleDeleteRemoteObjects(const DeleteRemoteObjectsRequest* request,
                                 DeleteRemoteObjectsResponse* response,
                                 CallbackFn done) final;

  void HandleKeepAlive(const KeepAliveRequest* request,
                       KeepAliveResponse* response, CallbackFn done) final;

 private:
  HostContext* host_ctx() { return server_context_->GetHostContext(); }

  ServerContext* server_context_;
};

void RequestHandler::HandleGetDevices(const GetDevicesRequest* request,
                                      GetDevicesResponse* response,
                                      CallbackFn done) {
  auto devices = host_ctx()->GetDeviceManager()->ListDevices<Device>();
  for (auto& d : devices) {
    auto* device_info = response->add_devices();
    device_info->set_name(d->name().str());
    device_info->set_type(d->type().name().str());
  }
  done(Error::success());
}

void HandleCreateContextInternal(ServerContext* server_context,
                                 const CreateContextRequest* request,
                                 CreateContextResponse* response,
                                 CallbackFn done, bool is_multi_client) {
  const uint64_t ctx_id = request->context_id();
  auto expected = server_context->GetDistributedContext(ctx_id);
  if (expected) {
    done(llvm::make_error<DistributedContextAlreadyExistsErrorInfo>(
        StrCat("Failed to create DistributedContext: the context with id <",
               ctx_id, "> already exists.")));
    return;
  }

  Expected<DistributedContext*> context =
      server_context->CreateDistributedContext(ctx_id, request->dist_config());
  if (!context) {
    done(context.takeError());
    return;
  }
  DistributedContext* dist_context = context.get();
  for (const auto& device_info : request->devices()) {
    TaskHandle task_handle = dist_context->GetTaskHandle(device_info.name());
    auto expected =
        NewRemoteDevice(device_info.name(), device_info.type(), task_handle);
    if (expected) {
      dist_context->GetRemoteDeviceManager()->MaybeAddDevice(
          TakeRef(expected.get()));
    } else {
      done(expected.takeError());
      return;
    }
  }
  // Only track and GC expired context if its not created in multi-client mode
  if (!is_multi_client) {
    if (Error e = server_context->TrackContextAccessTime(ctx_id)) {
      done(std::move(e));
      return;
    }
  }
  ToProto(dist_context->LocalReadyChain(), response->mutable_ready_chain());
  done(Error::success());
}

void RequestHandler::HandleCreateContext(const CreateContextRequest* request,
                                         CreateContextResponse* response,
                                         CallbackFn done) {
  if (request->is_multi_client()) {
    auto remote_cb = [server_context = server_context_, request, response,
                      done = std::move(done)]() mutable -> Error {
      auto helper = server_context->GetDistributedInitHelper();
      if (!helper->IsConfigCompatible(request->dist_config())) {
        const std::string& error_msg = StrCat(
            "Incompatible distributed configuration when initializing "
            "multi-client distributed context. Expecting ",
            helper->GetLocalConfig().DebugString(), ", got ",
            request->dist_config().DebugString());
        done(llvm::make_error<InvalidArgumentErrorInfo>(error_msg));
        return llvm::make_error<InvalidArgumentErrorInfo>(error_msg);
      }
      HandleCreateContextInternal(server_context, request, response,
                                  std::move(done), request->is_multi_client());
      return Error::success();
    };
    server_context_->GetDistributedInitHelper()->RegisterRemoteCallback(
        std::move(remote_cb));
  } else {
    HandleCreateContextInternal(server_context_, request, response,
                                std::move(done), request->is_multi_client());
  }
}

void RequestHandler::HandleCloseContext(const CloseContextRequest* request,
                                        CloseContextResponse* response,
                                        CallbackFn done) {
  done(server_context_->CloseDistributedContext(request->context_id()));
}

void RequestHandler::HandleSendReadyChains(
    const SendReadyChainsRequest* request, SendReadyChainsResponse* response,
    CallbackFn done) {
  auto expected = server_context_->GetDistributedContext(request->context_id());
  if (!expected) {
    done(expected.takeError());
    return;
  }
  DistributedContext* context = expected.get();
  auto wrapped_done = [this, context, done = std::move(done)](Error e) mutable {
    if (auto* helper = server_context_->GetDistributedInitHelper()) {
      if (e) {
        helper->Complete(llvm::make_error<UnknownErrorInfo>(
            "Failed to finalize multi-client distributed context "
            "initialization with remote ready chains."));
      } else {
        helper->Complete(context);
      }
    }
    done(std::move(e));
  };

  for (const auto& ready_chain : request->ready_chains()) {
    TaskHandle task_handle = context->GetTaskHandle(ready_chain.device());
    // Do not add remote chain for the local task
    if (task_handle == context->GetTaskHandle()) continue;
    if (Error error = context->AddReadyChain(task_handle, ready_chain)) {
      wrapped_done(std::move(error));
      return;
    }
  }
  wrapped_done(Error::success());
}

void RequestHandler::HandleSendData(const SendDataRequest* request,
                                    SendDataResponse* response,
                                    CallbackFn done) {
  auto expected = server_context_->GetDistributedContext(request->context_id());
  if (!expected) {
    done(expected.takeError());
    return;
  }
  DistributedContext* dist_context = expected.get();

  InstanceKey key = request->instance_key();
  // TODO(ayushd): avoid string copy
  llvm::SmallVector<RCReference<HostBuffer>, 4> buffers;
  for (size_t i = 0; i < request->payload_size(); ++i) {
    auto buffer = tfrt::HostBuffer::CreateUninitialized(
        request->payload(i).size(), 1,
        server_context_->GetHostContext()->allocator());
    std::copy(request->payload(i).begin(), request->payload(i).end(),
              static_cast<char*>(buffer->data()));
    buffers.push_back(std::move(buffer));
  }
  dist_context->GetCallbackRegistry()->SetValue(key,
                                                Payload(std::move(buffers)));
  done(Error::success());
}

void RequestHandler::HandleRegisterFunction(
    const RegisterFunctionRequest* request, RegisterFunctionResponse* response,
    CallbackFn done) {
  auto expected = server_context_->GetDistributedContext(request->context_id());
  if (!expected) done(expected.takeError());
  DistributedContext* dist_context = expected.get();

  const CompilerPass* pass = GetCompilerPass(kCompilerPassName);
  BefBuffer bef_buffer;
  if (pass == nullptr) {
    done(llvm::make_error<NotFoundErrorInfo>(
        StrCat("Compiler pass not found for program: ",
               request->program_name()),
        dist_context->GetTaskName()));
    return;
  }
  mlir::MLIRContext context;
  mlir::OwningModuleRef module =
      pass->ParseMlirProgram(request->program(), &context);
  if (!module) {
    done(llvm::make_error<MalformattedMlirFileErrorInfo>(
        StrCat("Failed parsing program:", request->program_name()),
        dist_context->GetTaskName()));
    return;
  };
  if (request->need_compilation()) {
    llvm::Expected<CompilerPass::CompilationOutput> output_or =
        pass->Compile(module.get(), &context);
    if (!output_or) {
      done(llvm::make_error<CompilationFailedErrorInfo>(
          StrCat("Failed to convert MLIR to BEF: ", request->program_name()),
          dist_context->GetTaskName()));
      return;
    }
    CompilerPass::CompilationOutput output = std::move(output_or.get());
    for (const auto& output_device : output.output_devices) {
      response->add_output_device(output_device);
    }
    module = std::move(output.module);
  }
  bef_buffer = ConvertMLIRToBEF(module.get(),
                                /* disable_optional_sections = */ true);
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
  FunctionCache::CachedBEF* cached_bef =
      function_cache->Prepare(request->program_name());
  if (cached_bef == nullptr) {
    done(llvm::make_error<InvalidArgumentErrorInfo>(
        StrCat("Can't find program: [", request->program_name(), "]")));
    return;
  }
  RCReference<BEFFile>& bef_file = cached_bef->bef_file;
  if (bef_file.get() == nullptr) {
    done(llvm::make_error<InvalidArgumentErrorInfo>(
        StrCat("Can't find function: [", request->program_name(), "]")));
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
  Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(host_ctx(), &resource_context).build();
  if (!req_ctx) {
    done(llvm::make_error<UnknownErrorInfo>(
        StrCat("Failed to build RequestContext ", req_ctx.takeError())));
    return;
  }
  tfrt::ExecutionContext exec_ctx{std::move(*req_ctx)};

  RemoteObjectManager* manager = dist_context->GetRemoteObjectManager();
  SmallVector<AsyncValue*, 4> arguments;
  SmallVector<RCReference<AsyncValue>, 4> arguments_ref;
  arguments.reserve(fn->argument_types().size());
  arguments_ref.reserve(fn->argument_types().size());
  // Allow the first argument to be `DistributedContext`.
  if (cached_bef->require_distributed_context) {
    AsyncValue* dist_context_arg =
        server_context_->GetDistributedContextAsyncValue(request->context_id())
            .GetAsyncValue();
    arguments.push_back(dist_context_arg);
  }
  if (cached_bef->require_preallocated_outputs) {
    for (int i = 0; i < request->output_size(); ++i) {
      auto& id = request->output(i).id();
      RCReference<Device> device =
          dist_context->GetRemoteDeviceManager()->GetDeviceRef<Device>(
              id.device());
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
        dist_context->GetRemoteDeviceManager()->GetDeviceRef<Device>(
            id.device());
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
        dist_context->GetRemoteDeviceManager()->GetDeviceRef<Device>(
            id.device());
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

// Parse the RemoteObjectIdProto to get a RemoteObjectId.  Lookup the
// corresponding AsyncValue and emplace into `objects`.
llvm::Error GetRemoteObjectFromId(
    DeviceManager* device_manager, RemoteObjectManager* object_manager,
    const RemoteObjectIdProto& id_proto,
    llvm::SmallVectorImpl<RCReference<AsyncValue>>* objects) {
  auto device = device_manager->GetDeviceRef<Device>(id_proto.device());
  if (device.get() == nullptr) {
    return llvm::make_error<DeviceNotFoundErrorInfo>(
        StrCat("Cannot find device: ", id_proto.device()));
  }
  RemoteObjectId object_id(id_proto.prefix_id(), id_proto.local_id(),
                           std::move(device));
  objects->push_back(object_manager->GetRemoteObject(object_id));
  return Error::success();
}

// Parse the attrs proto to get OpAttrs for op execution.
// TODO(ayushd): add full attributes to op execution
void ParseOpAttrs(const RemoteExecuteOpRequest& request, OpAttrs* op_attrs) {
  for (const auto& attr : request.attributes()) {
    TFRT_DLOG(INFO) << "HandleRemoteExecuteOp " << request.op_name() << " attr "
                    << attr.DebugString();
    if (attr.is_array()) {
      op_attrs->SetArray(attr.name(), ArrayRef<int64_t>(attr.value().data(),
                                                        attr.value().size()));
    } else {
      op_attrs->Set(attr.name(), attr.value(0));
    }
  }
}

void RequestHandler::HandleRemoteExecuteOp(
    const RemoteExecuteOpRequest* request, RemoteExecuteOpResponse* response,
    CallbackFn done) {
  TFRT_DLOG(INFO) << "HandleRemoteExecuteOp " << request->op_name();
  auto expected_dist_ctx =
      server_context_->GetDistributedContext(request->context_id());
  if (!expected_dist_ctx) {
    done(expected_dist_ctx.takeError());
    TFRT_LOG(ERROR) << "Did not find DistributedContext with id "
                    << request->context_id();
    return;
  }
  DistributedContext* dist_ctx = expected_dist_ctx.get();
  HostContext* host_ctx = dist_ctx->GetHostContext();
  CoreRuntime* corert = CoreRuntime::GetFromHostContext(host_ctx);
  DeviceManager* device_manager = dist_ctx->GetRemoteDeviceManager();
  RemoteObjectManager* object_manager = dist_ctx->GetRemoteObjectManager();

  OpHandler* op_handler = corert->GetOpHandler(request->op_handler_name());
  Expected<CoreRuntimeOp> expected_op =
      corert->MakeOp(request->op_name(), op_handler);
  if (!expected_op) {
    done(expected_op.takeError());
    TFRT_LOG(ERROR) << "Could not MakeOp in RemoteOpHandler "
                    << request->op_name();
    return;
  }
  auto op = std::move(expected_op.get());
  auto device =
      device_manager->GetDeviceRef<Device>(request->in_chain().device());

  auto async_args =
      std::make_unique<llvm::SmallVector<RCReference<AsyncValue>, 4>>();
  async_args->reserve(request->input_size() + 1);  // TH inputs + in chain

  // Get the potentially async input chain.
  if (auto e = GetRemoteObjectFromId(device_manager, object_manager,
                                     request->in_chain(), async_args.get())) {
    TFRT_LOG(ERROR) << "Error while getting remote object "
                    << request->in_chain().DebugString() << " error: " << e;
    done(std::move(e));
    return;
  }

  // Get all other potentially async input tensors.
  for (auto i = 0; i < request->input_size(); ++i) {
    if (auto e = GetRemoteObjectFromId(device_manager, object_manager,
                                       request->input(i), async_args.get())) {
      done(std::move(e));
      return;
    }
    TFRT_DLOG(INFO) << "HandleRemoteExecuteOp wait for input "
                    << request->input(i).DebugString() << " av "
                    << async_args->back().get();
  }

  TFRT_DLOG(INFO) << "HandleRemoteExecuteOp " << request->op_name()
                  << " wait for " << async_args->size() << " inputs";
  auto async_args_ref = async_args.get();
  RunWhenReady(*async_args_ref, [host_ctx, dist_ctx, request, response,
                                 done = std::move(done), op = std::move(op),
                                 device = std::move(device),
                                 async_args = std::move(async_args)]() mutable {
    TFRT_DLOG(INFO) << "HandleRemoteExecuteOp " << request->op_name();
    AsyncValueRef<Chain> chain(FormRef((*async_args)[0].get()));

    // Get the actual Tensor inputs which should now be available.
    llvm::SmallVector<TensorHandle, 4> args;
    args.reserve(request->input_size());
    for (auto i = 1; i < async_args->size(); ++i) {
      AsyncValueRef<Tensor> tensor((*async_args)[i].CopyRef());
      args.emplace_back(device.CopyRef(), tensor->metadata(), tensor.CopyRef());
    }
    llvm::SmallVector<TensorHandle, 4> results;
    results.resize(request->output_size());

    // TODO(bramandia): Propagate RequestContext from the request.
    ResourceContext resource_context;
    Expected<RCReference<tfrt::RequestContext>> req_ctx =
        RequestContextBuilder(host_ctx, &resource_context).build();
    if (!req_ctx) {
      done(llvm::make_error<UnknownErrorInfo>(
          StrCat("Failed to build RequestContext ", req_ctx.takeError())));
      return;
    }
    tfrt::ExecutionContext exec_ctx{std::move(*req_ctx)};

    // Setup op attributes.
    OpAttrs op_attrs;
    ParseOpAttrs(*request, &op_attrs);

    op(exec_ctx, args, OpAttrsRef(op_attrs), results, &chain);

    // Set the output chain mapping in the remote object manager.
    RemoteObjectId out_chain_id(request->out_chain().prefix_id(),
                                request->out_chain().local_id(),
                                device.CopyRef());
    dist_ctx->GetRemoteObjectManager()->SetRemoteObject(out_chain_id,
                                                        chain.CopyRCRef());
    TFRT_DLOG(INFO) << "HandleRemoteExecuteOp " << request->op_name()
                    << " out chain " << request->out_chain().DebugString()
                    << " av " << chain.GetAsyncValue();

    // Wait for results to be ready before sending the response to client.
    // Also add a mapping in the remote object manager for each output.
    llvm::SmallVector<RCReference<AsyncValue>, 4> async_results;
    async_results.reserve(results.size() + 1);  // TH results + out chain
    async_results.push_back(chain.CopyRCRef());
    for (auto i = 0; i < results.size(); ++i) {
      async_results.push_back(FormRef(results[i].GetAsyncTensor()));

      auto& output_id = request->output(i).id();
      auto output_device =
          dist_ctx->GetRemoteDeviceManager()->GetDeviceRef<Device>(
              output_id.device());
      if (device.get() == nullptr) {
        done(llvm::make_error<DeviceNotFoundErrorInfo>(
            StrCat("Can't find device: ", output_id.device())));
        return;
      }
      RemoteObjectId object_id(output_id.prefix_id(), output_id.local_id(),
                               std::move(output_device));
      dist_ctx->GetRemoteObjectManager()->SetRemoteObject(
          object_id, async_results.back().CopyRef());
      TFRT_DLOG(INFO) << "HandleRemoteExecuteOp " << request->op_name()
                      << " output " << output_id.DebugString() << " av "
                      << async_results.back().get();
    }

    RunWhenReady(async_results, [request, response, device = std::move(device),
                                 chain = chain.CopyRef(),
                                 results = std::move(results),
                                 done = std::move(done)]() mutable {
      TFRT_DLOG(INFO) << "HandleRemoteExecuteOp " << request->op_name();
      if (chain.IsError()) {
        // TODO(ayushd): choose a more informative error code.
        done(llvm::make_error<UnknownErrorInfo>(chain.GetError().message));
      }
      for (auto& result : results) {
        auto serialized =
            SerializeTensorMetadata(result.GetAvailableMetadata());
        TFRT_DLOG(INFO) << "HandleRemoteExecuteOp " << request->op_name()
                        << " result metadata " << result.GetAvailableMetadata();
        response->add_metadata(serialized);
      }
      done(Error::success());
    });
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
        dist_context->GetRemoteDeviceManager()->GetDeviceRef<Device>(
            id.device());
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

void RequestHandler::HandleKeepAlive(const KeepAliveRequest* request,
                                     KeepAliveResponse* response,
                                     CallbackFn done) {
  auto expected = server_context_->GetDistributedContext(request->context_id());
  if (expected) {
    done(Error::success());
  } else {
    done(expected.takeError());
  }
}
}  // namespace

std::unique_ptr<RequestHandlerInterface> NewRequestHandler(
    ServerContext* server_context) {
  return std::make_unique<RequestHandler>(server_context);
}
}  // namespace tfrt
