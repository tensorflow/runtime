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

//===- request_handler.cc - Request Handler --------------*- C++ -*--------===//
//
// This file contains implementation of RequestHandler class.
//
//===----------------------------------------------------------------------===//
#include "tfrt/distributed_runtime/request_handler.h"

#include <unordered_map>

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
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {
namespace {
const char* kCompilerPassName = "partition_tf_dialect";
}  // namespace
// TODO(bramandia): Replace this with TFRT FunctionLibrary once available.
class RequestHandler::FunctionCache {
 public:
  explicit FunctionCache(HostContext* host_context) : host_(host_context) {}

  // Register the given program. A program can have multiple functions in it.
  // The program_name serves as both unique ID of this program.
  Error Register(const std::string& program_name, BEFBuffer bef_buffer);

  // Create BEFFile corresponding to the program with the given name.
  // A struct representing a BEFFile and the respective buffer.
  struct CachedBEF {
    CachedBEF() {}
    CachedBEF(const CachedBEF& cached_bef)
        : bef_file(cached_bef.bef_file.CopyRef()),
          require_distributed_context(cached_bef.require_distributed_context),
          require_preallocated_outputs(
              cached_bef.require_preallocated_outputs) {}

    RCReference<BEFFile> bef_file;
    bool require_distributed_context = false;
    bool require_preallocated_outputs = false;
  };
  CachedBEF Prepare(const std::string& program_name);

 private:
  HostContext* host_;

  mutex cached_bef_mutex_;
  // Map from the program name to the CachedBEF.
  std::unordered_map<std::string, std::pair<BEFBuffer, CachedBEF>> cached_bef_
      TFRT_GUARDED_BY(cached_bef_mutex_);
};

Error RequestHandler::FunctionCache::Register(const std::string& program_name,
                                              BEFBuffer bef_buffer) {
  RCReference<BEFFile> bef_file =
      tfrt::BEFFile::Open(bef_buffer, host_->GetKernelRegistry(),
                          host_->diag_handler(), host_->allocator());

  if (!bef_file) {
    return llvm::make_error<MalformattedMlirFileErrorInfo>(
        StrCat("Failed to open lowered BEF for function ", program_name, "."));
  }
  const Function* fn = bef_file->GetFunction(program_name);
  int arg_index = 0;
  bool require_distributed_context = false;
  if (fn->num_arguments() > 0 &&
      fn->argument_types()[0].GetName() == "!tfrt_dist.dist_context") {
    require_distributed_context = true;
    arg_index++;
  }
  // If the next argument is RemoteObjectId, this is the RemoteObjectId outputs.
  // We have to pass all the output remote object IDs.
  bool require_preallocated_outputs = false;
  if (fn->num_arguments() > arg_index &&
      fn->argument_types()[arg_index].GetName() ==
          "!tfrt_dist.remote_object_id") {
    require_preallocated_outputs = true;
  }
  mutex_lock lock(cached_bef_mutex_);
  if (cached_bef_.find(program_name) != cached_bef_.end()) {
    return llvm::make_error<RemoteFunctionAlreadyExistsErrorInfo>(
        StrCat("Program ", program_name, " already registered."));
  }
  auto& cached = cached_bef_[program_name];
  cached.first = std::move(bef_buffer);
  cached.second.bef_file = bef_file.CopyRef();
  cached.second.require_distributed_context = require_distributed_context;
  cached.second.require_preallocated_outputs = require_preallocated_outputs;
  return Error::success();
}

RequestHandler::FunctionCache::CachedBEF RequestHandler::FunctionCache::Prepare(
    const std::string& program_name) {
  mutex_lock lock(cached_bef_mutex_);
  auto iter = cached_bef_.find(program_name);
  if (iter != cached_bef_.end()) {
    return iter->second.second;
  }
  return CachedBEF();
}

RequestHandler::RequestHandler(AsyncValueRef<DistributedContext> context)
    : dist_ctx_(context.GetAsyncValue()) {
  function_cache_ = std::make_unique<FunctionCache>(host_ctx());
}

RequestHandler::~RequestHandler() {}

void RequestHandler::HandleRemoteRegister(
    const RemoteRegisterInvocation& request, RemoteRegisterCallbackFn done) {
  auto response =
      std::make_unique<RemoteRegisterInvocationResult>(Error::success());
  BEFBuffer bef_buffer;
  if (request.need_compilation) {
    // Create MLIR module from the request.
    mlir::MLIRContext context;
    context.allowUnregisteredDialects();
    auto module = mlir::parseSourceString(request.program, &context);

    const CompilerPass* pass = GetCompilerPass(kCompilerPassName);
    if (pass == nullptr) {
      TFRT_LOG(ERROR) << "Not implemented";
      response->error = llvm::make_error<NotFoundErrorInfo>(
          StrCat("Compiler pass not found for program: ", request.program_name),
          dist_ctx()->GetTaskName());
      done(std::move(response));
      return;
    }
    llvm::Expected<CompilerPass::CompilationOutput> output_or =
        pass->Compile(module.get());
    if (!output_or) {
      response->error = llvm::make_error<CompilationFailedErrorInfo>(
          StrCat("Failed to convert MLIR to BEF: ", request.program_name),
          dist_ctx()->GetTaskName());
      done(std::move(response));
      return;
    }

    CompilerPass::CompilationOutput output = std::move(output_or.get());
    bef_buffer = ConvertMLIRToBEF(output.module.get(),
                                  /* disable_optional_sections = */ true);
    for (const auto& output_device : output.output_devices) {
      response->output_device.push_back(output_device);
    }
  } else {
    bef_buffer = ConvertMLIRSrcToBEF(request.program,
                                     /* disable_optional_sections = */ true);
  }
  if (bef_buffer.empty()) {
    response->error = llvm::make_error<MalformattedMlirFileErrorInfo>(
        StrCat("Failed to convert MLIR to BEF: ", request.program_name),
        dist_ctx()->GetTaskName());
    done(std::move(response));
    return;
  }
  response->error = function_cache_->Register(request.program_name.str(),
                                              std::move(bef_buffer));
  done(std::move(response));
}

void RequestHandler::HandleRemoteExecute(const RemoteExecuteInvocation& request,
                                         RemoteExecuteCallbackFn done) {
  auto cached_bef = function_cache_->Prepare(request.program_name.str());
  RCReference<BEFFile>& bef_file = cached_bef.bef_file;
  if (bef_file.get() == nullptr) {
    Error e = llvm::make_error<InvalidArgumentErrorInfo>(
        StrCat("Can't find program: [", request.program_name, "]"));
    done(std::make_unique<RemoteExecuteInvocationResult>(std::move(e)));
    return;
  }
  const Function* fn = bef_file->GetFunction(request.program_name);
  if (fn == nullptr) {
    Error e = llvm::make_error<InvalidArgumentErrorInfo>(
        StrCat("Failed to get program from BEFFile with name ",
               request.program_name, "."));
    done(std::make_unique<RemoteExecuteInvocationResult>(std::move(e)));
    return;
  }
  if (fn->result_types().size() != request.outputs.size()) {
    Error e = llvm::make_error<InvalidArgumentErrorInfo>(
        StrCat("Result size mismatch: fn #result: ", fn->result_types().size(),
               " Received #outputs: ", request.outputs.size()));
    done(std::make_unique<RemoteExecuteInvocationResult>(std::move(e)));
    return;
  }

  // TODO(bramandia): Propagate RequestContext from the request.
  ResourceContext resource_context;
  RCReference<tfrt::RequestContext> req_ctx =
      RequestContext::Create(host_ctx(), &resource_context);

  tfrt::ExecutionContext exec_ctx{std::move(req_ctx)};

  RemoteObjectManager* manager = dist_ctx()->GetRemoteObjectManager();
  SmallVector<AsyncValue*, 4> arguments;
  SmallVector<RCReference<AsyncValue>, 4> arguments_ref;
  arguments.reserve(fn->argument_types().size());
  arguments_ref.reserve(fn->argument_types().size());
  // Allow the first argument to be `DistributedContext`.
  if (cached_bef.require_distributed_context) {
    arguments.push_back(dist_ctx_);
  }
  if (cached_bef.require_preallocated_outputs) {
    for (int i = 0; i < request.outputs.size(); ++i) {
      auto& id = request.outputs[i].id;
      RCReference<Device> device =
          host_ctx()->GetDeviceManager()->GetDeviceRef<Device>(id.device);
      if (device.get() == nullptr) {
        Error e = llvm::make_error<DeviceNotFoundErrorInfo>(
            StrCat("Can't find device: ", id.device));
        done(std::make_unique<RemoteExecuteInvocationResult>(std::move(e)));
        return;
      }
      RCReference<AsyncValue> remote_object_id =
          MakeAvailableAsyncValueRef<RemoteObjectId>(
              host_ctx(), id.prefix_id, id.local_id, device.CopyRef());
      arguments_ref.push_back(remote_object_id.CopyRef());
      arguments.push_back(remote_object_id.get());
    }
  }
  if (fn->argument_types().size() != arguments.size() + request.inputs.size()) {
    Error e = llvm::make_error<InvalidArgumentErrorInfo>(
        StrCat("Argument size mismatch: fn #arg: ", fn->argument_types().size(),
               " Received #inputs: ", request.inputs.size()));
    done(std::make_unique<RemoteExecuteInvocationResult>(std::move(e)));
    return;
  }
  for (int i = 0; i < request.inputs.size(); ++i) {
    auto& id = request.inputs[i];

    RCReference<Device> device =
        host_ctx()->GetDeviceManager()->GetDeviceRef<Device>(id.device);
    if (device.get() == nullptr) {
      Error e = llvm::make_error<DeviceNotFoundErrorInfo>(
          StrCat("Can't find device: ", id.device));
      done(std::make_unique<RemoteExecuteInvocationResult>(std::move(e)));
      return;
    }
    RemoteObjectId input_id(id.prefix_id, id.local_id, device.CopyRef());
    RCReference<AsyncValue> val = manager->GetRemoteObject(input_id);
    arguments_ref.push_back(val.CopyRef());
    arguments.push_back(val.get());
  }
  auto results = std::make_unique<SmallVector<RCReference<AsyncValue>, 4>>();
  results->resize(fn->result_types().size());

  fn->Execute(exec_ctx, arguments, *results);
  for (int i = 0; i < request.outputs.size(); ++i) {
    auto& id = request.outputs[i].id;
    RCReference<Device> device =
        host_ctx()->GetDeviceManager()->GetDeviceRef<Device>(id.device);
    if (device.get() == nullptr) {
      Error e = llvm::make_error<DeviceNotFoundErrorInfo>(
          StrCat("Can't find device: ", id.device));
      done(std::make_unique<RemoteExecuteInvocationResult>(std::move(e)));
      return;
    }
    // TODO(bramandia): Do not store the output in the map if the device is not
    // a local device.
    RemoteObjectId output_id(id.prefix_id, id.local_id, device.CopyRef());
    manager->SetRemoteObject(output_id, (*results)[i].CopyRef());
  }

  // get the pointer of results before being moved on the lambda capture.
  auto result_ref = results.get();
  // Request will live as long as done is not called yet.
  RunWhenReady(*result_ref, [fn, done = std::move(done), request,
                             results = std::move(results),
                             arguments = std::move(arguments),
                             arguments_ref =
                                 std::move(arguments_ref)]() mutable {
    auto response =
        std::make_unique<RemoteExecuteInvocationResult>(Error::success());
    for (int i = 0; i < request.outputs.size(); ++i) {
      if (request.outputs[i].need_metadata) {
        if (fn->result_types()[i].GetName() == "!t.tensor") {
          std::string serialized =
              SerializeTensorMetadata((*results)[i]->get<Tensor>().metadata());
          response->metadata.push_back(serialized);
        } else if (fn->result_types()[i].GetName() == "!corert.tensorhandle") {
          std::string serialized = SerializeTensorMetadata(
              (*results)[i]->get<TensorHandle>().GetAvailableMetadata());
          response->metadata.push_back(serialized);
        } else {
          response->error = llvm::make_error<InvalidArgumentErrorInfo>(
              StrCat("Invalid type ", fn->result_types()[i].GetName()));
          done(std::move(response));
          return;
        }
      }
    }
    done(std::move(response));
  });
}

HostContext* RequestHandler::host_ctx() {
  return dist_ctx_->get<DistributedContext>().GetHostContext();
}

DistributedContext* RequestHandler::dist_ctx() {
  return &dist_ctx_->get<DistributedContext>();
}
}  // namespace tfrt
