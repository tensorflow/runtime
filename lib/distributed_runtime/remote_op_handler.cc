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

// RemoteOpHandler
//
// This file contains an implementation of RemoteOpHandler.

#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/remote_chain_manager.h"
#include "tfrt/distributed_runtime/remote_device.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/distributed_runtime/remote_tensor.h"
#include "tfrt/distributed_runtime/task_handle.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {

namespace {

struct TensorAndMetadata {
  RCReference<AsyncValue> tensor;
  AsyncValueRef<TensorMetadata> metadata;
  std::unique_ptr<RemoteObjectId> remote_object_id;
};

void PopulateRemoteObjectIdProto(RemoteObjectIdProto* proto,
                                 const RemoteObjectId& id) {
  proto->set_prefix_id(id.prefix_id);
  proto->set_local_id(id.local_id);
  proto->set_device(id.device->name().str());
}

void PopulateRemoteExecuteOutputProto(RemoteExecuteOutput* proto,
                                      const RemoteObjectId& id) {
  PopulateRemoteObjectIdProto(proto->mutable_id(), id);
  proto->set_need_metadata(true);
}

// TODO(ayushd): support attribute types other than I64.
Error PopulateRequestAttrsProto(RemoteExecuteOpRequest* request,
                                OpAttrsRef attrs) {
  Error attr_error = Error::success();
  attrs.IterateEntries(
      [&attrs, request, &attr_error](const OpAttrsRawEntry& entry) {
        if (entry.type != OpAttrType::I64) {
          attr_error = MakeStringError(
              "unexpected attribute type for RemoteOpHandler, only I64 "
              "supported");
          return;
        }

        auto* attr = request->add_attributes();
        attr->set_name(entry.name);
        if (entry.IsArray()) {
          attr->set_is_array(true);
          llvm::ArrayRef<int64_t> int_array =
              attrs.GetArrayAsserting<int64_t>(entry.name);
          for (int64_t value : int_array) {
            attr->add_value(value);
          }
        } else {
          attr->set_is_array(false);
          attr->add_value(attrs.GetAsserting<int64_t>(entry.name));
        }
      });
  return attr_error;
}

}  // namespace

class RemoteOpHandler : public OpHandler {
 public:
  explicit RemoteOpHandler(DistributedContext* dist_ctx,
                           RemoteChainManager* remote_chain_manager,
                           RCReference<RemoteDevice> remote_device);

  ~RemoteOpHandler() override;

  Expected<CoreRuntimeOp> MakeOp(string_view op_name) override;

 private:
  void Execute(const std::string& op_name, const OpInvocation& invocation);

  DistributedContext* dist_ctx_;
  RemoteChainManager* remote_chain_manager_;
  RCReference<RemoteDevice> remote_device_;
};

RemoteOpHandler::RemoteOpHandler(DistributedContext* dist_ctx,
                                 RemoteChainManager* remote_chain_manager,
                                 RCReference<RemoteDevice> remote_device)
    : OpHandler(/*name=*/"remote",
                CoreRuntime::GetFromHostContext(dist_ctx->GetHostContext()),
                /*fallback=*/nullptr),
      dist_ctx_(dist_ctx),
      remote_chain_manager_(remote_chain_manager),
      remote_device_(std::move(remote_device)) {}

RemoteOpHandler::~RemoteOpHandler() {}

Expected<CoreRuntimeOp> RemoteOpHandler::MakeOp(string_view op_name) {
  return CoreRuntimeOp(
      [op_name = op_name.str(), this](const OpInvocation& invocation) {
        Execute(op_name, invocation);
      },
      /*is_fallback=*/false, remote_device_);
}

void RemoteOpHandler::Execute(const std::string& op_name,
                              const OpInvocation& invocation) {
  // Wait for async input tensors before starting to dispatch the
  // remote op.
  llvm::SmallVector<AsyncValue*, 4> to_wait;

  auto chain = MakeConstructedAsyncValueRef<Chain>(dist_ctx_->GetHostContext());
  *invocation.chain = chain.CopyRef();

  // We may need to wait for async inputs, so we add input object ids
  // after waiting.
  auto arguments =
      std::make_unique<llvm::SmallVector<RCReference<AsyncValue>, 8>>();
  arguments->resize(invocation.arguments.size());
  for (auto i = 0; i < invocation.arguments.size(); ++i) {
    auto* tensor_av = invocation.arguments[i].GetAsyncTensor();
    (*arguments)[i] = FormRef(tensor_av);
    if (!tensor_av->IsAvailable()) {
      to_wait.push_back(tensor_av);
    }
  }

  auto request = std::make_unique<RemoteExecuteOpRequest>();
  request->set_context_id(dist_ctx_->GetContextId());
  request->set_op_handler_name(remote_device_->name().str());
  request->set_op_name(op_name);

  // Add output object ids to the request.
  // TODO(ayushd): optimize so that metadata is not always asynchronous.
  auto results = std::make_unique<llvm::SmallVector<TensorAndMetadata, 8>>();
  results->resize(invocation.results.size());
  for (auto i = 0; i < invocation.results.size(); ++i) {
    auto tensor = MakeUnconstructedAsyncValueRef<RemoteTensor>(
        dist_ctx_->GetHostContext());
    auto metadata = MakeUnconstructedAsyncValueRef<TensorMetadata>(
        dist_ctx_->GetHostContext());
    invocation.results[i] =
        TensorHandle(remote_device_, metadata.CopyRef(), tensor.CopyRef());
    (*results)[i].tensor = tensor.ReleaseRCRef();
    (*results)[i].metadata = std::move(metadata);
    (*results)[i].remote_object_id = std::make_unique<RemoteObjectId>(
        dist_ctx_->GetRemoteObjectManager()->AllocateRemoteObject(
            remote_device_));

    PopulateRemoteExecuteOutputProto(request->add_output(),
                                     *(*results)[i].remote_object_id);
  }

  // Add information about the chains to the request.
  TaskHandle remote_task = remote_device_->GetTaskHandle();
  auto in_chain_id =
      remote_chain_manager_->GetRemoteChain(remote_device_->GetTaskHandle());
  PopulateRemoteObjectIdProto(
      request->mutable_in_chain(),
      remote_chain_manager_->GetRemoteChain(remote_task));
  auto out_chain_id =
      dist_ctx_->GetRemoteObjectManager()->AllocateRemoteObject(remote_device_);
  PopulateRemoteObjectIdProto(request->mutable_out_chain(), out_chain_id);
  remote_chain_manager_->SetRemoteChain(remote_task, out_chain_id);

  // Add op attributes to the request.
  if (Error attr_error =
          PopulateRequestAttrsProto(request.get(), invocation.attrs.freeze())) {
    chain.SetError(attr_error);
    for (auto& output_th : invocation.results) {
      output_th.GetAsyncTensor()->SetError(DecodedDiagnostic(attr_error));
    }
    return;
  }

  RunWhenReady(
      to_wait,
      [dist_ctx = dist_ctx_, arguments = std::move(arguments),
       results = std::move(results), request = std::move(request),
       attrs = invocation.attrs.freeze(), remote_task = std::move(remote_task),
       chain = std::move(chain)]() mutable {
        // Add input object ids to the request.
        for (auto& input : *arguments) {
          // Each TensorHandle should contain an available RemoteTensor.
          // The corresponding tensor on the remote side may be
          // unavailable.
          assert(input->IsAvailable());
          auto* request_input = request->add_input();
          PopulateRemoteObjectIdProto(
              request_input, input->get<RemoteTensor>().remote_object_id());
          TFRT_DLOG(INFO) << "RemoteOpHandler input "
                          << request_input->DebugString();
        }

        auto response = std::make_unique<RemoteExecuteOpResponse>();
        RemoteClientInterface* remote_client =
            dist_ctx->GetRemoteClient(remote_task);
        remote_client->RemoteExecuteOpAsync(
            RemoteCallContext::GetDefault(), request.get(), response.get(),
            [results = std::move(results), request = std::move(request),
             response = std::move(response),
             chain = chain.CopyRef()](Error e) mutable {
              if (e) {
                chain.SetError(std::move(e));
                return;
              }
              if (response->metadata_size() != results->size()) {
                chain.SetError("unexpected number of remote results");
                return;
              }
              for (auto i = 0; i < response->metadata_size(); ++i) {
                Expected<TensorMetadata> metadata =
                    DeserializeTensorMetadata(response->metadata(i));
                if (metadata) {
                  (*results)[i].metadata.emplace(metadata.get());
                  (*results)[i].tensor->emplace<RemoteTensor>(
                      metadata.get(), *(*results)[i].remote_object_id);
                } else {
                  (*results)[i].tensor->SetError(DecodedDiagnostic(
                      "could not deserialize metadata in response"));
                }
              }
              chain.SetStateConcrete();
            });
      });
}

llvm::Expected<OpHandler*> CreateRemoteOpHandler(
    DistributedContext* dist_ctx, RemoteChainManager* remote_chain_manager,
    RCReference<RemoteDevice> remote_device) {
  auto remote_op_handler = std::make_unique<RemoteOpHandler>(
      dist_ctx, remote_chain_manager, std::move(remote_device));
  auto remote_op_handler_ptr = remote_op_handler.get();
  auto core_rt = CoreRuntime::GetFromHostContext(dist_ctx->GetHostContext());
  // TODO(ayushd): flesh out the lifetime and integration with c-api.
  core_rt->TakeOpHandler(std::move(remote_op_handler));
  return remote_op_handler_ptr;
}

}  // namespace tfrt
