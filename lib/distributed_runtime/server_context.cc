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

//===- server_context.cc - Server Context -----------------------*- C++ -*-===//
//
// This file contains the implementation of server context.
//
//===----------------------------------------------------------------------===//

#include "tfrt/distributed_runtime/server_context.h"

#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/distributed_runtime/request_handler.h"
#include "tfrt/distributed_runtime/request_handler_impl.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

ServerContext::ServerContext(HostContext* host_context,
                             ServerContextConfiguration configuration)
    : host_context_{host_context}, configuration_{std::move(configuration)} {
  request_handler_ = NewRequestHandler(this);
  GetOrCreateFabricCommunicator();
}

ServerContext::~ServerContext() { ShutDown(); }

FabricCommunicator* ServerContext::GetOrCreateFabricCommunicator() {
  mutex_lock lock(communicator_mutex_);
  return GetOrCreateFabricCommunicatorUnsafe();
}

FabricCommunicator* ServerContext::GetOrCreateFabricCommunicatorUnsafe() {
  // Don't create a new communicator if cached
  if (fabric_communicator_ != nullptr) {
    return fabric_communicator_.get();
  }

  const auto& communicator_configuration = configuration_.fabric_configuration;
  const auto& communicator_type = communicator_configuration.type;
  // Create FabricCommunicator
  fabric_communicator_.reset(
      FabricCommunicator::CreateFabricCommunicator(communicator_type, this));
  return fabric_communicator_.get();
}

void ServerContext::ResetRequestHandler(
    std::unique_ptr<RequestHandlerInterface> request_handler) {
  {
    mutex_lock l(communicator_mutex_);
    fabric_communicator_.reset();
    request_handler_ = std::move(request_handler);
  }
  GetOrCreateFabricCommunicator();
}

Error ServerContext::CreateDistributedContext(
    uint64_t context_id, DistributedContextConfiguration configuration) {
  AsyncValueRef<DistributedContext> dist_context =
      MakeAvailableAsyncValueRef<DistributedContext>(
          host_context_, context_id, this, std::move(configuration));
  mutex_lock l(context_mu_);
  bool inserted =
      dist_contexts_.try_emplace(context_id, std::move(dist_context)).second;
  if (!inserted) {
    return llvm::make_error<UnknownErrorInfo>(
        StrCat("Failed to create DistributedContext: context ID ", context_id,
               " already exists!"));
  }
  return Error::success();
}

Expected<DistributedContext*> ServerContext::GetDistributedContext(
    uint64_t context_id) const {
  mutex_lock l(context_mu_);
  auto it = dist_contexts_.find(context_id);
  if (it == dist_contexts_.end()) {
    return llvm::make_error<InvalidDistributedContextIdErrorInfo>(
        StrCat("Context with id ", context_id, " does not exist."));
  }
  return &it->second.get();
}

AsyncValueRef<DistributedContext>
ServerContext::GetDistributedContextAsyncValue(uint64_t context_id) const {
  mutex_lock l(context_mu_);
  auto it = dist_contexts_.find(context_id);
  if (it == dist_contexts_.end()) {
    return MakeErrorAsyncValueRef(
        host_context_,
        StrCat("Context with id ", context_id, " does not exist."));
  }
  return it->second.CopyRef();
}

void ServerContext::ShutDown() {
  {
    mutex_lock l(context_mu_);
    dist_contexts_.clear();
  }
  {
    mutex_lock l(communicator_mutex_);
    fabric_communicator_.reset();
  }
}

}  // namespace tfrt
