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

// Server Context
//
// This file contains the implementation of server context.

#include "tfrt/distributed_runtime/server_context.h"

#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/distributed_init_helper.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/distributed_runtime/request_handler.h"
#include "tfrt/distributed_runtime/request_handler_impl.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/random_util.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

ServerContext::ServerContext(HostContext* host_context,
                             ServerContextConfiguration configuration)
    : host_context_{host_context},
      configuration_{std::move(configuration)},
      init_helper_{std::make_unique<DistributedInitHelper>(this)} {
  request_handler_ = NewRequestHandler(this);
  GetOrCreateFabricCommunicator();
  GarbageCollectInactiveDistributedContexts(
      configuration_.context_gc_timeout_secs);
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

Expected<DistributedContext*> ServerContext::CreateDistributedContext(
    uint64_t context_id, DistributedContextConfiguration configuration) {
  AsyncValueRef<DistributedContext> dist_context =
      MakeAvailableAsyncValueRef<DistributedContext>(
          host_context_, context_id, this, std::move(configuration));
  DistributedContext* context = &dist_context.get();

  mutex_lock l(context_mu_);
  bool inserted =
      dist_contexts_.try_emplace(context_id, std::move(dist_context)).second;
  if (!inserted) {
    return llvm::make_error<UnknownErrorInfo>(
        StrCat("Failed to create DistributedContext: context ID ", context_id,
               " already exists!"));
  }
  return context;
}

Error ServerContext::CloseDistributedContext(uint64_t context_id) {
  mutex_lock l(context_mu_);
  if (!dist_contexts_.erase(context_id)) {
    return llvm::make_error<InvalidDistributedContextIdErrorInfo>(
        StrCat("Context with id ", context_id, " does not exist."));
  }
  context_last_access_times_.erase(context_id);
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
  RecordContextAccess(context_id);
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
  RecordContextAccess(context_id);
  return it->second.CopyRef();
}

Error ServerContext::TrackContextAccessTime(uint64_t context_id) {
  mutex_lock l(context_mu_);
  if (dist_contexts_.find(context_id) == dist_contexts_.end()) {
    return llvm::make_error<InvalidDistributedContextIdErrorInfo>(
        StrCat("Context with id ", context_id, " does not exist."));
  }
  context_last_access_times_.try_emplace(context_id,
                                         std::chrono::system_clock::now());
  return Error::success();
}

void ServerContext::RecordContextAccess(uint64_t context_id) const {
  auto it = context_last_access_times_.find(context_id);
  if (it != context_last_access_times_.end()) {
    it->second = std::chrono::system_clock::now();
  }
}

void ServerContext::GarbageCollectInactiveDistributedContexts(int delay_secs) {
  auto gc_fn = [this, delay_secs]() {
    llvm::SmallVector<AsyncValueRef<DistributedContext>, 4> gc_contexts;
    auto inactive_time_point =
        std::chrono::system_clock::now() -
        std::chrono::seconds(configuration_.context_gc_timeout_secs);
    {
      mutex_lock l(context_mu_);
      for (auto it = context_last_access_times_.begin();
           it != context_last_access_times_.end(); it++) {
        if (it->second < inactive_time_point) {
          auto ctx_it = dist_contexts_.find(it->first);
          if (ctx_it != dist_contexts_.end()) {
            TFRT_LOG(WARNING) << "Distributed context with id " << it->first
                              << " is garbage collected due to inactivity.";
            // Temporarily move the context ref into a separate list to defer
            // the distruction of DistributedContext after releasing the mutex.
            gc_contexts.emplace_back(std::move(ctx_it->second));
            dist_contexts_.erase(ctx_it);
          }
          context_last_access_times_.erase(it);
        }
      }
    }
    gc_contexts.clear();
    GarbageCollectInactiveDistributedContexts(delay_secs);
  };
  mutex_lock l(context_gc_timer_mu_);
  context_gc_timer_ = host_context_->GetTimerQueue()->ScheduleTimer(
      std::chrono::seconds(delay_secs), gc_fn);
}

void ServerContext::ShutDown() {
  {
    mutex_lock l(context_gc_timer_mu_);
    if (context_gc_timer_.get() != nullptr) {
      host_context_->GetTimerQueue()->CancelTimer(context_gc_timer_);
    }
  }
  {
    mutex_lock l(context_mu_);
    dist_contexts_.clear();
    context_last_access_times_.clear();
  }
  {
    mutex_lock l(communicator_mutex_);
    fabric_communicator_.reset();
  }
}

}  // namespace tfrt
