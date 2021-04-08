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

// Distributed Init Helper
//
// This file contains the implementation of DistributedInitHelper.

#include "tfrt/distributed_runtime/distributed_init_helper.h"

#include "google/protobuf/util/message_differencer.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/proto/cluster_config.pb.h"
#include "tfrt/distributed_runtime/server_context.h"
#include "tfrt/distributed_runtime/task_name_util.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/random_util.h"

namespace tfrt {

namespace {
bool IsMultiClientLead(DistributedContextConfiguration& configuration) {
  std::string lead_job;
  int lead_task;
  DieIfError(TaskNameUtil::ParseTaskName(
      configuration.cluster_config().lead_task_name(), &lead_job, &lead_task));

  return configuration.job_name() == lead_job &&
         configuration.task_id() == lead_task;
}
}  // namespace

void DistributedInitHelper::InitializeSingleClientDistributedContext(
    DistributedContextConfiguration configuration,
    llvm::unique_function<void(Expected<DistributedContext*>)> done) const {
  uint64_t context_id = random::New64();
  // Create distributed context on current task.
  auto expected = server_context_->CreateDistributedContext(
      context_id, std::move(configuration));
  if (!expected) {
    done(expected.takeError());
    return;
  }
  DistributedContext* dist_context = expected.get();

  // Create distributed contexts on remote tasks in the cluster.
  dist_context->GetRemoteDevices(
      [dist_context, done = std::move(done)](Error e) mutable {
        if (e) {
          done(std::move(e));
          return;
        }
        dist_context->CreateRemoteContexts(
            DistributedContext::RemoteInitMode::SINGLE_CLIENT,
            [dist_context, done = std::move(done)](Error e) mutable {
              if (e) {
                done(std::move(e));
              } else {
                done(dist_context);
              }
            });
      });
}

void DistributedInitHelper::InitializeMultiClientDistributedContext(
    DistributedContextConfiguration configuration,
    llvm::unique_function<void(Expected<DistributedContext*>)> done) {
  if (IsMultiClientLead(configuration)) {
    uint64_t context_id = random::New64();
    auto expected = server_context_->CreateDistributedContext(
        context_id, std::move(configuration));
    DistributedContext* context = expected.get();
    context->GetRemoteDevices(
        [context, done = std::move(done)](Error e) mutable {
          if (e) {
            done(std::move(e));
            return;
          }
          context->CreateRemoteContexts(
              DistributedContext::RemoteInitMode::MULTI_CLIENT,
              [context, done = std::move(done)](Error e) mutable {
                if (e) {
                  done(std::move(e));
                  return;
                }
                context->BroadcastRemoteReadyChains(
                    [context, done = std::move(done)](Error e) mutable {
                      if (e) {
                        done(std::move(e));
                        return;
                      }
                      done(context);
                    });
              });
        });
  } else {
    mutex_lock l(mu_);
    if (state_ != State::NOT_READY) {
      done(llvm::make_error<UnknownErrorInfo>(
          StrCat("Trying to initialize multi-client distributed context "
                 "while a previous initialization attempt has started.")));
      return;
    }
    local_config_ =
        std::make_unique<DistributedContextConfiguration>(configuration);
    local_cb_ = std::move(done);
    state_ = State::WAIT_FOR_CONTEXT;

    // The CreateContext RPC from leader comes before invoking local init call.
    if (remote_cb_) {
      if (Error e = remote_cb_()) {
        local_cb_(std::move(e));
        state_ = State::ERROR;
      }
      state_ = State::WAIT_FOR_CHAINS;
    }
  }
}

bool DistributedInitHelper::IsConfigCompatible(
    const DistributedContextConfiguration& config) const {
  return ::google::protobuf::util::MessageDifferencer::Equals(config,
                                                              *local_config_);
}

void DistributedInitHelper::RegisterRemoteCallback(
    llvm::unique_function<Error()> remote_cb) {
  mutex_lock l(mu_);
  assert(state_ == State::NOT_READY || state_ == State::WAIT_FOR_CONTEXT);
  remote_cb_ = std::move(remote_cb);
  if (state_ == State::WAIT_FOR_CONTEXT) {
    if (Error e = remote_cb_()) {
      local_cb_(std::move(e));
      state_ = State::ERROR;
    }
    state_ = State::WAIT_FOR_CHAINS;
  }
}

void DistributedInitHelper::Complete(Expected<DistributedContext*> expected) {
  mutex_lock l(mu_);
  if (state_ == State::WAIT_FOR_CHAINS) {
    if (local_cb_) {
      local_cb_(std::move(expected));
    }
    state_ = State::FINISHED;
  }
}
}  // namespace tfrt
