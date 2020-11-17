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

//===- distributed_context.cc - Distributed Context ------*- C++ -*--------===//
//
// Contains implementation of DistributedContext class.
//
//===----------------------------------------------------------------------===//

#include "tfrt/distributed_runtime/distributed_context.h"

#include "llvm/ADT/DenseMap.h"
#include "tfrt/bef_converter/bef_buffer.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/cluster_info.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/function_cache.h"
#include "tfrt/distributed_runtime/proto/remote_message.pb.h"
#include "tfrt/distributed_runtime/remote_client.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/distributed_runtime/server_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/refcounted_callback.h"

namespace tfrt {

DistributedContext::DistributedContext(
    uint64_t context_id, ServerContext* server_context,
    DistributedContextConfiguration configuration)
    : context_id_(context_id),
      server_context_(server_context),
      dist_config_(configuration),
      cluster_info_(configuration),
      collective_groups_(InitializeCollectiveGroups(configuration)),
      remote_manager_(std::make_unique<RemoteObjectManager>(
          cluster_info_.GetTaskHandle(), GetHostContext())),
      callback_registry_(new CallbackRegistry()),
      function_cache_(new FunctionCache(GetHostContext())) {
  // TODO(bramandia,haoyuzhang): Create remote device manager inside
  // DistributedContext, and add the list of devices from the create context
  // request.
  for (const auto& job_config : dist_config_.cluster_config().jobs()) {
    for (const auto& task_addr : job_config.tasks()) {
      const std::string device_name =
          StrCat("/job:", job_config.name(), "/task:", task_addr.first,
                 "/device:", HostContext::kDefaultHostDeviceName);
      TaskHandle task_handle =
          GetTaskHandle(job_config.name(), task_addr.first);
      server_context_->GetHostContext()->GetDeviceManager()->MaybeAddDevice(
          TakeRef(new RemoteCpuDevice(device_name, task_handle)));
    }
  }

  const std::string device_name =
      StrCat("/job:", dist_config_.job_name(), "/task:", dist_config_.task_id(),
             "/device:", HostContext::kDefaultHostDeviceName);
  RCReference<Device> cpu_device =
      GetHostContext()->GetDeviceManager()->GetDeviceRef<Device>(device_name);
  assert(cpu_device);
  if (cpu_device.get() != nullptr) {
    local_ready_chain_ = std::make_unique<RemoteObjectId>(
        remote_manager_->AllocateRemoteObject(cpu_device.CopyRef()));
    remote_manager_->SetRemoteObject(
        *local_ready_chain_, GetReadyChain(GetHostContext()).CopyRCRef());
  }
}

DistributedContext::~DistributedContext() {}

llvm::StringMap<CollectiveGroup> DistributedContext::InitializeCollectiveGroups(
    const DistributedContextConfiguration& config) {
  llvm::StringMap<CollectiveGroup> collective_groups;
  for (const auto& group_config : config.collective_groups()) {
    llvm::SmallVector<TaskHandle, 8> members;
    members.reserve(group_config.members_size());
    for (const auto& task : group_config.members()) {
      members.push_back(cluster_info_.GetTaskHandle(task).get());
    }
    collective_groups.try_emplace(
        group_config.name(), CollectiveGroup{group_config.name(), members});
  }
  return collective_groups;
}

const CollectiveGroup& DistributedContext::GetCollectiveGroup(
    string_view name) const {
  const auto& it = collective_groups_.find(name);
  assert(it != collective_groups_.end() && "Failed to find collective group.");
  return it->second;
}

RemoteClientInterface* DistributedContext::GetRemoteClient(
    TaskHandle task_handle) {
  mutex_lock l(remote_clients_mu_);
  auto it = remote_clients_.find(task_handle);
  if (it == remote_clients_.end()) {
    auto* communicator = server_context_->GetOrCreateFabricCommunicator();
    auto ret = remote_clients_.try_emplace(
        task_handle, communicator->CreateRemoteClient(this, task_handle));
    assert(ret.second && "Failed to create remote client.");
    it = ret.first;
  }
  return it->second.get();
}

void DistributedContext::CreateRemoteContexts(
    DistributedContext::CallbackFn done_callback) {
  // Reference-counted done callback is invoked after all remote calls finish
  auto rc_done = TakeRef(new RefCountedCallback(std::move(done_callback)));

  // Base request contains information that is the shared by all the requests
  // sent to different tasks, including the cluster configuration and collective
  // groups of distributed context configuration.
  // Individual requests can directly set to use the allocated fields of the
  // base one without memory copies. The base request must be alive until all
  // uses of individual requests have finished.
  auto base_request = std::make_shared<CreateContextRequest>();
  *base_request->mutable_dist_config()->mutable_cluster_config() =
      dist_config_.cluster_config();
  *base_request->mutable_dist_config()->mutable_collective_groups() =
      dist_config_.collective_groups();
  for (const auto& job_config : dist_config_.cluster_config().jobs()) {
    for (const auto& task : job_config.tasks()) {
      if (job_config.name() == dist_config_.job_name() &&
          task.first == dist_config_.task_id()) {
        continue;
      }

      auto request = std::make_unique<CreateContextRequest>();
      request->set_context_id(context_id_);
      auto* request_dist_config = request->mutable_dist_config();
      request_dist_config->set_job_name(job_config.name());
      request_dist_config->set_task_id(task.first);
      request_dist_config->set_allocated_cluster_config(
          base_request->mutable_dist_config()->mutable_cluster_config());
      for (auto& cg :
           *base_request->mutable_dist_config()->mutable_collective_groups()) {
        request_dist_config->mutable_collective_groups()->AddAllocated(&cg);
      }

      TaskHandle task_handle = GetTaskHandle(job_config.name(), task.first);
      RemoteClientInterface* client = GetRemoteClient(task_handle);
      auto response = std::make_unique<CreateContextResponse>();
      client->CreateContextAsync(
          request.get(), response.get(),
          [this, task_handle, base_request, request = std::move(request),
           response = std::move(response),
           rc_done = rc_done.CopyRef()](Error e) mutable {
            if (e) {
              rc_done->UpdateState(std::move(e));
            } else {
              rc_done->UpdateState(
                  AddReadyChain(task_handle, response->ready_chain()));
            }
            // NOTE: `base_request` is the owner of `cluster_config` and
            // `collective_groups`. Release these fields from `request` so that
            // these fields are not destructed multiple times.
            auto request_dist_config = request->mutable_dist_config();
            request_dist_config->release_cluster_config();
            for (int i = 0; i < request_dist_config->collective_groups_size();
                 i++) {
              request_dist_config->mutable_collective_groups()->ReleaseLast();
            }
          });
    }
  }
}

Error DistributedContext::AddReadyChain(TaskHandle task_handle,
                                        const RemoteObjectIdProto& chain) {
  RCReference<Device> device =
      GetHostContext()->GetDeviceManager()->GetDeviceRef<Device>(
          chain.device());
  if (device.get() == nullptr) {
    return llvm::make_error<DeviceNotFoundErrorInfo>(
        StrCat("Can't find device: ", chain.device()));
  }
  RemoteObjectId ready_chain(chain.prefix_id(), chain.local_id(),
                             device.CopyRef());
  mutex_lock l(ready_chains_mu_);
  ready_chains_.insert({task_handle, ready_chain});
  return Error::success();
}

void DistributedContext::CloseRemoteContexts(
    DistributedContext::CallbackFn done_callback) {
  // Reference-counted done callback is invoked after all remote calls finish
  auto rc_done = TakeRef(new RefCountedCallback(std::move(done_callback)));

  auto request = std::make_shared<CloseContextRequest>();
  request->set_context_id(context_id_);
  for (const auto& job_config : dist_config_.cluster_config().jobs()) {
    for (const auto& task : job_config.tasks()) {
      if (job_config.name() == dist_config_.job_name() &&
          task.first == dist_config_.task_id()) {
        continue;
      }
      TaskHandle task_handle = GetTaskHandle(job_config.name(), task.first);
      RemoteClientInterface* client = GetRemoteClient(task_handle);
      auto response = std::make_shared<CloseContextResponse>();
      client->CloseContextAsync(
          request.get(), response.get(),
          [request, response, rc_done = rc_done.CopyRef()](Error e) mutable {
            rc_done->UpdateState(std::move(e));
          });
    }
  }
}

llvm::DenseMap<TaskHandle, RemoteObjectId>
DistributedContext::RemoteReadyChains() {
  mutex_lock l(ready_chains_mu_);
  return ready_chains_;
}

}  // namespace tfrt
