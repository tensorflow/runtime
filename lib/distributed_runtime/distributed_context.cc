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
#include "tfrt/distributed_runtime/remote_client.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/distributed_runtime/server_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

DistributedContext::DistributedContext(
    uint64_t context_id, ServerContext* server_context,
    DistributedContextConfiguration configuration)
    : context_id_(context_id),
      server_context_(server_context),
      cluster_info_(configuration.cluster_config),
      collective_groups_(InitializeCollectiveGroups(configuration)),
      remote_manager_(std::make_unique<RemoteObjectManager>(
          cluster_info_.GetTaskHandle(), server_context_->GetHostContext())),
      callback_registry_(new CallbackRegistry()),
      function_cache_(new FunctionCache(server_context->GetHostContext())) {
  InitializeRemoteDevices(configuration);
}

DistributedContext::~DistributedContext() {}

llvm::StringMap<CollectiveGroup> DistributedContext::InitializeCollectiveGroups(
    const DistributedContextConfiguration& config) {
  llvm::StringMap<CollectiveGroup> collective_groups;
  for (const auto& group_config : config.collective_groups) {
    llvm::SmallVector<TaskHandle, 8> members;
    members.reserve(group_config.members.size());
    for (const auto& task : group_config.members) {
      members.push_back(cluster_info_.GetTaskHandle(task).get());
    }
    collective_groups.try_emplace(group_config.name,
                                  CollectiveGroup{group_config.name, members});
  }
  return collective_groups;
}

// TODO(bramandia,haoyuzhang): Create remote device manager inside
// DistributedContext, and add the list of devices from the create context
// request.
void DistributedContext::InitializeRemoteDevices(
    const DistributedContextConfiguration& config) {
  for (const auto& pair : config.cluster_config.task_addresses) {
    const std::string device_name =
        StrCat(pair.first(), "/device:", HostContext::kDefaultHostDeviceName);
    TaskHandle task_handle = GetTaskHandle(pair.first());
    server_context_->GetHostContext()->GetDeviceManager()->MaybeAddDevice(
        TakeRef(new RemoteCpuDevice(device_name, task_handle)));
  }
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

}  // namespace tfrt
