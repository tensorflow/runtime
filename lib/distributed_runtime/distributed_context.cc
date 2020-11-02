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
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/function_cache.h"
#include "tfrt/distributed_runtime/remote_client.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/distributed_runtime/server_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

CollectiveGroup DistributedContext::GetCollectiveGroup(
    llvm::StringRef name) const {
  CollectiveGroup collective_group;
  bool found = false;
  for (const auto& registered_group : configuration_.collective_groups) {
    if (registered_group.name == name) {
      collective_group = registered_group;
      found = true;
    }
  }
  if (!found) {
    TFRT_LOG(WARNING) << "Did not find collective group";
  }
  return collective_group;
}

DistributedContext::DistributedContext(
    uint64_t context_id, ServerContext* server_context,
    DistributedContextConfiguration configuration)
    : context_id_(context_id),
      server_context_(server_context),
      configuration_(std::move(configuration)),
      remote_manager_(std::make_unique<RemoteObjectManager>(
          configuration.cluster_config.id, server_context_->GetHostContext())),
      callback_registry_(new CallbackRegistry()),
      function_cache_(new FunctionCache(server_context->GetHostContext())) {
  InitializeRemoteDevices();
}

DistributedContext::~DistributedContext() {}

// TODO(bramandia,haoyuzhang): Create remote device manager inside
// DistributedContext, and add the list of devices from the create context
// request.
void DistributedContext::InitializeRemoteDevices() {
  for (HostId host_id = 0;
       host_id < configuration_.cluster_config.addresses.size(); ++host_id) {
    const auto& address = configuration_.cluster_config.addresses[host_id];
    const std::string device_name =
        StrCat(address.name, "/device:", HostContext::kDefaultHostDeviceName);
    server_context_->GetHostContext()->GetDeviceManager()->MaybeAddDevice(
        TakeRef(new RemoteCpuDevice(device_name, host_id)));
  }
}

RemoteClientInterface* DistributedContext::GetRemoteClient(HostId id) {
  mutex_lock l(remote_clients_mu_);
  auto it = remote_clients_.find(id);
  if (it == remote_clients_.end()) {
    auto* communicator = server_context_->GetOrCreateFabricCommunicator();
    auto ret = remote_clients_.try_emplace(
        id, communicator->CreateRemoteClient(this, id));
    assert(ret.second && "Failed to create remote client.");
    it = ret.first;
  }
  return it->second.get();
}

}  // namespace tfrt
