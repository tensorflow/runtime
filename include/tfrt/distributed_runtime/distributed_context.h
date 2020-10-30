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

//===- distributed_context.h - Distributed Context --------------*- C++ -*-===//
//
// Declares DistributedContext, which represents the server-side state (tasks,
// remote objects, function registry, etc.) associated with a distributed
// execution environment.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DISTRIBUTED_RUNTIME_DISTRIBUTED_CONTEXT_H_
#define TFRT_DISTRIBUTED_RUNTIME_DISTRIBUTED_CONTEXT_H_

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "tfrt/distributed_runtime/function_cache.h"
#include "tfrt/distributed_runtime/remote_client.h"
#include "tfrt/distributed_runtime/remote_device.h"
#include "tfrt/distributed_runtime/server_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {

class CallbackRegistry;
class RemoteObjectManager;
class RemoteClientInterface;
class FunctionCache;

struct ClusterConfiguration {
  struct NameAddressPair {
    NameAddressPair(const std::string& name, const std::string& address)
        : name(name), address(address) {}

    // The name associated with this address.
    // This is something like: "/job:worker/task:1"
    std::string name;
    // The address that can be used by FabricCommunicator.
    // For instance, this can be hostname:port.
    std::string address;
  };
  // Ordered list of all addresses in the cluster.
  llvm::SmallVector<NameAddressPair, 8> addresses;
  // Id of this host.  Address of this host is `addresses[id]`.
  HostId id;
};

struct CollectiveGroup {
  std::string name;  // unique identifier for this group
  llvm::SmallVector<HostId, 8> members;
};

// Configurations at the client side and can be propagated through network.
struct DistributedContextConfiguration {
  ClusterConfiguration cluster_config;
  llvm::SmallVector<CollectiveGroup, 4> collective_groups;
};

// DistributedContext owns a collection of server-side state related to
// distributed execution. It is created from a ServerContext and is uniquely
// identified by its context id.
// The state it owns include:
//   * Callback registry
//   * Function cache
//   * Clients for communicating with other peers in the cluster
class DistributedContext {
 public:
  DistributedContext(uint64_t context_id, ServerContext* server,
                     DistributedContextConfiguration configuration);
  ~DistributedContext();

  DistributedContext(DistributedContext&&) = delete;
  DistributedContext& operator=(DistributedContext&&) = delete;

  DistributedContext(const DistributedContext&) = delete;
  DistributedContext& operator=(const DistributedContext&) = delete;

  HostContext* GetHostContext() { return server_context_->GetHostContext(); }

  ServerContext* GetServerContext() { return server_context_; }

  uint64_t GetContextId() const { return context_id_; }

  HostId GetHostId() const { return configuration_.cluster_config.id; }

  string_view GetTaskName() const {
    HostId my_id = configuration_.cluster_config.id;
    return configuration_.cluster_config.addresses[my_id].name;
  }

  string_view GetTaskName(HostId id) const {
    return configuration_.cluster_config.addresses[id].name;
  }

  string_view GetRemoteAddress(HostId id) const {
    return configuration_.cluster_config.addresses[id].address;
  }

  CollectiveGroup GetCollectiveGroup(llvm::StringRef name) const;

  CallbackRegistry* GetCallbackRegistry() const {
    return callback_registry_.get();
  }

  RemoteObjectManager* GetRemoteObjectManager() const {
    return remote_manager_.get();
  }

  FunctionCache* GetFunctionCache() const { return function_cache_.get(); }

  RemoteClientInterface* GetRemoteClient(HostId id);

 private:
  void InitializeRemoteDevices();

  const uint64_t context_id_;
  ServerContext* const server_context_;
  const DistributedContextConfiguration configuration_;

  mutex remote_clients_mu_;
  llvm::DenseMap<HostId, std::unique_ptr<RemoteClientInterface>> remote_clients_
      TFRT_GUARDED_BY(remote_clients_mu_);

  std::unique_ptr<RemoteObjectManager> remote_manager_;
  std::unique_ptr<CallbackRegistry> callback_registry_;

  std::unique_ptr<FunctionCache> function_cache_;
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_DISTRIBUTED_CONTEXT_H_
