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
#include "llvm/ADT/StringMap.h"
#include "tfrt/distributed_runtime/cluster_info.h"
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
  // Map from task name (e.g., "/job:worker/task:1") to network address
  // (e.g., "hostname:port") for all tasks in cluster.
  llvm::StringMap<std::string> task_addresses;

  // Self task name.
  std::string task_name;
};

struct CollectiveGroupConfiguration {
  // Unique identifier for this group.
  std::string name;
  // List of group members with full task names, e.g., "/job:worker/task:1"
  llvm::SmallVector<std::string, 8> members;
};

// Configurations at the client side and can be propagated through network.
struct DistributedContextConfiguration {
  ClusterConfiguration cluster_config;
  llvm::SmallVector<CollectiveGroupConfiguration, 4> collective_groups;
};

// Collective group membership stored inside DistributedContext. Different from
// the CollectiveGroupConfiguration, the members are represented by TaskHandles
// which are only meaningful within the host and should be serialized directly.
struct CollectiveGroup {
  // Unique identifier for this group.
  std::string name;
  // List of group members.
  llvm::SmallVector<TaskHandle, 8> members;
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

  TaskHandle GetTaskHandle() const { return cluster_info_.GetTaskHandle(); }

  TaskHandle GetTaskHandle(string_view task_name) const {
    return cluster_info_.GetTaskHandle(task_name).get();
  }

  TaskHandle GetTaskHandle(string_view job, int task_id) const {
    return cluster_info_.GetTaskHandle(job, task_id).get();
  }

  string_view GetTaskName() const {
    return cluster_info_.GetTaskName(cluster_info_.GetTaskHandle()).get();
  }

  string_view GetTaskName(TaskHandle task_handle) const {
    return cluster_info_.GetTaskName(task_handle).get();
  }

  string_view GetRemoteAddress(TaskHandle task_handle) const {
    return cluster_info_.GetTaskAddress(task_handle).get();
  }

  const CollectiveGroup& GetCollectiveGroup(string_view name) const;

  CallbackRegistry* GetCallbackRegistry() const {
    return callback_registry_.get();
  }

  RemoteObjectManager* GetRemoteObjectManager() const {
    return remote_manager_.get();
  }

  FunctionCache* GetFunctionCache() const { return function_cache_.get(); }

  RemoteClientInterface* GetRemoteClient(TaskHandle task_handle);

 private:
  llvm::StringMap<CollectiveGroup> InitializeCollectiveGroups(
      const DistributedContextConfiguration&);
  void InitializeRemoteDevices(const DistributedContextConfiguration&);

  const uint64_t context_id_;
  ServerContext* const server_context_;

  const ClusterInfo cluster_info_;
  // Map from collective group name to the group members
  const llvm::StringMap<CollectiveGroup> collective_groups_;

  mutex remote_clients_mu_;
  llvm::DenseMap<TaskHandle, std::unique_ptr<RemoteClientInterface>>
      remote_clients_ TFRT_GUARDED_BY(remote_clients_mu_);

  std::unique_ptr<RemoteObjectManager> remote_manager_;
  std::unique_ptr<CallbackRegistry> callback_registry_;

  std::unique_ptr<FunctionCache> function_cache_;
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_DISTRIBUTED_CONTEXT_H_
