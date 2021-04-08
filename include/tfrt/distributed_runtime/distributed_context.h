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

// Distributed Context
//
// Declares DistributedContext, which represents the server-side state (tasks,
// remote objects, function registry, etc.) associated with a distributed
// execution environment.

#ifndef TFRT_DISTRIBUTED_RUNTIME_DISTRIBUTED_CONTEXT_H_
#define TFRT_DISTRIBUTED_RUNTIME_DISTRIBUTED_CONTEXT_H_

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "tfrt/distributed_runtime/cluster_info.h"
#include "tfrt/distributed_runtime/function_cache.h"
#include "tfrt/distributed_runtime/proto/cluster_config.pb.h"
#include "tfrt/distributed_runtime/remote_client.h"
#include "tfrt/distributed_runtime/remote_device.h"
#include "tfrt/distributed_runtime/remote_object.h"
#include "tfrt/distributed_runtime/server_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {

class CallbackRegistry;
class RemoteObjectManager;
class RemoteClientInterface;
class FunctionCache;

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
  enum RemoteInitMode { SINGLE_CLIENT, MULTI_CLIENT };

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

  DeviceManager* GetRemoteDeviceManager() { return &cluster_device_mgr_; }

  FunctionCache* GetFunctionCache() const { return function_cache_.get(); }

  RemoteClientInterface* GetRemoteClient(TaskHandle task_handle);

  using CallbackFn = llvm::unique_function<void(Error)>;

  // Get device information on remote tasks.
  void GetRemoteDevices(CallbackFn done_callback);

  // Create contexts on remote tasks. The callback will be invoked after all
  // remote calls finish. This method should be invoked on the single client,
  // or the lead task in multi-client cluster.
  void CreateRemoteContexts(RemoteInitMode mode, CallbackFn done_callback);

  // Broadcast remote chains collected from all tasks. The callback will be
  // invoked after all remote calls finish.
  void BroadcastRemoteReadyChains(CallbackFn done_callback);

  // Close contexts on remote tasks. The callback will be invoked after all
  // remote calls finish.
  void CloseRemoteContexts(CallbackFn done_callback);

  RemoteObjectId LocalReadyChain() {
    assert(local_ready_chain_ != nullptr);
    return *local_ready_chain_;
  }
  llvm::DenseMap<TaskHandle, RemoteObjectId> RemoteReadyChains();

  Error AddReadyChain(TaskHandle task_handle, const RemoteObjectIdProto& chain);

 private:
  llvm::StringMap<CollectiveGroup> InitializeCollectiveGroups(
      const DistributedContextConfiguration&);

  // Periodically schedule functions to send KeepAlive messages to remote
  // distributed contexts created by `CreateRemoteContexts`.
  void SendKeepAlive(int delay_secs);

  const uint64_t context_id_;
  ServerContext* const server_context_;

  const DistributedContextConfiguration dist_config_;
  const ClusterInfo cluster_info_;
  // Map from collective group name to the group members
  const llvm::StringMap<CollectiveGroup> collective_groups_;

  mutex remote_clients_mu_;
  llvm::DenseMap<TaskHandle, std::unique_ptr<RemoteClientInterface>>
      remote_clients_ TFRT_GUARDED_BY(remote_clients_mu_);

  // Cluster device manager contains RemoteDevice instances for devices in all
  // tasks of this cluster. Note that for every local device, there is a
  // corresponding RemoteDevice instance in this manager as well.
  // The cluster device manager should be kept symmetric across the distributed
  // context in all tasks of the cluster.
  DeviceManager cluster_device_mgr_;

  std::unique_ptr<RemoteObjectManager> remote_manager_;
  std::unique_ptr<CallbackRegistry> callback_registry_;

  std::unique_ptr<FunctionCache> function_cache_;

  std::unique_ptr<RemoteObjectId> local_ready_chain_;
  mutex ready_chains_mu_;
  llvm::DenseMap<TaskHandle, RemoteObjectId> ready_chains_
      TFRT_GUARDED_BY(ready_chains_mu_);

  mutex keep_alive_mu_;
  TimerQueue::TimerHandle keep_alive_timer_ TFRT_GUARDED_BY(keep_alive_mu_);
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_DISTRIBUTED_CONTEXT_H_
