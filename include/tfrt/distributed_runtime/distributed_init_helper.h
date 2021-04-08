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
// Declares DistributedInitHelper, which drives the process of initializing
// single- and multi-client distributed contexts.

#ifndef TFRT_DISTRIBUTED_RUNTIME_MULTI_CLIENT_INIT_HELPER_H_
#define TFRT_DISTRIBUTED_RUNTIME_MULTI_CLIENT_INIT_HELPER_H_

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Error.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
class ServerContext;

// Helper class for distributed initialization.
class DistributedInitHelper {
 public:
  explicit DistributedInitHelper(ServerContext* server_context)
      : server_context_(server_context) {}

  // Initialize single-client distributed context.
  // Should be invoked on the master of a cluster.
  void InitializeSingleClientDistributedContext(
      DistributedContextConfiguration configuration,
      llvm::unique_function<void(Expected<DistributedContext*>)> done) const;

  // Initialize multi-client distributed context.
  // Must be called on all clients in the cluster.
  //
  // If current task is the leader of multi-client cluster, creates distributed
  // contexts on local and remote tasks in the cluster with configuration
  // compatibility checks, and propagate device and ready chain information
  // across the cluster. The done callback is invoked after finishing all above.
  // This process typically involves 3 rounds of RPCs, with the first two rounds
  // almost the same as in single-client initialization.
  // 1. The leader task collect remote device info from all other tasks;
  // 2. The leader task creates distributed contexts on all other tasks, and
  //    gets back remote ready chains from them.
  // 3. The leader task broadcasts all remote chains to other tasks.
  //
  // If current task is not the leader, wait for the leader to create
  // distributed context and populate the device and ready chain information.
  // The done callback is invoked after the context is ready to be used.
  void InitializeMultiClientDistributedContext(
      DistributedContextConfiguration configuration,
      llvm::unique_function<void(Expected<DistributedContext*>)> done);

  // Return true if the given config is the same as local_config_.
  bool IsConfigCompatible(const DistributedContextConfiguration& config) const;

  // Return distributed configuration specified in the local init call.
  const DistributedContextConfiguration& GetLocalConfig() const {
    return *local_config_;
  }

  // For a non-leader client task, register remote callback function.
  // If the local init is in READY state, invoke the remote_cb function to
  // create distributed context directly.
  void RegisterRemoteCallback(llvm::unique_function<Error()> remote_cb);

  // For a non-leader client task, finish distributed context initialization.
  // This is usually invoked after the last step of multi-client init (i.e.,
  // receiving all remote ready chains from the leader).
  void Complete(Expected<DistributedContext*> expected);

 private:
  ServerContext* const server_context_;

  // State for non-leader tasks in multi-client initialization.
  // 1. starts with NOT_READY
  // 2. invoke InitializeMultiClientdistributedContext locally: WAIT_FOR_CONTEXT
  // 3. process remote call CreateContext from leader: WAIT_FOR_CHAINS
  // 4. process remote call SendReadyChains from leader: FINISHED
  // 5. if previous steps go wrong: ERROR
  enum class State {
    NOT_READY,
    WAIT_FOR_CONTEXT,
    WAIT_FOR_CHAINS,
    FINISHED,
    ERROR
  };
  mutable mutex mu_;
  State state_ TFRT_GUARDED_BY(mu_) = State::NOT_READY;

  // Locally specified distributed config for non-leader task. Hold a copy here
  // for compatibility check.
  std::unique_ptr<DistributedContextConfiguration> local_config_;
  // Callbacks from local InitializeMultiClientDistributedContext call.
  llvm::unique_function<void(Expected<DistributedContext*>)> local_cb_;
  // Callbacks from CreateContext RPC from leader.
  llvm::unique_function<Error()> remote_cb_;
};
}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_MULTI_CLIENT_INIT_HELPER_H_
