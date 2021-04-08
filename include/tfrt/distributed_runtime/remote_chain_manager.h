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

// Remote Chain Manager
//
// This file declares RemoteChainManager. RemoteChainManager manages a single
// chain for every host.

#ifndef TFRT_DISTRIBUTED_RUNTIME_REMOTE_CHAIN_MANAGER_H_
#define TFRT_DISTRIBUTED_RUNTIME_REMOTE_CHAIN_MANAGER_H_

#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/remote_object.h"
#include "tfrt/distributed_runtime/task_handle.h"

namespace tfrt {

class RemoteChainManager {
 public:
  // Populates ready chains for every host.
  explicit RemoteChainManager(DistributedContext* context);

  RemoteObjectId GetRemoteChain(TaskHandle task);
  void SetRemoteChain(TaskHandle task, RemoteObjectId chain);

 private:
  mutex chains_mu_;
  llvm::DenseMap<TaskHandle, RemoteObjectId> chains_
      TFRT_GUARDED_BY(chains_mu_);
};

}  // namespace tfrt
#endif  // TFRT_DISTRIBUTED_RUNTIME_REMOTE_CHAIN_MANAGER_H_
