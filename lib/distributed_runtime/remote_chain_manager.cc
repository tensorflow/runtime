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

// RemoteChainManager
//
// This file contains an implementation of RemoteChainManager.

#include "tfrt/distributed_runtime/remote_chain_manager.h"

namespace tfrt {

RemoteChainManager::RemoteChainManager(DistributedContext* context)
    : chains_(context->RemoteReadyChains()) {}

RemoteObjectId RemoteChainManager::GetRemoteChain(TaskHandle task) {
  mutex_lock lock(chains_mu_);
  auto iter = chains_.find(task);
  assert(iter != chains_.end());
  return iter->second;
}

void RemoteChainManager::SetRemoteChain(TaskHandle task, RemoteObjectId chain) {
  // TODO(bramandia): Send batch delete remote object request for previous
  // chain
  mutex_lock lock(chains_mu_);
  chains_.erase(task);
  chains_.insert({task, chain});
}

}  // namespace tfrt
