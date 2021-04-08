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

// Remote Object Manager
//
// This file declares RemoteObjectManager which manages remote objects stored
// locally as well as allocates unique RemoteObjectId.
#ifndef TFRT_DISTRIBUTED_RUNTIME_REMOTE_OBJECT_MANAGER_H_
#define TFRT_DISTRIBUTED_RUNTIME_REMOTE_OBJECT_MANAGER_H_

#include <unordered_map>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "tfrt/distributed_runtime/remote_object.h"
#include "tfrt/distributed_runtime/task_handle.h"
#include "tfrt/host_context/async_value.h"

namespace tfrt {
class RemoteObjectManager {
 public:
  static const uint64_t kInvalidPrefixId;

  RemoteObjectManager(TaskHandle task_handle, HostContext* host_context);
  ~RemoteObjectManager();

  // Create new unique RemoteObjectId.
  // This is called by remote_execute kernels to allocate unique RemoteObject
  // for the outputs of execution.
  RemoteObjectId AllocateRemoteObject(RCReference<Device> output_device);

  // Store a remote object with the given id and value
  // This is called by RequestHandler implementation to store the remote
  // objects outputs of the remote_execute.
  void SetRemoteObject(const RemoteObjectId& id, RCReference<AsyncValue> value);

  // Retrieve Remote Object with given id.
  // This is called by RequestHandler implementation to retrieve the remote
  // objects input to the remote_execute.
  RCReference<AsyncValue> GetRemoteObject(const RemoteObjectId& id);

  // Delete the given remote object ids.
  Error DeleteRemoteObjects(const llvm::SmallVectorImpl<RemoteObjectId>& ids);

 private:
  std::atomic<int64_t> next_unique_id_{1};
  const uint64_t prefix_id_;
  HostContext* host_context_;

  tfrt::mutex mutex_;

  llvm::DenseMap<RemoteObjectId, RCReference<AsyncValue>> object_maps_
      TFRT_GUARDED_BY(mutex_);
};
}  // namespace tfrt
namespace llvm {
template <>
struct DenseMapInfo<tfrt::RemoteObjectId> {
  static tfrt::RemoteObjectId getEmptyKey() {
    return {tfrt::RemoteObjectManager::kInvalidPrefixId, 0,
            tfrt::RCReference<tfrt::Device>()};
  }
  static tfrt::RemoteObjectId getTombstoneKey() {
    return {tfrt::RemoteObjectManager::kInvalidPrefixId, 0,
            tfrt::RCReference<tfrt::Device>()};
  }
  static unsigned getHashValue(const tfrt::RemoteObjectId& id) {
    return id.local_id;
  }
  static bool isEqual(const tfrt::RemoteObjectId& left,
                      const tfrt::RemoteObjectId& right) {
    return left.local_id == right.local_id && left.prefix_id == right.prefix_id;
  }
};
}  // namespace llvm

#endif  // TFRT_DISTRIBUTED_RUNTIME_REMOTE_OBJECT_MANAGER_H_
