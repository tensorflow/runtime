// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements RemoteObjectManager.

#include "tfrt/distributed_runtime/remote_object_manager.h"

#include "tfrt/host_context/async_value_ref.h"

namespace tfrt {
const uint64_t RemoteObjectManager::kInvalidPrefixId = 0;

RemoteObjectManager::RemoteObjectManager(TaskHandle task_handle,
                                         HostContext* host_context)
    : prefix_id_(task_handle.get_value()), host_context_(host_context) {
  assert(prefix_id_ != kInvalidPrefixId);
}
RemoteObjectManager::~RemoteObjectManager() {}

RemoteObjectId RemoteObjectManager::AllocateRemoteObject(
    RCReference<Device> output_device) {
  const uint64_t local_id = next_unique_id_.fetch_add(1);
  return RemoteObjectId(prefix_id_, local_id, output_device);
}

void RemoteObjectManager::SetRemoteObject(const RemoteObjectId& id,
                                          RCReference<AsyncValue> value) {
  RCReference<AsyncValue> val = GetRemoteObject(id);
  assert(val->IsUnresolvedIndirect());
  cast<IndirectAsyncValue>(val.get())->ForwardTo(value);
}

RCReference<AsyncValue> RemoteObjectManager::GetRemoteObject(
    const RemoteObjectId& id) {
  tfrt::mutex_lock lock(mutex_);
  auto iter = object_maps_.find(id);
  if (iter != object_maps_.end()) {
    return iter->second;
  }
  RCReference<AsyncValue> value = MakeIndirectAsyncValue(host_context_);
  object_maps_[id] = value;
  return value;
}

// Delete the given remote object ids.
Error RemoteObjectManager::DeleteRemoteObjects(
    const llvm::SmallVectorImpl<RemoteObjectId>& ids) {
  tfrt::mutex_lock lock(mutex_);
  std::unique_ptr<ErrorCollection> errors;
  for (const RemoteObjectId& id : ids) {
    if (!object_maps_.erase(id)) {
      if (!errors) {
        errors = std::make_unique<ErrorCollection>();
      }
      errors->AddError(llvm::make_error<InvalidArgumentErrorInfo>(
          StrCat("Could not find object: ", id)));
    }
  }
  if (errors) {
    return Error(std::move(errors));
  } else {
    return Error::success();
  }
}

}  // namespace tfrt
