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

//===- callback_registry.cc - Callback Registry ---------------------------===//
//
// This file contains the implementation of callback registry.

#include "tfrt/distributed_runtime/callback_registry.h"

#include "tfrt/distributed_runtime/distributed_context.h"

namespace tfrt {

void CallbackRegistry::SetValue(const InstanceKey& key, CallbackValue value) {
  bool execute_callback = false;
  Callback callback = nullptr;
  CallbackMap::iterator callback_iter;
  {
    mutex_lock lock(hashmap_mutex_);

    // If the callback for the key already exists, directly
    // execute the callback on the value.
    callback_iter = callbacks_.find(key);
    if (callback_iter == callbacks_.end()) {
      values_.insert_or_assign(key, std::move(value));
    } else {
      execute_callback = true;
      callback = std::move(callback_iter->second);
      callbacks_.erase(callback_iter);
    }
  }

  if (execute_callback) {
    ExecuteCallback(key, &callback, std::move(value));
  }
}

void CallbackRegistry::ExecuteCallback(const InstanceKey& key,
                                       Callback* callback,
                                       CallbackValue value) {
  (*callback)(key, std::move(value));
}

void CallbackRegistry::SetCallback(const InstanceKey& key, Callback callback) {
  bool execute_callback = false;
  CallbackValue value = Payload({});
  CallbackMap::iterator callback_iter;
  {
    mutex_lock lock(hashmap_mutex_);

    // If the value for the key already exists, directly execute
    // the callback on the value.
    auto value_iter = values_.find(key);
    if (value_iter != values_.end()) {
      value = std::move(value_iter->second);
      values_.erase(value_iter);
      execute_callback = true;
    } else {
      auto insert_ret = callbacks_.insert_or_assign(key, std::move(callback));
      callback_iter = insert_ret.first;
    }
  }

  if (execute_callback) {
    ExecuteCallback(key, &callback, std::move(value));
  }
}

void CallbackRegistry::DebugDump() {
  mutex_lock l(hashmap_mutex_);
  TFRT_LOG(INFO) << "Callbacks waiting:";
  for (const auto& p : callbacks_) {
    TFRT_LOG(INFO) << "  this=" << this << " " << p.first();
  }
  TFRT_LOG(INFO) << "Values waiting:";
  for (const auto& p : values_) {
    TFRT_LOG(INFO) << "  this=" << this << " " << p.first();
  }
}

}  // namespace tfrt
