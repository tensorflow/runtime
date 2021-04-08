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

// Callback Registry
//
// This file declares Callback Registry.  The Callback Registry enables
// asynchronously invoking a callback with a value corresponding to a key.

#ifndef TFRT_DISTRIBUTED_RUNTIME_CALLBACK_REGISTRY_H_
#define TFRT_DISTRIBUTED_RUNTIME_CALLBACK_REGISTRY_H_

#include "llvm/ADT/StringMap.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/payload.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"

namespace tfrt {

// CallbackRegistry calls a callback on a value for a key. Callback and
// value may be set in any order. If the value set first, CallbackRegistry will
// own it until a callback is set.
class CallbackRegistry {
 public:
  using CallbackValue = Payload;
  using Callback =
      llvm::unique_function<void(const InstanceKey&, CallbackValue)>;

  CallbackRegistry() = default;
  CallbackRegistry(const CallbackRegistry&) = delete;
  CallbackRegistry& operator=(const CallbackRegistry&) = delete;

  void SetValue(const InstanceKey& key, CallbackValue value);

  void SetCallback(const InstanceKey& key, Callback callback);

 private:
  using ValueMap = llvm::StringMap<CallbackValue>;
  using CallbackMap = llvm::StringMap<Callback>;

  void ExecuteCallback(const InstanceKey& key, Callback* callback,
                       CallbackValue value);
  // Print the registry state - for debugging.
  void DebugDump();

  // TODO(xldrx): use separate mutexes for values and callbacks
  mutex hashmap_mutex_;
  ValueMap values_ TFRT_GUARDED_BY(hashmap_mutex_);
  CallbackMap callbacks_ TFRT_GUARDED_BY(hashmap_mutex_);
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_CALLBACK_REGISTRY_H_
