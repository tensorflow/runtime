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
// This file declares DistributedContext, which constructs and owns fabric
// communicator.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DISTRIBUTED_RUNTIME_DISTRIBUTED_CONTEXT_H_
#define TFRT_DISTRIBUTED_RUNTIME_DISTRIBUTED_CONTEXT_H_

#include <string>

#include "llvm/ADT/StringMap.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/mutex.h"

namespace tfrt {

#define __TFRT_DIST_UNIQUE_NAME(base_name) \
  __TFRT_DIST_NAME_MERGE(base_name, __COUNTER__)
#define __TFRT_DIST_NAME_MERGE(name1, name2) name1##name2

#define TFRT_STATIC_FABRIC_COMMUNICATOR_REGISTRATION(communicator_type_name, \
                                                     factory_function)       \
  static bool __TFRT_DIST_UNIQUE_NAME(__tfrt_static_communicator_) = []() {  \
    ::tfrt::DistributedContext::RegisterFabricCommunicatorType(              \
        communicator_type_name, std::move(factory_function));                \
    return true;                                                             \
  }()

class FabricCommunicator;
class CallbackRegistry;

using InstanceKey = std::string;
using Rank = int32_t;

struct FabricCommunicatorConfiguration {
  std::string type;                       // fabric type, e.g. grpc
  llvm::StringMap<std::string> settings;  // fabric-specific settings
};

struct DistributedContextConfiguration {
  // Mapping from FabricCommunicator name to FabricCommunicator configuration
  llvm::StringMap<FabricCommunicatorConfiguration> communicators;
};

// DistributedContext constructs and owns fabric communicators.
class DistributedContext {
 public:
  using FabricCommunicatorFactory = std::function<FabricCommunicator*(
      llvm::StringRef communicator_name,
      DistributedContext* distributed_context,
      const FabricCommunicatorConfiguration& configuration)>;

  explicit DistributedContext(HostContext* host_context,
                              DistributedContextConfiguration configuration);

  DistributedContext(DistributedContext&&) = delete;
  DistributedContext& operator=(DistributedContext&&) = delete;

  DistributedContext(const DistributedContext&) = delete;
  DistributedContext& operator=(const DistributedContext&) = delete;

  FabricCommunicator* GetOrCreateFabricCommunicator(
      const std::string& communicator_name) TFRT_EXCLUDES(communicators_mutex_);

  HostContext* GetHostContext() { return host_context_; }
  CallbackRegistry* GetCallbackRegistry() { return callback_registry_.get(); }

  static void RegisterFabricCommunicatorType(
      const std::string& communicator_type_name,
      FabricCommunicatorFactory factory_function);

 private:
  static llvm::StringMap<FabricCommunicatorFactory>*
  GetFabricCommunicatorFactories();
  FabricCommunicator* GetOrCreateFabricCommunicatorUnsafe(
      llvm::StringRef communicator_name) TFRT_REQUIRES(communicators_mutex_);
  void InitializeAllFabricCommunicators() TFRT_EXCLUDES(communicators_mutex_);

  HostContext* const host_context_;
  const DistributedContextConfiguration configuration_;
  std::unique_ptr<CallbackRegistry> callback_registry_;
  mutex communicators_mutex_;
  llvm::StringMap<std::unique_ptr<FabricCommunicator>> fabric_communicators_
      TFRT_GUARDED_BY(communicators_mutex_);
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_DISTRIBUTED_CONTEXT_H_
