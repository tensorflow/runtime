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

//===- distributed_context.cc - Distributed Context -------------*- C++ -*-===//
//
// This file contains the implementation of distributed context.
//
//===----------------------------------------------------------------------===//

#include "tfrt/distributed_runtime/distributed_context.h"

#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/mutex.h"

namespace tfrt {

DistributedContext::DistributedContext(
    HostContext* host_context, DistributedContextConfiguration configuration)
    : host_context_{host_context},
      configuration_{std::move(configuration)},
      callback_registry_(new CallbackRegistry()) {
  GetOrCreateFabricCommunicator();
}

FabricCommunicator* DistributedContext::GetOrCreateFabricCommunicator() {
  mutex_lock lock(communicator_mutex_);
  return GetOrCreateFabricCommunicatorUnsafe();
}

CollectiveGroup DistributedContext::GetCollectiveGroup(llvm::StringRef name) {
  CollectiveGroup collective_group;
  bool found = false;
  for (const auto& registered_group : configuration_.collective_groups) {
    if (registered_group.name == name) {
      collective_group = registered_group;
      found = true;
    }
  }
  if (!found) {
    TFRT_LOG(WARNING) << "Did not find collective group";
  }
  return collective_group;
}

FabricCommunicator* DistributedContext::GetOrCreateFabricCommunicatorUnsafe() {
  // Don't create a new communicator if cached
  if (fabric_communicator_ != nullptr) {
    return fabric_communicator_.get();
  }

  // Get communicator type factory
  const auto& communicator_configuration = configuration_.fabric_configuration;
  const auto& communicator_type = communicator_configuration.type;
  const auto* communicator_factories = GetFabricCommunicatorFactories();
  auto factories_iter = communicator_factories->find(communicator_type);
  if (factories_iter == communicator_factories->end()) {
    TFRT_LOG(WARNING) << "Did not find fabric communicator factory for "
                         "communicator with type "
                      << communicator_type;
    return nullptr;
  }
  const auto& factory_function = factories_iter->second;

  // Create FabricCommunicator
  fabric_communicator_.reset(
      factory_function(this, communicator_configuration));
  return fabric_communicator_.get();
}

void DistributedContext::RegisterFabricCommunicatorType(
    const std::string& communicator_type_name,
    FabricCommunicatorFactory factory_function) {
  auto communicator_factories = GetFabricCommunicatorFactories();
  communicator_factories->try_emplace(communicator_type_name,
                                      std::move(factory_function));
}

llvm::StringMap<DistributedContext::FabricCommunicatorFactory>*
DistributedContext::GetFabricCommunicatorFactories() {
  static auto* communicator_factories =
      new llvm::StringMap<FabricCommunicatorFactory>();
  return communicator_factories;
}

}  // namespace tfrt
