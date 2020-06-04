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

#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/mutex.h"

namespace tfrt {

DistributedContext::DistributedContext(
    HostContext* host_context, DistributedContextConfiguration configuration)
    : host_context_{host_context}, configuration_{std::move(configuration)} {
  InitializeAllFabricCommunicators();
}

FabricCommunicator* DistributedContext::GetOrCreateFabricCommunicator(
    const std::string& communicator_name) {
  mutex_lock lock(communicators_mutex_);
  return GetOrCreateFabricCommunicatorUnsafe(communicator_name);
}

FabricCommunicator* DistributedContext::GetOrCreateFabricCommunicatorUnsafe(
    llvm::StringRef communicator_name) {
  // Don't create a new communicator if cached
  auto comms_iter = fabric_communicators_.find(communicator_name);
  if (comms_iter != fabric_communicators_.end()) {
    return comms_iter->second.get();
  }

  // Get communicator configuration
  const auto& communicators_config = configuration_.communicators;
  auto config_iter = communicators_config.find(communicator_name);
  if (config_iter == communicators_config.end()) {
    TFRT_LOG(WARNING) << "Did not find configuration for fabric communicator "
                      << communicator_name;
    return nullptr;
  }
  const auto& communicator_configuration = config_iter->second;
  const auto& communicator_type = communicator_configuration.type;

  // Get communicator type factory
  const auto* communicator_factories = GetFabricCommunicatorFactories();
  auto factories_iter = communicator_factories->find(communicator_type);
  if (factories_iter == communicator_factories->end()) {
    TFRT_LOG(WARNING) << "Did not find fabric communicator factory for "
                         "communicator with name "
                      << communicator_name << " and type " << communicator_type;
    ;
    return nullptr;
  }
  const auto& factory_function = factories_iter->second;

  // Create FabricCommunicator
  FabricCommunicator* communicator =
      factory_function(communicator_name, this, communicator_configuration);

  auto emplace_ret = fabric_communicators_.try_emplace(
      communicator_name, std::unique_ptr<FabricCommunicator>(communicator));
  assert(emplace_ret.second);
  return emplace_ret.first->getValue().get();
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

void DistributedContext::InitializeAllFabricCommunicators() {
  mutex_lock lock(communicators_mutex_);
  const auto& communicators_config = configuration_.communicators;
  for (const auto& communicator_entry : communicators_config) {
    const auto& communicator_name = communicator_entry.first();
    GetOrCreateFabricCommunicatorUnsafe(communicator_name);
  }
}

}  // namespace tfrt
