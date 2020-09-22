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
#include "tfrt/distributed_runtime/remote_device.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/mutex.h"

namespace tfrt {

DistributedContext::DistributedContext(
    HostContext* host_context, DistributedContextConfiguration configuration)
    : host_context_{host_context},
      remote_manager_(std::make_unique<RemoteObjectManager>(
          configuration.fabric_configuration.host_configuration.id,
          host_context_)),
      configuration_{std::move(configuration)},
      callback_registry_(new CallbackRegistry()) {}

void DistributedContext::Init(
    std::unique_ptr<FabricCommunicatorRequestHandler> request_handler) {
  request_handler_ = std::move(request_handler);
  GetOrCreateFabricCommunicator();
}

DistributedContext::~DistributedContext() {}

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
  fabric_communicator_.reset(factory_function(this, request_handler_.get(),
                                              communicator_configuration));

  // TODO(bramandia): Get the list of devices from each worker and register
  // them.
  for (const auto& address :
       communicator_configuration.host_configuration.addresses) {
    const std::string device_name =
        StrCat(address.name, "/device:", HostContext::kDefaultHostDeviceName);
    host_context_->GetDeviceManager()->MaybeAddDevice(
        TakeRef(new RemoteCpuDevice(device_name)));
  }

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
