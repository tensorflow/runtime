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

// Fabric Communicator
//
// This file declares Fabric Communicator, which represents the network layer.

#ifndef TFRT_DISTRIBUTED_RUNTIME_FABRIC_COMMUNICATOR_H_
#define TFRT_DISTRIBUTED_RUNTIME_FABRIC_COMMUNICATOR_H_

#include <string>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/distributed_runtime/task_handle.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/logging.h"

namespace tfrt {

class DistributedContext;
class RemoteClientInterface;
class RequestHandlerInterface;
class ServerContext;

#define __TFRT_DIST_UNIQUE_NAME(base_name) \
  __TFRT_DIST_NAME_MERGE(base_name, __COUNTER__)
#define __TFRT_DIST_NAME_MERGE(name1, name2) name1##name2

#define TFRT_STATIC_FABRIC_COMMUNICATOR_REGISTRATION(communicator_type_name, \
                                                     factory_function)       \
  static bool __TFRT_DIST_UNIQUE_NAME(__tfrt_static_communicator_) = []() {  \
    ::tfrt::FabricCommunicator::RegisterFabricCommunicatorType(              \
        communicator_type_name, std::move(factory_function));                \
    return true;                                                             \
  }()

// TODO(pisong, ayushd): remove `using`.
using InstanceKey = std::string;

struct FabricCommunicatorConfiguration {
  std::string type;            // fabric type, (e.g., grpc)
  std::string server_address;  // Server address (e.g., hostname:port)
};

// FabricCommunicator is an abstraction for a layer between the kernel and the
// network fabric which is able to send and receive data.
class FabricCommunicator {
 public:
  using FabricCommunicatorFactory =
      std::function<FabricCommunicator*(ServerContext* server_context)>;

  // Create FabricCommunicator and redirects the received requests to the
  // request_handler.
  FabricCommunicator(llvm::StringRef name, ServerContext* server_context)
      : name_{name}, server_context_(server_context) {}

  virtual ~FabricCommunicator() = default;

  const std::string& GetFabricCommunicatorName() const { return name_; }

  static void RegisterFabricCommunicatorType(
      const std::string& communicator_type_name,
      FabricCommunicatorFactory factory_function) {
    auto communicator_factories = GetFabricCommunicatorFactories();
    communicator_factories->try_emplace(communicator_type_name,
                                        std::move(factory_function));
  }

  static FabricCommunicator* CreateFabricCommunicator(
      string_view communicator_type, ServerContext* server) {
    // Get communicator type factory
    const auto* communicator_factories = GetFabricCommunicatorFactories();
    auto factories_iter = communicator_factories->find(communicator_type);
    if (factories_iter == communicator_factories->end()) {
      TFRT_LOG(WARNING) << "Did not find fabric communicator factory for "
                           "communicator with type "
                        << communicator_type;
      return nullptr;
    }
    const auto& factory_function = factories_iter->second;
    return factory_function(server);
  }

  virtual std::unique_ptr<RemoteClientInterface> CreateRemoteClient(
      DistributedContext* dist_context, TaskHandle task_handle) = 0;

 protected:
  const std::string name_;
  ServerContext* server_context_;

 private:
  static llvm::StringMap<FabricCommunicatorFactory>*
  GetFabricCommunicatorFactories() {
    static auto* communicator_factories =
        new llvm::StringMap<FabricCommunicatorFactory>();
    return communicator_factories;
  }
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_FABRIC_COMMUNICATOR_H_
