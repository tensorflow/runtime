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

//===- fabric_communicator.h - Fabric Communicator --------------*- C++ -*-===//
//
// This file declares Fabric Communicator, which represents the network layer.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DISTRIBUTED_RUNTIME_FABRIC_COMMUNICATOR_H_
#define TFRT_DISTRIBUTED_RUNTIME_FABRIC_COMMUNICATOR_H_

#include <string>

#include "tfrt/distributed_runtime/distributed_context.h"

namespace tfrt {

// Arguments for remote execute request
struct RemoteExecuteInvocation {
  std::string program_name;  // The name of the program to be executed
};

// Arguments for remote register request
struct RemoteRegisterInvocation {
  std::string program;       // The body of the program to be registered
  std::string program_name;  // The name of the program to be registered
};

// Define the handler for various incoming requests that are received by
// FabricCommunicator.
// Users of FabricCommunicator supplies an implementation of
// FabricCommunicatorRequestHandler to be called whenever an incoming request
// is received.
class FabricCommunicatorRequestHandler {
 public:
  virtual ~FabricCommunicatorRequestHandler() {}

  virtual void HandleRemoteRegister(
      const RemoteRegisterInvocation& request) = 0;
  virtual void HandleRemoteExecute(const RemoteExecuteInvocation& request) = 0;
};

// FabricCommunicator is an abstraction for a layer between the kernel and the
// network fabric which is able to send and receive data.
class FabricCommunicator {
 public:
  // Create FabricCommunicator and redirects the received requests to the
  // request_handler.
  explicit FabricCommunicator(llvm::StringRef name,
                              DistributedContext* distributed_context,
                              FabricCommunicatorRequestHandler* request_handler)
      : name_{name},
        distributed_context_{distributed_context},
        request_handler_(request_handler){};

  virtual ~FabricCommunicator() = default;

  virtual void Send(InstanceKey instance_key, HostId destination,
                    llvm::StringRef payload) = 0;

  // The callback will be called once the program has been successfully
  // registered in the destination.
  using CallbackFn = llvm::unique_function<void(bool /* success */)>;
  virtual void RemoteRegister(HostId destination,
                              const RemoteRegisterInvocation& request,
                              CallbackFn done) = 0;

  // The callback will be called once a response is received from the
  // destination. This might not mean the actual execution has completed in
  // the destination.
  virtual void RemoteExecute(HostId destination,
                             const RemoteExecuteInvocation& request,
                             CallbackFn done) = 0;

  const std::string& GetFabricCommunicatorName() const { return name_; }

 protected:
  std::string name_;
  DistributedContext* distributed_context_;
  FabricCommunicatorRequestHandler* request_handler_;
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_FABRIC_COMMUNICATOR_H_
