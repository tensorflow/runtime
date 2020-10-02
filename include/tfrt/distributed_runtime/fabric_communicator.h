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
#include "tfrt/distributed_runtime/remote_object.h"

namespace tfrt {

// Arguments for remote execute request
struct RemoteExecuteInvocation {
  string_view program_name;  // The name of the program to be executed

  // This is a serializable version of GlobalId.
  struct Id {
    Id(int32_t prefix_id, int64_t local_id, string_view device)
        : prefix_id(prefix_id), local_id(local_id), device(device) {}
    int32_t prefix_id;
    int64_t local_id;
    string_view device;
  };
  struct Output {
    Output(int32_t prefix_id, int64_t local_id, string_view device,
           bool need_metadata)
        : id(prefix_id, local_id, device), need_metadata(need_metadata) {}
    Id id;
    bool need_metadata;
  };
  llvm::SmallVector<Id, 4> inputs;       // The list of inputs arguments
  llvm::SmallVector<Output, 4> outputs;  // The list of output arguments
};

struct RemoteExecuteInvocationResult {
  // TODO(bramandia): Propagate error message.
  bool ok;
  // Serialized metadata.
  llvm::SmallVector<std::string, 4> metadata;
};

// Arguments for remote register request
struct RemoteRegisterInvocation {
  string_view program;       // The body of the program to be registered
  string_view program_name;  // The name of the program to be registered
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
  using RemoteExecuteCallbackFn = llvm::unique_function<void(
      std::unique_ptr<RemoteExecuteInvocationResult>)>;
  virtual void HandleRemoteExecute(const RemoteExecuteInvocation& request,
                                   RemoteExecuteCallbackFn done) = 0;
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

  // TODO(b/168132685): Use Status input for async callback function.
  using CallbackFn = llvm::unique_function<void(bool /* success */)>;
  virtual void Send(InstanceKey instance_key, HostId destination,
                    llvm::StringRef payload, CallbackFn done) = 0;

  // The callback will be called once the program has been successfully
  // registered in the destination.
  virtual void RemoteRegister(HostId destination,
                              const RemoteRegisterInvocation& request,
                              CallbackFn done) = 0;

  // The callback will be called once a response is received from the
  // destination. This might not mean the actual execution has completed in
  // the destination.
  using RemoteExecuteCallbackFn = llvm::unique_function<void(
      std::unique_ptr<RemoteExecuteInvocationResult>)>;
  virtual void RemoteExecute(HostId destination,
                             const RemoteExecuteInvocation& request,
                             RemoteExecuteCallbackFn done) = 0;

  const std::string& GetFabricCommunicatorName() const { return name_; }

 protected:
  std::string name_;
  DistributedContext* distributed_context_;
  FabricCommunicatorRequestHandler* request_handler_;
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_FABRIC_COMMUNICATOR_H_
