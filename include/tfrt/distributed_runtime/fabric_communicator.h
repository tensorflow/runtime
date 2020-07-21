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

// FabricCommunicator is an abstraction for a layer between the kernel and the
// network fabric which is able to send and receive data.
class FabricCommunicator {
 public:
  explicit FabricCommunicator(llvm::StringRef name,
                              DistributedContext* distributed_context)
      : name_{name}, distributed_context_{distributed_context} {};

  virtual ~FabricCommunicator() = default;

  virtual void Send(InstanceKey instance_key, HostId destination,
                    llvm::StringRef payload) = 0;

  const std::string& GetFabricCommunicatorName() const { return name_; }

 protected:
  std::string name_;
  DistributedContext* distributed_context_;
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_FABRIC_COMMUNICATOR_H_
