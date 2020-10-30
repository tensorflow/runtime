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

//===- server_context.h - Server Context ------------------------*- C++ -*-===//
//
// This file declares ServerContext, which constructs and owns fabric
// communicator.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DISTRIBUTED_RUNTIME_SERVER_CONTEXT_H_
#define TFRT_DISTRIBUTED_RUNTIME_SERVER_CONTEXT_H_

#include "llvm/ADT/DenseMap.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

class FabricCommunicator;
class RequestHandlerInterface;
class DistributedContext;
class DistributedContextConfiguration;

// Configurations that should be set at server startup time.
struct ServerContextConfiguration {
  FabricCommunicatorConfiguration fabric_configuration;
};

// ServerContext constructs and owns fabric communicators.
class ServerContext {
 public:
  explicit ServerContext(HostContext* host_context,
                         ServerContextConfiguration configuration);

  ServerContext(ServerContext&&) = delete;
  ServerContext& operator=(ServerContext&&) = delete;

  ServerContext(const ServerContext&) = delete;
  ServerContext& operator=(const ServerContext&) = delete;
  ~ServerContext();

  FabricCommunicator* GetOrCreateFabricCommunicator()
      TFRT_EXCLUDES(communicator_mutex_);

  HostContext* GetHostContext() const { return host_context_; }

  ServerContextConfiguration GetConfiguration() const { return configuration_; }

  Expected<DistributedContext*> GetDistributedContext(
      uint64_t context_id) const;

  AsyncValueRef<DistributedContext> GetDistributedContextAsyncValue(
      uint64_t context_id) const;

  Error CreateDistributedContext(uint64_t context_id,
                                 DistributedContextConfiguration configuration)
      TFRT_EXCLUDES(context_mu_);

  RequestHandlerInterface* GetRequestHandler() const {
    return request_handler_.get();
  }

  void ShutDown();

 protected:
  // Test-only method.
  void ResetRequestHandler(
      std::unique_ptr<RequestHandlerInterface> request_handler);

 private:
  FabricCommunicator* GetOrCreateFabricCommunicatorUnsafe()
      TFRT_REQUIRES(communicator_mutex_);

  HostContext* const host_context_;
  const ServerContextConfiguration configuration_;
  std::unique_ptr<RequestHandlerInterface> request_handler_;
  mutex communicator_mutex_;
  std::unique_ptr<FabricCommunicator> fabric_communicator_
      TFRT_GUARDED_BY(communicator_mutex_);

  mutable mutex context_mu_;
  llvm::DenseMap<uint64_t, AsyncValueRef<DistributedContext>> dist_contexts_;
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_SERVER_CONTEXT_H_
