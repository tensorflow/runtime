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

//===- request_handler.h - Request Handler ----------------------*- C++ -*-===//
//
// This file declares RequestHandler, which implements logic to register and
// execute programs.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_DISTRIBUTED_RUNTIME_REQUEST_HANDLER_H_
#define TFRT_DISTRIBUTED_RUNTIME_REQUEST_HANDLER_H_

#include <llvm/ADT/ArrayRef.h>
#include <tfrt/support/ref_count.h>

#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/resource_context.h"

namespace tfrt {

class HostContext;
class RequestContext;

// Implementation of processing of RemoteRegister and RemoteExecute requests
// at the callee location.
class RequestHandler : public FabricCommunicatorRequestHandler {
 public:
  RequestHandler(DistributedContext* context);
  virtual ~RequestHandler();

  void HandleRemoteRegister(const RemoteRegisterInvocation& request) final;
  void HandleRemoteExecute(const RemoteExecuteInvocation& request) final;

 private:
  DistributedContext* context_;

  class FunctionCache;
  std::unique_ptr<FunctionCache> function_cache_;
};

}  // namespace tfrt
#endif  // TFRT_DISTRIBUTED_RUNTIME_REQUEST_HANDLER_H_
