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

//===- remote_client.h ------------------------------------------*- C++ -*-===//
//
// This file declares remote client interface.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DISTRIBUTED_RUNTIME_REMOTE_CLIENT_H_
#define TFRT_DISTRIBUTED_RUNTIME_REMOTE_CLIENT_H_

#include "tfrt/distributed_runtime/proto/remote_message.pb.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

class RemoteClientInterface {
 public:
  virtual ~RemoteClientInterface() {}

  // Callback function for async remote calls.
  // The function should have request and response messages captured to ensure
  // they are always safe to access before invoking the callback function.
  //
  // If invoked with application error (originated from the remote TF instance),
  // the response might be imcomplete and the validity of its message fields
  // depend on the specific type of error.
  // If invoked with network error, the response message will be empty.
  using CallbackFn = llvm::unique_function<void(Error error)>;

  virtual void SendAsync(const SendDataRequest* request,
                         SendDataResponse* response, CallbackFn done) = 0;

  virtual void RegisterFunctionAsync(const RegisterFunctionRequest* request,
                                     RegisterFunctionResponse* response,
                                     CallbackFn done) = 0;

  // The callback will be invoked once a response is received from the
  // destination. This might not mean the actual execution has completed in
  // the destination.
  virtual void RemoteExecuteAsync(const RemoteExecuteRequest* request,
                                  RemoteExecuteResponse* response,
                                  CallbackFn done) = 0;

  // Delete Remote objects on remote location. The callback will be invoked
  // after the deletion has completed on remote location.
  virtual void DeleteRemoteObjectsAsync(
      const DeleteRemoteObjectsRequest* request,
      DeleteRemoteObjectsResponse* response, CallbackFn done) = 0;
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_REMOTE_CLIENT_H_
