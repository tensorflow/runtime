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

// This file declares remote client interface.

#ifndef TFRT_DISTRIBUTED_RUNTIME_REMOTE_CLIENT_H_
#define TFRT_DISTRIBUTED_RUNTIME_REMOTE_CLIENT_H_

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Error.h"
#include "tfrt/distributed_runtime/proto/remote_message.pb.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {

class RemoteCallContext {
 public:
  using CancelCallback = llvm::unique_function<void()>;

  // TODO(haoyuzhang): consider preventing callers from invoking StartCancel,
  // SetCancelCallback, etc. on the default RemoteCallContext instance.
  static RemoteCallContext* GetDefault() {
    // Default to no retries and unlimited timeout.
    static RemoteCallContext* default_ctx =
        new RemoteCallContext(/*max_retries=*/0, /*timeout_ms=*/0);
    return default_ctx;
  }

  RemoteCallContext(uint32_t max_retries, uint64_t timeout_ms)
      : max_retries_(max_retries), timeout_ms_(timeout_ms) {}

  uint32_t GetMaxRetries() const { return max_retries_; }
  uint64_t GetTimeoutMs() const { return timeout_ms_; }

  void SetCancelCallback(CancelCallback callback) {
    mutex_lock l(mu_);
    cancel_cb_ = std::move(callback);
  }

  void StartCancel() {
    mutex_lock l(mu_);
    if (cancel_cb_) {
      cancel_cb_();
    }
  }

  void ClearCancelCallback() {
    mutex_lock l(mu_);
    cancel_cb_ = CancelCallback();
  }

 private:
  // Number of maximum retries if the RPC call fails.
  const uint32_t max_retries_;

  // Remote call timeout in milliseconds.
  // If set to 0, there is no timeout for the remote call.
  const uint64_t timeout_ms_;

  mutex mu_;
  // Callback function when this remote call is cancelled.
  CancelCallback cancel_cb_ TFRT_GUARDED_BY(mu_);
};

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

#define CLIENT_METHOD(method)                                \
  virtual void method##Async(RemoteCallContext* call_ctx,    \
                             const method##Request* request, \
                             method##Response* response, CallbackFn done) = 0;

  // Get device information on remote task.
  CLIENT_METHOD(GetDevices);

  // Create a DistributedContext on the remote task. If the remote task already
  // has a context with the requested context_id, the done callback will be
  // invoked with DistributedContextAlreadyExists error.
  CLIENT_METHOD(CreateContext);

  // Close DistributedContext on the remote task. If the remote task does not
  // have a context with the requested context_id, the done callback will be
  // invoked with InvalidDistributedContextId error.
  CLIENT_METHOD(CloseContext);

  // Broadcast the remote ready chains to all tasks in the cluster. This should
  // be used by the multi-client leader task to share the collected remote ready
  // chains to all non-leader tasks.
  CLIENT_METHOD(SendReadyChains);

  CLIENT_METHOD(SendData);
  CLIENT_METHOD(RegisterFunction);

  // The callback will be invoked once a response is received from the
  // destination. This might not mean the actual execution has completed in
  // the destination.
  CLIENT_METHOD(RemoteExecute);

  // Execute a single op on a remote device.  The callback is invoked when
  // execution completes on the remote side.
  CLIENT_METHOD(RemoteExecuteOp);

  // Delete Remote objects on remote location. The callback will be invoked
  // after the deletion has completed on remote location.
  CLIENT_METHOD(DeleteRemoteObjects);

  // Update the last active time of the remote DistributedContext to prevent it
  // from being garbage collected by the remote host.
  CLIENT_METHOD(KeepAlive);

#undef CLIENT_METHOD
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_REMOTE_CLIENT_H_
