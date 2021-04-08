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

// This file declares CreateRemoteOpHandler, which creates Remote Op Handler
// responsible for executing ops on remote device.

#ifndef TFRT_DISTRIBUTED_KERNELS_REMOTE_OP_HANDLER_H_
#define TFRT_DISTRIBUTED_KERNELS_REMOTE_OP_HANDLER_H_

#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/remote_chain_manager.h"
#include "tfrt/distributed_runtime/remote_device.h"
#include "tfrt/distributed_runtime/task_handle.h"

namespace tfrt {

llvm::Expected<OpHandler*> CreateRemoteOpHandler(
    DistributedContext* dist_ctx, RemoteChainManager* remote_chain_manager,
    RCReference<RemoteDevice> remote_device);

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_KERNELS_REMOTE_OP_HANDLER_H_
