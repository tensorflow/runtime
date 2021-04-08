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

// This file contains TFRT kernels for creating RemoteOpHandler.

#include "op_handler_kernels.h"

#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/distributed_runtime/remote_device.h"
#include "tfrt/distributed_runtime/remote_op_handler.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {

static Expected<OpHandler*> CreateRemoteOpHandlerKernel(
    DistributedContext* dist_ctx, RemoteChainManager* remote_chain_manager,
    const RCReference<Device>& remote_device,
    const ExecutionContext& exec_ctx) {
  return CreateRemoteOpHandler(
      dist_ctx, remote_chain_manager,
      FormRef(static_cast<RemoteDevice*>(remote_device.get())));
}

void RegisterRemoteOpHandlerKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_dist.create_remote_op_handler",
                      TFRT_KERNEL(CreateRemoteOpHandlerKernel));
}

}  // namespace tfrt
