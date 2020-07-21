// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- test_kernels.cc ----------------------------------------------------===//
//
// This file implements kernels for testing distributed kernels.
//
//===----------------------------------------------------------------------===//

#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {

namespace {

void TestSetupFromString(Argument<HostId> id,
                         Result<DistributedContext> dist_context,
                         const ExecutionContext& exec_ctx) {
  HostConfiguration host_config{{"localhost:40002", "localhost:50002",
                                 "localhost:40001", "localhost:50001"},
                                id.get()};
  FabricCommunicatorConfiguration fabric_config{"grpc_communicator",
                                                host_config};
  CollectiveGroup group1{/*name=*/"group1", /*members=*/{0, 1, 2, 3}};
  DistributedContextConfiguration dist_context_config{fabric_config, {group1}};
  dist_context.Emplace(exec_ctx.host(), dist_context_config);
}

}  // namespace

void RegisterDistributedTestKernels(KernelRegistry* registry) {
  registry->AddKernel("dist.test_setup_from_string",
                      TFRT_KERNEL(TestSetupFromString));
}

}  // namespace tfrt
