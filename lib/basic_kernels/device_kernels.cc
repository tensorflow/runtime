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

// This file implements host executor kernels for devices.

#include "tfrt/basic_kernels/basic_kernels.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace {

Expected<RCReference<Device>> GetDeviceKernel(
    Chain chain, StringAttribute device_name,
    const ExecutionContext& exec_ctx) {
  auto result = exec_ctx.host()->GetDeviceManager()->GetDeviceRef<Device>(
      device_name.get());
  if (!result) {
    return MakeStringError("cannot find device ", device_name.get());
  }
  return std::move(result);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void RegisterDeviceKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt.get_device", TFRT_KERNEL(GetDeviceKernel));
}

}  // namespace tfrt
