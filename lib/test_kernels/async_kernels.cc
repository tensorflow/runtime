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

//===- async_kernels.cc ---------------------------------------------------===//
//
// This file implements a few simple classes of asynchronous kernels for
// testing.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <thread>

#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/test_kernels.h"

namespace tfrt {

static AsyncValueRef<int32_t> HexAsyncAddI32(int32_t arg0, int32_t arg1,
                                             HostContext* host) {
  // Even though a single scalar add is trivial, we can do it on a background
  // thread if we'd like!
  return host->EnqueueWork([arg0, arg1] { return arg0 + arg1; });
}

static AsyncValueRef<bool> HexAsyncConstantI1(Attribute<int8_t> arg,
                                              HostContext* host) {
  return host->EnqueueWork([arg = *arg] { return arg != 0; });
}

static AsyncValueRef<int32_t> HexAsyncConstantI32(Attribute<int32_t> arg,
                                                  HostContext* host) {
  return host->EnqueueWork([arg = *arg] { return arg; });
}

static AsyncValueRef<int32_t> TestAsyncCopy(int32_t in, HostContext* host) {
  return host->EnqueueWork([in] { return in; });
}

// Install some async kernels for use by the test driver.
void RegisterAsyncKernels(KernelRegistry* registry) {
  registry->AddKernel("hex.async_constant.i1", TFRT_KERNEL(HexAsyncConstantI1));
  registry->AddKernel("hex.async_constant.i32",
                      TFRT_KERNEL(HexAsyncConstantI32));
  registry->AddKernel("hex.async_add.i32", TFRT_KERNEL(HexAsyncAddI32));
  registry->AddKernel("hex.async_copy.i32", TFRT_KERNEL(TestAsyncCopy));
}
}  // namespace tfrt
