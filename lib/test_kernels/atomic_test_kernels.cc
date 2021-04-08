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

// This file implements atomic kernels, useful for writing tests for various
// concurrency situations.

#include <atomic>

#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/test_kernels.h"

namespace tfrt {

// NOTE: This should be writable as a normal TFRT_KERNEL but we need return a
// non-movable type.
static void TestAtomicCreateI32(Result<std::atomic<int32_t>> result) {
  result.Emplace(0);
}

// NOTE: This should be writable as a normal TFRT_KERNEL but we need to mutate
// an argument.
static Chain TestAtomicIncI32(Argument<std::atomic<int32_t>> in) {
  in->fetch_add(1);
  return Chain();
}

static Chain TestAtomicAddI32(Argument<std::atomic<int32_t>> in,
                              Argument<int32_t> value) {
  in->fetch_add(*value);
  return Chain();
}

static std::pair<int32_t, Chain> TestAtomicGetI32(
    Argument<std::atomic<int32_t>> in) {
  return {in->load(), Chain()};
}

// Install some atomic kernels and types for use by the test driver.
void RegisterAtomicTestKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_test.atomic.create.i32",
                      TFRT_KERNEL(TestAtomicCreateI32));
  registry->AddKernel("tfrt_test.atomic.inc.i32",
                      TFRT_KERNEL(TestAtomicIncI32));
  registry->AddKernel("tfrt_test.atomic.add.i32",
                      TFRT_KERNEL(TestAtomicAddI32));
  registry->AddKernel("tfrt_test.atomic.get.i32",
                      TFRT_KERNEL(TestAtomicGetI32));
}
}  // namespace tfrt
