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

// This file implements a few simple classes of asynchronous kernels for
// testing.

#include <chrono>
#include <random>
#include <thread>

#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/test_kernels.h"

namespace tfrt {
namespace {
// Sleeps in the current thread for a random duration between 1 and 10
// milliseconds. This is needed to produce multile async values with random
// completion order in tests.
void SleepForRandomDuration() {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(1, 10);
  std::this_thread::sleep_for(std::chrono::milliseconds(dist(rng)));
}

static AsyncValueRef<int32_t> TestAsyncAddI32(
    int32_t arg0, int32_t arg1, const ExecutionContext& exec_ctx) {
  // Even though a single scalar add is trivial, we can do it on a background
  // thread if we'd like!
  return EnqueueWork(exec_ctx, [arg0, arg1] { return arg0 + arg1; });
}

static AsyncValueRef<bool> TestAsyncConstantI1(
    Attribute<int8_t> arg, const ExecutionContext& exec_ctx) {
  return EnqueueWork(exec_ctx, [arg = *arg] { return arg != 0; });
}

static AsyncValueRef<int32_t> TestAsyncConstantI32(
    Attribute<int32_t> arg, const ExecutionContext& exec_ctx) {
  return EnqueueWork(exec_ctx, [arg = *arg] { return arg; });
}

// This implementation of TestAsyncCopy returns results directly.
template <typename T>
static AsyncValueRef<T> TestAsyncCopy(Argument<T> in,
                                      const ExecutionContext& exec_ctx) {
  return EnqueueWork(exec_ctx,
                     [in_ref = in.ValueRef()] { return in_ref.get(); });
}

// This implementation of TestAsyncCopy returns results via a 'Result'
// parameter.
static void TestAsyncCopy2(Argument<int32_t> in, Result<int32_t> out,
                           const ExecutionContext& exec_ctx) {
  EnqueueWork(exec_ctx, [in_ref = in.ValueRef(), out_ref = out.Allocate()] {
    out_ref.emplace(in_ref.get());
  });
}

// Returns a copy of an argument after a random delay.
template <typename T>
static AsyncValueRef<T> TestAsyncCopyWithDelay(
    Argument<T> in, const ExecutionContext& exec_ctx) {
  return EnqueueWork(exec_ctx, [in_ref = in.ValueRef()] {
    SleepForRandomDuration();
    return in_ref.get();
  });
}

}  // namespace

// Install some async kernels for use by the test driver.
void RegisterAsyncKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_test.async_constant.i1",
                      TFRT_KERNEL(TestAsyncConstantI1));
  registry->AddKernel("tfrt_test.async_constant.i32",
                      TFRT_KERNEL(TestAsyncConstantI32));
  registry->AddKernel("tfrt_test.async_add.i32", TFRT_KERNEL(TestAsyncAddI32));
  registry->AddKernel("tfrt_test.async_copy.i32",
                      TFRT_KERNEL(TestAsyncCopy<int32_t>));
  registry->AddKernel("tfrt_test.async_copy.with_delay.i32",
                      TFRT_KERNEL(TestAsyncCopyWithDelay<int32_t>));
  registry->AddKernel("tfrt_test.async_copy.with_delay.i64",
                      TFRT_KERNEL(TestAsyncCopyWithDelay<int64_t>));
  registry->AddKernel("tfrt_test.async_copy_2.i32",
                      TFRT_KERNEL(TestAsyncCopy2));
}
}  // namespace tfrt
