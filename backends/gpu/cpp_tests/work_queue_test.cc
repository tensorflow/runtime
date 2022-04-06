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

// Unit test for GpuWorkQueue.

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "common.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/gpu/gpu_executor.h"
#include "tfrt/host_context/async_dispatch.h"

namespace tfrt {
namespace gpu {
using wrapper::Test;

TEST_F(Test, RunBlockingWork) {
  mlir::MLIRContext context;
  auto host_ctx = CreateHostContext(GetDiagHandler(&context));

  std::mutex mutex;
  std::condition_variable cond_var;
  const int count = 15;  // 2^n - 1
  int run = 0;

  auto condition = [&] { return run == count; };

  std::function<void()> work;
  work = [&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::lock_guard<std::mutex> lock(mutex);
    EXPECT_LE(++run, count);
    if (condition()) return cond_var.notify_all();
    if (run <= count / 2) {  // First half of tasks launch 2 more.
      EXPECT_TRUE(RunBlockingWork(host_ctx.get(), work));
      EXPECT_TRUE(RunBlockingWork(host_ctx.get(), work));
    }
  };

  EXPECT_TRUE(RunBlockingWork(host_ctx.get(), work));

  std::unique_lock<std::mutex> lock(mutex);
  EXPECT_TRUE(cond_var.wait_for(lock, std::chrono::seconds(1), condition));
}

TEST_F(Test, Await) {
  mlir::MLIRContext context;
  auto host_ctx = CreateHostContext(GetDiagHandler(&context));

  AsyncValueRef<int> async_value = MakeConstructedAsyncValueRef<int>();

  EXPECT_TRUE(RunBlockingWork(host_ctx.get(), [&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    async_value.SetStateConcrete();
  }));

  host_ctx->Await(async_value.CopyRCRef());
  EXPECT_TRUE(async_value.IsAvailable());
}

TEST_F(Test, Quiesce) {
  mlir::MLIRContext context;
  auto host_ctx = CreateHostContext(GetDiagHandler(&context));

  std::atomic<int> count(5);

  std::function<void()> work;
  work = [&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (count.fetch_sub(1) == 1) return;
    EXPECT_TRUE(RunBlockingWork(host_ctx.get(), work));
  };

  EXPECT_TRUE(RunBlockingWork(host_ctx.get(), work));
  host_ctx->Quiesce();
  EXPECT_EQ(count.load(), 0);
}

}  // namespace gpu
}  // namespace tfrt
