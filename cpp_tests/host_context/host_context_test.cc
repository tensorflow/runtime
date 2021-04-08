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

// Unit test for TFRT HostContext.

#include "tfrt/host_context/host_context.h"

#include <set>
#include <thread>

#include "gtest/gtest.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/latch.h"
#include "tfrt/support/mutex.h"

namespace tfrt {
namespace {

std::unique_ptr<HostContext> CreateTestHostContext(int num_threads) {
  return std::make_unique<HostContext>(
      [](const DecodedDiagnostic&) {}, CreateMallocAllocator(),
      CreateMultiThreadedWorkQueue(num_threads, num_threads));
}

TEST(HostContextTest, RunBlockingWork) {
  // Create a host context with one thread to test that RunBlockingWork
  // dynamically starts threads to run submitted work.
  auto host = CreateTestHostContext(1);

  latch start(1);
  latch done(10);

  mutex mu;
  std::set<std::thread::id> threads;

  for (int i = 0; i < 10; ++i) {
    bool submitted = RunBlockingWork(host.get(), [&] {
      start.wait();
      {
        mutex_lock lock(mu);
        threads.insert(std::this_thread::get_id());
      }
      done.count_down();
    });
    ASSERT_TRUE(submitted);
  }

  start.count_down();
  done.wait();

  ASSERT_EQ(threads.size(), 10);
}

TEST(HostContextTest, RunBlockingWorkWithResult) {
  auto host = CreateTestHostContext(1);

  AsyncValueRef<int> result = RunBlockingWork(host.get(), [] { return 42; });

  llvm::SmallVector<RCReference<AsyncValue>, 4> refs;
  refs.push_back(result.CopyRCRef());
  host->Await(refs);

  ASSERT_EQ(result.get(), 42);
}

}  // namespace
}  // namespace tfrt
