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

// Unit test for TFRT parallel algorithms.

#include "tfrt/host_context/parallel_for.h"

#include <chrono>
#include <thread>
#include <utility>

#include "gtest/gtest.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/latch.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

using BlockSizes = ParallelFor::BlockSizes;
using Range = std::pair<size_t, size_t>;

static std::unique_ptr<HostContext> CreateTestHostContext(int num_threads) {
  return std::make_unique<HostContext>(
      [](const DecodedDiagnostic&) {}, CreateMallocAllocator(),
      CreateMultiThreadedWorkQueue(num_threads, num_threads));
}

static ExecutionContext CreateTestExecutionContext(HostContext* host) {
  Expected<RCReference<RequestContext>> request_ctx =
      RequestContextBuilder(host, /*resource_context=*/nullptr).build();
  EXPECT_FALSE(!request_ctx);
  return ExecutionContext{std::move(*request_ctx)};
}

TEST(ParallelForTest, FixedBlockSize) {
  auto host = CreateTestHostContext(4);
  ParallelFor pfor(CreateTestExecutionContext(host.get()));

  latch barrier(5);  // 4 tasks + 1 on-done callback
  mutex mu;
  std::vector<Range> ranges;

  AsyncValueRef<Chain> done =
      pfor.Execute(100, BlockSizes::Fixed(25), [&](size_t begin, size_t end) {
        mutex_lock lock(mu);
        ranges.push_back({begin, end});
        barrier.count_down();
      });
  done.AndThen([&]() { barrier.count_down(); });

  barrier.wait();

  std::sort(ranges.begin(), ranges.end());
  const std::vector<Range> expected = {{0, 25}, {25, 50}, {50, 75}, {75, 100}};

  ASSERT_EQ(ranges.size(), 4);
  ASSERT_EQ(ranges, expected);
}

TEST(ParallelForTest, MinBlockSize) {
  auto host = CreateTestHostContext(4);
  ParallelFor pfor(CreateTestExecutionContext(host.get()));

  latch barrier(3);  // 2 tasks + 1 on-done callback
  mutex mu;
  std::vector<Range> ranges;

  AsyncValueRef<Chain> done =
      pfor.Execute(100, BlockSizes::Min(75), [&](size_t begin, size_t end) {
        mutex_lock lock(mu);
        ranges.push_back({begin, end});
        barrier.count_down();
      });
  done.AndThen([&]() { barrier.count_down(); });

  barrier.wait();

  std::sort(ranges.begin(), ranges.end());
  const std::vector<Range> expected = {{0, 75}, {75, 100}};

  ASSERT_EQ(ranges.size(), 2);
  ASSERT_EQ(ranges, expected);
}

TEST(ParallelForTest, BlockTasksCompletion) {
  auto host = CreateTestHostContext(4);
  ParallelFor pfor(CreateTestExecutionContext(host.get()));

  latch barrier(1);
  std::atomic<int32_t> completed_tasks{0};

  AsyncValueRef<Chain> done =
      pfor.Execute(100, BlockSizes::Fixed(1),
                   [&](size_t begin, size_t end) { completed_tasks++; });

  done.AndThen([&]() {
    // All tasks should have completed when AndThen is called.
    ASSERT_EQ(completed_tasks.load(), 100);
    barrier.count_down();
  });

  barrier.wait();
}

TEST(ParallelForTest, ExecuteNestedParallelism) {
  auto host = CreateTestHostContext(4);
  auto exec_ctx = CreateTestExecutionContext(host.get());
  ParallelFor pfor(exec_ctx);

  auto fork = [&](size_t begin, size_t end) -> AsyncValueRef<Range> {
    return EnqueueWork(exec_ctx, [begin, end]() -> Range {
      std::this_thread::sleep_for(std::chrono::milliseconds(150 - end));
      return std::make_pair(begin, end);
    });
  };

  auto join =
      [&](ArrayRef<AsyncValueRef<Range>> results) -> std::vector<Range> {
    std::vector<Range> out;
    for (auto& result : results) out.push_back(result.get());
    return out;
  };

  auto result = pfor.Execute<Range, std::vector<Range>>(
      100, BlockSizes::Fixed(25), fork, join);

  latch barrier(1);
  result.AndThen([&]() { barrier.count_down(); });
  barrier.wait();

  std::vector<Range> ranges = result.get();
  std::sort(ranges.begin(), ranges.end());
  const std::vector<Range> expected = {{0, 25}, {25, 50}, {50, 75}, {75, 100}};

  ASSERT_EQ(ranges.size(), 4);
  ASSERT_EQ(ranges, expected);
}

}  // namespace tfrt
