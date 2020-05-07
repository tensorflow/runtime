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

//===- parallel_for_test.cc -------------------------------------*- C++ -*-===//
//
// Unit test for TFRT parallel algorithms.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/parallel_for.h"

#include <chrono>
#include <thread>
#include <utility>

#include "gtest/gtest.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/latch.h"
#include "tfrt/support/mutex.h"

namespace tfrt {

using BlockSizes = ParallelFor::BlockSizes;
using Range = std::pair<size_t, size_t>;

std::unique_ptr<HostContext> CreateTestHostContext(int num_threads) {
  return std::make_unique<HostContext>(
      [](const DecodedDiagnostic&) {}, CreateMallocAllocator(),
      CreateMultiThreadedWorkQueue(num_threads, num_threads));
}

TEST(ParallelForTest, FixedBlockSize) {
  auto host = CreateTestHostContext(4);
  ParallelFor pfor(host.get());

  latch barrier(5);  // 4 tasks + 1 on-done callback
  mutex mu;
  std::vector<Range> ranges;

  pfor.Execute(
      100, BlockSizes::Fixed(25),
      [&](size_t begin, size_t end) {
        mutex_lock lock(mu);
        ranges.push_back({begin, end});
        barrier.count_down();
      },
      [&]() { barrier.count_down(); });

  barrier.wait();

  std::sort(ranges.begin(), ranges.end());
  const std::vector<Range> expected = {{0, 25}, {25, 50}, {50, 75}, {75, 100}};

  ASSERT_EQ(ranges.size(), 4);
  ASSERT_EQ(ranges, expected);
}

TEST(ParallelForTest, MinBlockSize) {
  auto host = CreateTestHostContext(4);
  ParallelFor pfor(host.get());

  latch barrier(3);  // 2 tasks + 1 on-done callback
  mutex mu;
  std::vector<Range> ranges;

  pfor.Execute(
      100, BlockSizes::Min(75),
      [&](size_t begin, size_t end) {
        mutex_lock lock(mu);
        ranges.push_back({begin, end});
        barrier.count_down();
      },
      [&]() { barrier.count_down(); });

  barrier.wait();

  std::sort(ranges.begin(), ranges.end());
  const std::vector<Range> expected = {{0, 75}, {75, 100}};

  ASSERT_EQ(ranges.size(), 2);
  ASSERT_EQ(ranges, expected);
}

TEST(ParallelForTest, ExecuteNestedParallelism) {
  auto host = CreateTestHostContext(4);
  ParallelFor pfor(host.get());

  auto fork = [&](size_t begin, size_t end) -> AsyncValueRef<Range> {
    return host->EnqueueWork([begin, end]() -> Range {
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

  host->Quiesce();
}

}  // namespace tfrt
