// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Unit tests and benchmarks for TaskQueue.

#include "task_queue.h"

#include <optional>
#include <random>
#include <thread>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "tfrt/host_context/task_function.h"

namespace tfrt {
namespace {

using TaskQueue = ::tfrt::internal::TaskQueue;

// Helper class to create TaskFunction with an observable side effect.
struct TaskFunctions {
  TaskFunction Next(int value) {
    return TaskFunction([this, value]() { this->value = value; });
  }

  int Run(std::optional<TaskFunction> task) {
    if (!task.has_value()) return -1;
    (*task)();
    return value;
  }

  int value = -1;
};

TEST(TaskQueueTest, QueueCreatedEmpty) {
  TaskQueue queue;

  ASSERT_TRUE(queue.Empty());
  ASSERT_EQ(queue.Size(), 0);
  ASSERT_EQ(queue.PopBack(), std::nullopt);
}

TEST(TaskQueueTest, PushAndPop) {
  TaskFunctions fn;
  TaskQueue queue;

  ASSERT_EQ(queue.PushFront(fn.Next(1)), std::nullopt);
  ASSERT_EQ(queue.Size(), 1);
  ASSERT_EQ(fn.Run(queue.PopBack()), 1);
  ASSERT_EQ(queue.PopBack(), std::nullopt);
  ASSERT_EQ(queue.Size(), 0);
}

TEST(TaskQueueTest, PushFrontToOverflow) {
  TaskFunctions fn;
  TaskQueue queue;

  for (int i = 0; i < TaskQueue::kCapacity; ++i) {
    ASSERT_EQ(queue.PushFront(fn.Next(i)), std::nullopt);
  }

  auto overflow = queue.PushFront(fn.Next(12345));
  ASSERT_TRUE(overflow.has_value());
  ASSERT_EQ(fn.Run(std::move(overflow)), 12345);

  for (int i = 0; i < TaskQueue::kCapacity; ++i) {
    ASSERT_EQ(fn.Run(queue.PopBack()), i);
  }

  ASSERT_EQ(queue.PopBack(), std::nullopt);
  ASSERT_EQ(queue.Size(), 0);
}

// Check that queue correctly reports its emptyness status under contention.
TEST(TaskQueueTest, EmptynessCheckMultipleWorkers) {
  TaskFunctions fn;
  TaskQueue queue;

  constexpr int kNumIterations = 1 << 14;
  constexpr int kNumWorkers = 10;

  std::atomic<int> live_workers = kNumWorkers;

  ASSERT_EQ(queue.PushFront(fn.Next(1)), std::nullopt);
  ASSERT_FALSE(queue.Empty());

  // Failed asserts inside a worker thread leads to test deadlock.
  std::atomic<int> pre_empty = 0;
  std::atomic<int> post_empty = 0;

  auto worker = [&]() {
    for (int i = 0; i < kNumIterations; ++i) {
      // Queue is never empty before we push a task.
      pre_empty.fetch_or(queue.Empty());

      // Under contention queue might spuriously fail to push a new task.
      auto overflow = queue.PushFront(fn.Next(1));
      if (overflow.has_value()) continue;

      std::this_thread::yield();

      // Pop back might be empty if concurrent PushFront updated front
      // index and ackquired a storage element, but did not update the state.
      std::optional<TaskFunction> task = queue.PopBack();
      while (!task.has_value()) task = queue.PopBack();

      // And it's never empty after we pop a task.
      post_empty.fetch_or(queue.Empty());
    }

    live_workers--;
  };

  // Start worker threads.
  std::vector<std::thread> worker_threads;
  for (int i = 0; i < kNumWorkers; ++i) worker_threads.emplace_back(worker);

  while (live_workers > 0) {
    ASSERT_EQ(pre_empty, 0);
    ASSERT_EQ(post_empty, 0);

    ASSERT_FALSE(queue.Empty());
    unsigned size = queue.Size();
    ASSERT_GE(size, 1);
    ASSERT_LE(size, 1 + kNumWorkers);
  }

  ASSERT_EQ(pre_empty, 0);
  ASSERT_EQ(post_empty, 0);

  ASSERT_EQ(queue.Size(), 1);
  ASSERT_TRUE(queue.PopBack().has_value());

  // Wait for worker threads completion.
  for (auto& worker_thread : worker_threads) worker_thread.join();
}

void BM_PushAndPop(benchmark::State& state) {
  const int num_tasks = state.range(0);

  TaskQueue queue;
  for (auto _ : state) {
    for (int i = 0; i < num_tasks; ++i) (void)queue.PushFront({});
    for (int i = 0; i < num_tasks; ++i) (void)queue.PopBack();
  }

  state.SetItemsProcessed(num_tasks * state.iterations());
}

BENCHMARK(BM_PushAndPop)->Arg(1)->Arg(10)->Arg(100)->Arg(1000);

}  // namespace
}  // namespace tfrt
