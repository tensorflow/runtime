// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Unit tests and benchmarks for TaskDeque.

#include "task_deque.h"

#include <optional>
#include <random>
#include <thread>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "tfrt/host_context/task_function.h"

namespace tfrt {
namespace {

using TaskDeque = ::tfrt::internal::TaskDeque;

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

TEST(TaskDequeTest, QueueCreatedEmpty) {
  TaskDeque queue;

  ASSERT_TRUE(queue.Empty());
  ASSERT_EQ(queue.Size(), 0);

  ASSERT_EQ(queue.PopFront(), std::nullopt);
  ASSERT_EQ(queue.PopBack(), std::nullopt);

  std::vector<TaskFunction> tasks;
  ASSERT_EQ(queue.PopBackHalf(&tasks), 0u);
  ASSERT_EQ(tasks.size(), 0u);
}

TEST(TaskDequeTest, PushAndPopFront) {
  TaskFunctions fn;
  TaskDeque queue;

  ASSERT_EQ(queue.PushFront(fn.Next(1)), std::nullopt);
  ASSERT_EQ(queue.Size(), 1);
  ASSERT_EQ(fn.Run(queue.PopFront()), 1);
  ASSERT_EQ(queue.Size(), 0);
}

TEST(TaskDequeTest, PushAndPopBack) {
  TaskFunctions fn;
  TaskDeque queue;

  ASSERT_EQ(queue.PushBack(fn.Next(1)), std::nullopt);
  ASSERT_EQ(queue.Size(), 1);
  ASSERT_EQ(fn.Run(queue.PopBack()), 1);
  ASSERT_EQ(queue.Size(), 0);
}

TEST(TaskDequeTest, PushFrontAndPopBack) {
  TaskFunctions fn;
  TaskDeque queue;

  ASSERT_EQ(queue.PushFront(fn.Next(1)), std::nullopt);
  ASSERT_EQ(queue.Size(), 1);
  ASSERT_EQ(fn.Run(queue.PopBack()), 1);
  ASSERT_EQ(queue.Size(), 0);
}

TEST(TaskDequeTest, PushFrontToOverflow) {
  TaskFunctions fn;
  TaskDeque queue;

  for (int i = 0; i < TaskDeque::kCapacity; ++i) {
    ASSERT_EQ(queue.PushFront(fn.Next(i)), std::nullopt);
  }

  auto overflow = queue.PushFront(fn.Next(12345));
  ASSERT_TRUE(overflow.has_value());
  ASSERT_EQ(fn.Run(std::move(overflow)), 12345);

  for (int i = 0; i < TaskDeque::kCapacity; ++i) {
    ASSERT_EQ(fn.Run(queue.PopBack()), i);
  }
}

TEST(TaskDequeTest, PushBackToOverflow) {
  TaskFunctions fn;
  TaskDeque queue;

  for (int i = 0; i < TaskDeque::kCapacity; ++i) {
    ASSERT_EQ(queue.PushBack(fn.Next(i)), std::nullopt);
  }

  auto overflow = queue.PushFront(fn.Next(12345));
  ASSERT_TRUE(overflow.has_value());
  ASSERT_EQ(fn.Run(std::move(overflow)), 12345);

  for (int i = 0; i < TaskDeque::kCapacity; ++i) {
    ASSERT_EQ(fn.Run(queue.PopFront()), i);
  }
}

TEST(TaskDequeTest, PopBackHalf) {
  TaskFunctions fn;
  TaskDeque queue;

  for (int i = 0; i < TaskDeque::kCapacity; ++i) {
    ASSERT_EQ(queue.PushBack(fn.Next(i)), std::nullopt);
  }

  std::vector<TaskFunction> half0;
  std::vector<TaskFunction> half1;

  queue.PopBackHalf(&half0);
  ASSERT_EQ(half0.size(), TaskDeque::kCapacity / 2);

  queue.PopBackHalf(&half1);
  ASSERT_EQ(half1.size(), TaskDeque::kCapacity / 4);

  for (int i = 0; i < half0.size(); ++i) {
    ASSERT_EQ(fn.Run({std::move(half0[i])}), TaskDeque::kCapacity / 2 + i);
  }

  for (int i = 0; i < half1.size(); ++i) {
    ASSERT_EQ(fn.Run({std::move(half1[i])}), TaskDeque::kCapacity / 4 + i);
  }

  for (int i = 0; i < TaskDeque::kCapacity / 4; ++i) {
    ASSERT_EQ(fn.Run(queue.PopFront()), i);
  }
}

// Check that queue correctly reports its emptyness status under contention.
// Worker thread constantly push and pop one task to/from the queue, that
// initially was added a single task.
TEST(TaskDequeTest, EmptynessCheckSingleWorker) {
  TaskFunctions fn;
  TaskDeque queue;

  constexpr int kNumIterations = 1 << 16;

  std::atomic<bool> done = false;

  ASSERT_EQ(queue.PushBack(fn.Next(1)), std::nullopt);
  ASSERT_FALSE(queue.Empty());

  auto worker = [&]() {
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int i = 0; i < kNumIterations; ++i) {
      // Queue is never empty before we push a task.
      ASSERT_FALSE(queue.Empty());

      if (rng() % 2) {
        ASSERT_EQ(queue.PushFront(fn.Next(1)), std::nullopt);
      } else {
        ASSERT_EQ(queue.PushBack(fn.Next(1)), std::nullopt);
      }

      std::this_thread::yield();

      int choice = rng() % 3;

      if (choice == 0) {
        ASSERT_TRUE(queue.PopFront().has_value());
      } else if (choice == 1) {
        ASSERT_TRUE(queue.PopBack().has_value());
      } else if (choice == 2) {
        std::vector<TaskFunction> tasks;
        queue.PopBackHalf(&tasks);
        ASSERT_EQ(tasks.size(), 1);
      }

      // And it's never empty after we pop a task.
      ASSERT_FALSE(queue.Empty());
    }

    done = true;
  };

  // Start a worker thread.
  std::thread worker_thread(worker);

  while (!done) {
    ASSERT_FALSE(queue.Empty());
    unsigned size = queue.Size();
    ASSERT_GE(size, 1);
    ASSERT_LE(size, 2);
  }

  ASSERT_EQ(queue.Size(), 1);
  ASSERT_TRUE(queue.PopBack().has_value());

  // Wait for worker thread completion.
  worker_thread.join();
}

// Check that queue correctly reports its emptyness status under contention.
// With a multiple worker threads we can use the front of the queue only from
// one thread, all other threads must use the back of the queue.
TEST(TaskDequeTest, EmptynessCheckMultipleWorkers) {
  TaskFunctions fn;
  TaskDeque queue;

  constexpr int kNumIterations = 1 << 14;
  constexpr int kNumWorkers = 10;

  ASSERT_EQ(queue.PushBack(fn.Next(1)), std::nullopt);
  ASSERT_FALSE(queue.Empty());

  struct ScopedLiveWorker {
    ~ScopedLiveWorker() { live_workers--; }
    std::atomic<int>& live_workers;
  };

  std::atomic<int> live_workers = kNumWorkers;

  std::atomic<int> worker_id = 0;

  auto worker = [&]() {
    std::random_device rd;
    std::mt19937 rng(rd());

    int id = worker_id.fetch_add(1);

    ScopedLiveWorker live_worker{live_workers};

    for (int i = 0; i < kNumIterations; ++i) {
      // Queue is never empty before we push a task.
      ASSERT_FALSE(queue.Empty());

      if (id == 0 && rng() % 2 == 0) {
        ASSERT_EQ(queue.PushFront(fn.Next(1)), std::nullopt);
      } else {
        ASSERT_EQ(queue.PushBack(fn.Next(1)), std::nullopt);
      }

      std::this_thread::yield();

      // PopFront and PopBack can return empty optional if atomic exchange
      // operation failed under contention, but they should be successfull
      // after a reasonable number of iterations.
      std::optional<TaskFunction> popped = std::nullopt;
      if (id == 0 && rng() % 2 == 0) {
        for (int i = 0; i < 100 && !popped.has_value(); ++i)
          popped = queue.PopFront();
      } else {
        for (int i = 0; i < 100 && !popped.has_value(); ++i)
          popped = queue.PopBack();
      }
      ASSERT_TRUE(popped.has_value());

      // And it's never empty after we pop a task.
      ASSERT_FALSE(queue.Empty());
    }
  };

  // Start worker threads.
  std::vector<std::thread> worker_threads;
  for (int i = 0; i < kNumWorkers; ++i) worker_threads.emplace_back(worker);

  while (live_workers > 0) {
    ASSERT_FALSE(queue.Empty());
    unsigned size = queue.Size();
    ASSERT_GE(size, 1);
    ASSERT_LE(size, 1 + kNumWorkers);
  }

  ASSERT_EQ(queue.Size(), 1);
  ASSERT_TRUE(queue.PopBack().has_value());

  // Wait for worker threads completion.
  for (auto& worker_thread : worker_threads) worker_thread.join();
}

void BM_PushFrontAndPopFront(benchmark::State& state) {
  const int num_tasks = state.range(0);

  TaskDeque queue;
  for (auto _ : state) {
    for (int i = 0; i < num_tasks; ++i) (void)queue.PushFront({});
    for (int i = 0; i < num_tasks; ++i) (void)queue.PopFront();
  }

  state.SetItemsProcessed(num_tasks * state.iterations());
}

void BM_PushBackAndPopFront(benchmark::State& state) {
  const int num_tasks = state.range(0);

  TaskDeque queue;
  for (auto _ : state) {
    for (int i = 0; i < num_tasks; ++i) (void)queue.PushBack({});
    for (int i = 0; i < num_tasks; ++i) (void)queue.PopFront();
  }

  state.SetItemsProcessed(num_tasks * state.iterations());
}

BENCHMARK(BM_PushFrontAndPopFront)->Arg(1)->Arg(10)->Arg(100)->Arg(1000);
BENCHMARK(BM_PushBackAndPopFront)->Arg(1)->Arg(10)->Arg(100)->Arg(1000);

}  // namespace
}  // namespace tfrt
