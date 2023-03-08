// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Unit tests and benchmarks for TaskPriorityDeque.

#include "task_priority_deque.h"

#include <optional>
#include <random>
#include <thread>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "tfrt/host_context/task_function.h"

namespace tfrt {
namespace {

static constexpr int kNumPriorities = 4;

using TaskPriorityDeque = ::tfrt::internal::TaskPriorityDeque;
using TaskPriority = ::tfrt::internal::TaskPriority;

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

TEST(TaskPriorityDequeTest, QueueCreatedEmpty) {
  TaskPriorityDeque queue;

  ASSERT_TRUE(queue.Empty());
  ASSERT_EQ(queue.Size(), 0);

  ASSERT_EQ(queue.PopFront(), std::nullopt);
  ASSERT_EQ(queue.PopBack(), std::nullopt);
}

TEST(TaskPriorityDequeTest, PushAndPopFrontDefaultPriority) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

  ASSERT_EQ(queue.PushFront(fn.Next(1)), std::nullopt);
  ASSERT_EQ(queue.Size(), 1);
  ASSERT_EQ(fn.Run(queue.PopFront()), 1);
  ASSERT_EQ(queue.Size(), 0);
}

TEST(TaskPriorityDequeTest, PushAndPopFrontWithPriority) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

  ASSERT_EQ(queue.PushFront(fn.Next(1), TaskPriority::kLow), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(2), TaskPriority::kLow), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(3), TaskPriority::kDefault), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(4), TaskPriority::kDefault), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(5), TaskPriority::kHigh), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(6), TaskPriority::kHigh), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(7), TaskPriority::kCritical), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(8), TaskPriority::kCritical), std::nullopt);
  ASSERT_EQ(queue.Size(), 8);

  ASSERT_EQ(fn.Run(queue.PopFront()), 8);
  ASSERT_EQ(fn.Run(queue.PopFront()), 7);
  ASSERT_EQ(fn.Run(queue.PopFront()), 6);
  ASSERT_EQ(fn.Run(queue.PopFront()), 5);
  ASSERT_EQ(fn.Run(queue.PopFront()), 4);
  ASSERT_EQ(fn.Run(queue.PopFront()), 3);
  ASSERT_EQ(fn.Run(queue.PopFront()), 2);
  ASSERT_EQ(fn.Run(queue.PopFront()), 1);
  ASSERT_EQ(queue.Size(), 0);
}

TEST(TaskPriorityDequeTest, PushAndPopBackDefaultPriority) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

  ASSERT_EQ(queue.PushBack(fn.Next(1)), std::nullopt);
  ASSERT_EQ(queue.Size(), 1);
  ASSERT_EQ(fn.Run(queue.PopBack()), 1);
  ASSERT_EQ(queue.Size(), 0);
}

TEST(TaskPriorityDequeTest, PushAndPopBackWithPriority) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

  ASSERT_EQ(queue.PushBack(fn.Next(1), TaskPriority::kLow), std::nullopt);
  ASSERT_EQ(queue.PushBack(fn.Next(2), TaskPriority::kLow), std::nullopt);
  ASSERT_EQ(queue.PushBack(fn.Next(3), TaskPriority::kDefault), std::nullopt);
  ASSERT_EQ(queue.PushBack(fn.Next(4), TaskPriority::kDefault), std::nullopt);
  ASSERT_EQ(queue.PushBack(fn.Next(5), TaskPriority::kHigh), std::nullopt);
  ASSERT_EQ(queue.PushBack(fn.Next(6), TaskPriority::kHigh), std::nullopt);
  ASSERT_EQ(queue.PushBack(fn.Next(7), TaskPriority::kCritical), std::nullopt);
  ASSERT_EQ(queue.PushBack(fn.Next(8), TaskPriority::kCritical), std::nullopt);
  ASSERT_EQ(queue.Size(), 8);

  ASSERT_EQ(fn.Run(queue.PopBack()), 8);
  ASSERT_EQ(fn.Run(queue.PopBack()), 7);
  ASSERT_EQ(fn.Run(queue.PopBack()), 6);
  ASSERT_EQ(fn.Run(queue.PopBack()), 5);
  ASSERT_EQ(fn.Run(queue.PopBack()), 4);
  ASSERT_EQ(fn.Run(queue.PopBack()), 3);
  ASSERT_EQ(fn.Run(queue.PopBack()), 2);
  ASSERT_EQ(fn.Run(queue.PopBack()), 1);
  ASSERT_EQ(queue.Size(), 0);
}

TEST(TaskPriorityDequeTest, PushFrontAndPopBackDefaultPriority) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

  ASSERT_EQ(queue.PushFront(fn.Next(1)), std::nullopt);
  ASSERT_EQ(queue.Size(), 1);
  ASSERT_EQ(fn.Run(queue.PopBack()), 1);
  ASSERT_EQ(queue.Size(), 0);
}

TEST(TaskPriorityDequeTest, PushFrontAndPopBackWithPriority) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

  ASSERT_EQ(queue.PushFront(fn.Next(1), TaskPriority::kLow), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(2), TaskPriority::kLow), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(3), TaskPriority::kDefault), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(4), TaskPriority::kDefault), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(5), TaskPriority::kHigh), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(6), TaskPriority::kHigh), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(7), TaskPriority::kCritical), std::nullopt);
  ASSERT_EQ(queue.PushFront(fn.Next(8), TaskPriority::kCritical), std::nullopt);

  ASSERT_EQ(fn.Run(queue.PopBack()), 7);
  ASSERT_EQ(fn.Run(queue.PopBack()), 8);
  ASSERT_EQ(fn.Run(queue.PopBack()), 5);
  ASSERT_EQ(fn.Run(queue.PopBack()), 6);
  ASSERT_EQ(fn.Run(queue.PopBack()), 3);
  ASSERT_EQ(fn.Run(queue.PopBack()), 4);
  ASSERT_EQ(fn.Run(queue.PopBack()), 1);
  ASSERT_EQ(fn.Run(queue.PopBack()), 2);
  ASSERT_EQ(queue.Size(), 0);
}

TEST(TaskPriorityDequeTest, PushFrontToOverflowDefaultPriority) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

  for (int i = 0; i < TaskPriorityDeque::kCapacity; ++i) {
    ASSERT_EQ(queue.PushFront(fn.Next(i)), std::nullopt);
  }

  auto overflow = queue.PushFront(fn.Next(12345));
  ASSERT_TRUE(overflow.has_value());
  ASSERT_EQ(fn.Run(std::move(overflow)), 12345);

  for (int i = 0; i < TaskPriorityDeque::kCapacity; ++i) {
    ASSERT_EQ(fn.Run(queue.PopBack()), i);
    ASSERT_EQ(queue.Size(), TaskPriorityDeque::kCapacity - i - 1);
  }
}

TEST(TaskPriorityDequeTest, PushFrontToOverflowWithPriority) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

  auto priority = [](int i) -> TaskPriority {
    if (i % kNumPriorities == 0) return TaskPriority::kLow;
    if (i % kNumPriorities == 1) return TaskPriority::kDefault;
    if (i % kNumPriorities == 2) return TaskPriority::kHigh;
    return TaskPriority::kCritical;
  };

  for (int i = 0; i < 4 * TaskPriorityDeque::kCapacity; ++i) {
    ASSERT_EQ(queue.PushFront(fn.Next(i), priority(i)), std::nullopt);
  }

  auto overflow0 = queue.PushFront(fn.Next(12345), TaskPriority::kLow);
  auto overflow1 = queue.PushFront(fn.Next(67891), TaskPriority::kDefault);
  auto overflow2 = queue.PushFront(fn.Next(23456), TaskPriority::kHigh);
  auto overflow3 = queue.PushFront(fn.Next(78910), TaskPriority::kCritical);

  ASSERT_TRUE(overflow0.has_value());
  ASSERT_TRUE(overflow1.has_value());
  ASSERT_TRUE(overflow2.has_value());
  ASSERT_TRUE(overflow3.has_value());

  ASSERT_EQ(fn.Run(std::move(overflow0)), 12345);
  ASSERT_EQ(fn.Run(std::move(overflow1)), 67891);
  ASSERT_EQ(fn.Run(std::move(overflow2)), 23456);
  ASSERT_EQ(fn.Run(std::move(overflow3)), 78910);

  for (int p = 0; p < 4; ++p) {
    for (int i = 0; i < TaskPriorityDeque::kCapacity; ++i) {
      ASSERT_EQ(fn.Run(queue.PopBack()), (4 - p - 1) + 4 * i);
      ASSERT_EQ(queue.Size(), (4 - p) * TaskPriorityDeque::kCapacity - i - 1);
    }
  }
}

TEST(TaskPriorityDequeTest, PushBackToOverflowDefaultPriority) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

  for (int i = 0; i < TaskPriorityDeque::kCapacity; ++i) {
    ASSERT_EQ(queue.PushBack(fn.Next(i)), std::nullopt);
  }

  auto overflow = queue.PushBack(fn.Next(12345));
  ASSERT_TRUE(overflow.has_value());
  ASSERT_EQ(fn.Run(std::move(overflow)), 12345);

  for (int i = 0; i < TaskPriorityDeque::kCapacity; ++i) {
    ASSERT_EQ(fn.Run(queue.PopFront()), i);
  }
}

TEST(TaskPriorityDequeTest, PushBackToOverflowWithPriority) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

  auto priority = [](int i) -> TaskPriority {
    if (i % kNumPriorities == 0) return TaskPriority::kLow;
    if (i % kNumPriorities == 1) return TaskPriority::kDefault;
    if (i % kNumPriorities == 2) return TaskPriority::kHigh;
    return TaskPriority::kCritical;
  };

  for (int i = 0; i < kNumPriorities * TaskPriorityDeque::kCapacity; ++i) {
    ASSERT_EQ(queue.PushBack(fn.Next(i), priority(i)), std::nullopt);
  }

  auto overflow0 = queue.PushBack(fn.Next(12345), TaskPriority::kLow);
  auto overflow1 = queue.PushBack(fn.Next(67891), TaskPriority::kDefault);
  auto overflow2 = queue.PushBack(fn.Next(23456), TaskPriority::kHigh);
  auto overflow3 = queue.PushBack(fn.Next(78901), TaskPriority::kHigh);

  ASSERT_TRUE(overflow0.has_value());
  ASSERT_TRUE(overflow1.has_value());
  ASSERT_TRUE(overflow2.has_value());
  ASSERT_TRUE(overflow3.has_value());

  ASSERT_EQ(fn.Run(std::move(overflow0)), 12345);
  ASSERT_EQ(fn.Run(std::move(overflow1)), 67891);
  ASSERT_EQ(fn.Run(std::move(overflow2)), 23456);
  ASSERT_EQ(fn.Run(std::move(overflow3)), 78901);

  for (int p = 0; p < 4; ++p) {
    for (int i = 0; i < TaskPriorityDeque::kCapacity; ++i) {
      ASSERT_EQ(fn.Run(queue.PopFront()), (4 - p - 1) + 4 * i);
      ASSERT_EQ(queue.Size(), (4 - p) * TaskPriorityDeque::kCapacity - i - 1);
    }
  }
}

// Check that queue correctly reports its emptyness status under contention.
// Worker thread constantly push and pop one task to/from the queue, that
// initially was added a single task.
TEST(TaskPriorityDequeTest, EmptynessCheckSingleWorker) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

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

      auto priority = [](int rng) -> TaskPriority {
        if (rng % kNumPriorities == 0) return TaskPriority::kLow;
        if (rng % kNumPriorities == 1) return TaskPriority::kDefault;
        if (rng % kNumPriorities == 2) return TaskPriority::kHigh;
        return TaskPriority::kCritical;
      };

      if (rng() % 2) {
        ASSERT_EQ(queue.PushFront(fn.Next(1), priority(rng())), std::nullopt);
      } else {
        ASSERT_EQ(queue.PushBack(fn.Next(1), priority(rng())), std::nullopt);
      }

      std::this_thread::yield();

      int choice = rng() % 2;

      if (choice == 0) {
        ASSERT_TRUE(queue.PopFront().has_value());
      } else if (choice == 1) {
        ASSERT_TRUE(queue.PopBack().has_value());
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
TEST(TaskPriorityDequeTest, EmptynessCheckMultipleWorkers) {
  TaskFunctions fn;
  TaskPriorityDeque queue;

  constexpr int kNumIterations = 1 << 14;
  constexpr int kNumWorkers = 10;

  ASSERT_EQ(queue.PushBack(fn.Next(1)), std::nullopt);
  ASSERT_FALSE(queue.Empty());

  std::atomic<int> worker_id = 0;

  struct ScopedLiveWorker {
    ~ScopedLiveWorker() { live_workers--; }
    std::atomic<int>& live_workers;
  };

  std::atomic<int> live_workers = kNumWorkers;

  auto worker = [&]() {
    std::random_device rd;
    std::mt19937 rng(rd());

    int id = worker_id.fetch_add(1);

    ScopedLiveWorker live_worker{live_workers};

    for (int i = 0; i < kNumIterations; ++i) {
      // Queue is never empty before we push a task.
      ASSERT_FALSE(queue.Empty());

      auto priority = [](int rng) -> TaskPriority {
        if (rng % kNumPriorities == 0) return TaskPriority::kLow;
        if (rng % kNumPriorities == 1) return TaskPriority::kDefault;
        if (rng % kNumPriorities == 2) return TaskPriority::kHigh;
        return TaskPriority::kCritical;
      };

      if (id == 0 && rng() % 2 == 0) {
        ASSERT_EQ(queue.PushFront(fn.Next(1), priority(rng())), std::nullopt);
      } else {
        ASSERT_EQ(queue.PushBack(fn.Next(1), priority(rng())), std::nullopt);
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

  TaskPriorityDeque queue;
  for (auto _ : state) {
    for (int i = 0; i < num_tasks; ++i) (void)queue.PushFront({});
    for (int i = 0; i < num_tasks; ++i) (void)queue.PopFront();
  }

  state.SetItemsProcessed(num_tasks * state.iterations());
}

void BM_PushBackAndPopFront(benchmark::State& state) {
  const int num_tasks = state.range(0);

  TaskPriorityDeque queue;
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
