// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Unit tests and benchmarks for BlockingWorkQueue.

#include "blocking_work_queue.h"

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "llvm/Support/FormatVariadic.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/support/latch.h"
#include "tfrt/support/thread_environment.h"

namespace tfrt {
namespace {

using WorkQueue = ::tfrt::internal::BlockingWorkQueue<ThreadingEnvironment>;

TEST(BlockingWorkQueueTest, RejectRunBlockingTask) {
  auto quiescing_state = std::make_unique<internal::QuiescingState>();
  WorkQueue work_queue(quiescing_state.get(), 2, 0);
  auto rejected = work_queue.RunBlockingTask({});
  ASSERT_TRUE(rejected.hasValue());
}

TEST(BlockingWorkQueueTest, RunBlockingTask) {
  auto quiescing_state = std::make_unique<internal::QuiescingState>();
  WorkQueue work_queue(quiescing_state.get(), 2, 2);

  latch barrier(1);
  latch executed(2);

  auto task = [&]() -> TaskFunction {
    return TaskFunction([&]() {
      barrier.wait();
      executed.count_down();
    });
  };

  auto t1 = work_queue.RunBlockingTask(task());
  ASSERT_FALSE(t1.hasValue());
  auto t2 = work_queue.RunBlockingTask(task());
  ASSERT_FALSE(t2.hasValue());
  auto t3 = work_queue.RunBlockingTask(task());
  ASSERT_TRUE(t3.hasValue());  // rejected

  // Let the tasks start.
  barrier.count_down();
  // Wait for completion.
  executed.wait();
}

TEST(BlockingWorkQueueTest, Quiescing) {
  auto quiescing_state = std::make_unique<internal::QuiescingState>();
  WorkQueue work_queue(quiescing_state.get(), 2, 2);

  auto quiescing = internal::Quiescing::Start(quiescing_state.get());

  latch barrier(1);
  latch executed(4);

  auto task = [&]() -> TaskFunction {
    return TaskFunction([&]() {
      barrier.wait();
      executed.count_down();
    });
  };

  auto t1 = work_queue.EnqueueBlockingTask(task());
  ASSERT_FALSE(t1.hasValue());
  auto t2 = work_queue.EnqueueBlockingTask(task());
  ASSERT_FALSE(t2.hasValue());
  auto t3 = work_queue.RunBlockingTask(task());
  ASSERT_FALSE(t3.hasValue());
  auto t4 = work_queue.RunBlockingTask(task());
  ASSERT_FALSE(t4.hasValue());
  auto t5 = work_queue.RunBlockingTask(task());
  ASSERT_TRUE(t5.hasValue());  // rejected

  ASSERT_TRUE(quiescing.HasPendingTasks());

  // Let the tasks start.
  barrier.count_down();
  // Wait for completion.
  executed.wait();

  // Verify that pending task counter was not attached to rejected 't5'.
  ASSERT_FALSE(quiescing.HasPendingTasks());
}

// -------------------------------------------------------------------------- //
// Performance benchmarks.
// -------------------------------------------------------------------------- //

// Benchmark work queue throughput.
//
// Submit `num_producers` tasks to `producer` work queue, each submitting
// `num_tasks` no-op tasks into the `worker` work queue.
//
// Mesures the time to complete all the tasks inside the `worker` queue.
void NoOp(WorkQueue& producer, WorkQueue& worker, benchmark::State& state) {
  const int num_producers = state.range(0);
  const int num_tasks = state.range(1);

  std::atomic<int> num_overflow = 0;

  for (auto _ : state) {
    ::tfrt::latch latch(2 * num_producers);

    std::atomic<int>* counters = new std::atomic<int>[num_producers];
    for (int i = 0; i < num_producers; ++i) counters[i] = num_tasks;

    // Submit `num_producers` tasks to `producer` queue, each submitting
    // `num_tasks` to `worker` queue.
    for (int i = 0; i < num_producers; ++i) {
      auto producer_overflow =
          producer.EnqueueBlockingTask(TaskFunction([&, i] {
            for (int j = 0; j < num_tasks; ++j) {
              auto worker_overflow =
                  worker.EnqueueBlockingTask(TaskFunction([&, i]() {
                    if (counters[i].fetch_sub(1) == 1) latch.count_down();
                  }));

              if (worker_overflow) {
                (*worker_overflow)();
                num_overflow++;
              }
            }
            latch.count_down();
          }));
      if (producer_overflow) (*producer_overflow)();
    }

    latch.wait();
    delete[] counters;
  }

  // WARN: Unique run-to-run labels breaks benchy.
  // std::string label =
  //     llvm::formatv("overflow: {0} / {1}", num_overflow,
  //                   num_producers * num_tasks * state.iterations());
  // state.SetLabel(label);
  state.SetItemsProcessed(num_producers * num_tasks * state.iterations());
}

#define BM_Run(FN, producer_threads, worker_threads)                 \
  static void BM_##FN##_tpool_##producer_threads##x##worker_threads( \
      benchmark::State& state) {                                     \
    BenchmarkUseRealTime();                                          \
    auto qstate = std::make_unique<internal::QuiescingState>();      \
    WorkQueue producer(qstate.get(), producer_threads);              \
    WorkQueue worker(qstate.get(), worker_threads);                  \
    FN(producer, worker, state);                                     \
  }                                                                  \
  BENCHMARK(BM_##FN##_tpool_##producer_threads##x##worker_threads)

#define BM_NoOp(producer_threads, worker_threads) \
  BM_Run(NoOp, producer_threads, worker_threads)  \
      ->ArgPair(10, 10)                           \
      ->ArgPair(10, 100)                          \
      ->ArgPair(10, 1000)                         \
      ->ArgPair(10, 10000)                        \
      ->ArgPair(100, 10)                          \
      ->ArgPair(100, 100)                         \
      ->ArgPair(100, 1000)

BM_NoOp(4, 4);
BM_NoOp(8, 8);
BM_NoOp(16, 16);
BM_NoOp(32, 32);

}  // namespace
}  // namespace tfrt
