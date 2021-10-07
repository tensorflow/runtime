// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Unit tests and benchmarks for NonBlockingWorkQueue.

#include "non_blocking_work_queue.h"

#include "benchmark/benchmark.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/support/latch.h"
#include "tfrt/support/thread_environment.h"

namespace tfrt {
namespace {

using WorkQueue = ::tfrt::internal::NonBlockingWorkQueue<ThreadingEnvironment>;

// Benchmark work queue throughput.
//
// Submit `num_producers` tasks to `producer` work queue, each submitting
// `num_tasks` no-op tasks into the `worker` work queue.
//
// Mesures the time to complete all the tasks inside the `worker` queue.
void NoOp(WorkQueue& producer, WorkQueue& worker, benchmark::State& state) {
  const int num_producers = state.range(0);
  const int num_tasks = state.range(1);

  for (auto _ : state) {
    ::tfrt::latch latch(2 * num_producers);

    std::atomic<int>* counters = new std::atomic<int>[num_producers];
    for (int i = 0; i < num_producers; ++i) counters[i] = num_tasks;

    // Submit `num_producers` tasks to `producer` queue, each submitting
    // `num_tasks` to `worker` queue.
    for (int i = 0; i < num_producers; ++i) {
      producer.AddTask(TaskFunction([&, i] {
        for (int j = 0; j < num_tasks; ++j) {
          worker.AddTask(TaskFunction([&, i]() {
            if (counters[i].fetch_sub(1) == 1) latch.count_down();
          }));
        }
        latch.count_down();
      }));
    }

    latch.wait();
    delete[] counters;
  }

  state.SetItemsProcessed(num_producers * num_tasks * state.iterations());
}

#define BM_Run(FN, producer_threads, worker_threads)                 \
  static void BM_##FN##_tpool_##producer_threads##x##worker_threads( \
      benchmark::State& state) {                                     \
    auto qstate = std::make_unique<internal::QuiescingState>();      \
    WorkQueue producer(qstate.get(), producer_threads);              \
    WorkQueue worker(qstate.get(), worker_threads);                  \
    FN(producer, worker, state);                                     \
  }                                                                  \
  BENCHMARK(BM_##FN##_tpool_##producer_threads##x##worker_threads)   \
      ->UseRealTime()

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
