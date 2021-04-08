// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Unit tests and benchmarks for MultiThreadedWorkQueue.

#include <atomic>

#include "gtest/gtest.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {
namespace {

std::unique_ptr<HostContext> CreateTestHostContext(int num_threads) {
  return std::make_unique<HostContext>(
      [](const DecodedDiagnostic&) {}, CreateMallocAllocator(),
      CreateMultiThreadedWorkQueue(num_threads, num_threads));
}

TEST(MultiThreadedWorkQueueTest, PingPong) {
  auto host = CreateTestHostContext(4);

  std::atomic<int64_t> last_executed_task = -1;
  const int64_t num_tasks = 10000;

  // Enqueue ping-pongs a single task between blocking and non-blocking work
  // queues `num_tasks` number of times.
  llvm::unique_function<void(int64_t)> enqueue;
  enqueue = [&](int64_t n) {
    if (n >= num_tasks) return;
    last_executed_task = n;

    if (n % 2 == 0) {
      EnqueueWork(host.get(), [&, n]() { enqueue(n + 1); });
    } else {
      bool enqueued =
          EnqueueBlockingWork(host.get(), [&, n]() { enqueue(n + 1); });
      if (!enqueued) last_executed_task = -100;
    }
  };

  // Check that Quiesce returns only when all tasks are completed.
  enqueue(0);
  host->Quiesce();
  ASSERT_EQ(last_executed_task, num_tasks - 1);
}

}  // namespace
}  // namespace tfrt
