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

// Tests for tfrt::ThreadLocal implementation.

#include "tfrt/support/thread_local.h"

#include <thread>

#include "gtest/gtest.h"
#include "tfrt/support/latch.h"

namespace tfrt {
namespace {

// Make WorkerId non-default-constructible and move-only to check that
// TheadLocal storage can properly handle such types.
struct WorkerId {
  explicit WorkerId(int id) : id(id) {}
  WorkerId(const WorkerId&) = default;
  WorkerId& operator=(const WorkerId&) = default;
  WorkerId(WorkerId&&) noexcept = default;
  WorkerId& operator=(WorkerId&&) noexcept = default;
  int id;
};

struct WorkerIdGenerator {
  explicit WorkerIdGenerator(int init) : counter(init) {}
  WorkerId Construct() { return WorkerId(counter.fetch_add(1)); }
  std::atomic<int> counter;
};

using CurrentWorkerId = ThreadLocal<WorkerId, WorkerIdGenerator>;

TEST(ThreadLocalTest, WorkerIdTest) {
  static constexpr int kInitWorkerId = 123;

  auto test = [](int num_threads, int capacity) {
    CurrentWorkerId worker_id(capacity, kInitWorkerId);

    // Wait until all threads are started.
    latch ready(num_threads);

    // Spin in the worker threads to force contention on the initial worker id
    // assignment (thread local element creation).
    std::atomic<bool> run{false};

    // Count the number of times when worker id changed for a thread. This
    // should never happen, once worker id assigned to a thread, it should
    // always stay the same.
    std::atomic<int> errors{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&]() {
        ready.count_down();
        while (!run.load()) {
        }

        // Get a unique worker id.
        WorkerId& id = worker_id.Local();

        // Check that every other call returns the same worker id.
        for (int i = 0; i < 1000; ++i) {
          if (id.id != worker_id.Local().id) errors.fetch_add(1);
        }
      });
    }

    ready.wait();     // wait for threads to arrive at run spin loop
    run.store(true);  // allow threads to run the test

    for (auto& thread : threads) thread.join();

    ASSERT_EQ(errors, 0);
    worker_id.ForEach([&](std::thread::id, WorkerId& id) {
      ASSERT_GE(id.id, kInitWorkerId);
      ASSERT_LT(id.id, kInitWorkerId + num_threads);
    });
  };

  for (int num_threads : {2, 4, 8, 16}) {
    for (int capacity : {0, 2, 4, 8, 16, 32, 64}) {
      test(num_threads, capacity);
    }
  }
}

}  // namespace
}  // namespace tfrt
