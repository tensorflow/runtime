// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file contains unit tests for tfrt::SingleThreadedWorkQueue

#include "tfrt/host_context/concurrent_work_queue.h"

#include <chrono>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace {

TEST(SingleThreadeWorkQueueTest, AsyncValueCompletesOnAnotherThread) {
  std::unique_ptr<ConcurrentWorkQueue> work_queue =
      CreateSingleThreadedWorkQueue();
  std::unique_ptr<HostAllocator> allocator = CreateMallocAllocator();

  HostContext host{{}, std::move(allocator), std::move(work_queue)};
  AsyncValueRef<int> av = MakeConstructedAsyncValueRef<int>(42);

  std::thread thread{[&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    av.SetStateConcrete();
  }};

  host.Await(av.CopyRCRef());
  EXPECT_TRUE(av.IsAvailable());
  thread.join();
}

TEST(SingleThreadeWorkQueueTest,
     AsyncValueCompletesFromWorkAddedByOnAnotherThread) {
  std::unique_ptr<ConcurrentWorkQueue> work_queue =
      CreateSingleThreadedWorkQueue();
  std::unique_ptr<HostAllocator> allocator = CreateMallocAllocator();

  HostContext host{{}, std::move(allocator), std::move(work_queue)};
  AsyncValueRef<int> av = MakeConstructedAsyncValueRef<int>(42);

  std::thread thread{[&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EnqueueWork(&host, [&] { av.SetStateConcrete(); });
  }};

  host.Await(av.CopyRCRef());
  EXPECT_TRUE(av.IsAvailable());
  thread.join();
}

TEST(SingleThreadeWorkQueueTest,
     AsyncValueCompletesFromBlockingWorkAddedByOnAnotherThread) {
  std::unique_ptr<ConcurrentWorkQueue> work_queue =
      CreateSingleThreadedWorkQueue();
  std::unique_ptr<HostAllocator> allocator = CreateMallocAllocator();

  HostContext host{{}, std::move(allocator), std::move(work_queue)};
  AsyncValueRef<int> av = MakeConstructedAsyncValueRef<int>(42);

  std::thread thread{[&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    bool success = EnqueueBlockingWork(&host, [&] { av.SetStateConcrete(); });
    EXPECT_TRUE(success);
  }};

  host.Await(av.CopyRCRef());
  EXPECT_TRUE(av.IsAvailable());
  thread.join();
}

TEST(SingleThreadeWorkQueueTest, AwaitAndDestroy) {
  std::unique_ptr<ConcurrentWorkQueue> work_queue =
      CreateSingleThreadedWorkQueue();

  int num_threads = 8;

  std::vector<RCReference<AsyncValue>> avs;
  for (int i = 0; i < num_threads; ++i) {
    avs.push_back(MakeConstructedAsyncValueRef<int>(42));
  }

  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i] {
      work_queue->AddTask([&, i] { avs[i]->SetStateConcrete(); });
    });
  }

  work_queue->Await(avs);

  // When Await() returns, previous AddTask() calls that setting relevant
  // AsyncValues should all return. So deleting the work queue should be safe
  // now if the implementation is correct.
  work_queue.reset();

  for (int i = 0; i < num_threads; ++i) {
    EXPECT_TRUE(avs[i]->IsAvailable());
    threads[i].join();
  }
}

}  // namespace
}  // namespace tfrt
