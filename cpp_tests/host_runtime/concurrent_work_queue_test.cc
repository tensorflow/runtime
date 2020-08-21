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

//===- concurrent_work_queue_test.cc --------------------------------------===//
//
// This file contains unit tests for tfrt::SingleThreadedWorkQueue
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/concurrent_work_queue.h"

#include <chrono>
#include <thread>

#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
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
  AsyncValueRef<int> av = MakeConstructedAsyncValueRef<int>(&host, 42);

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
  AsyncValueRef<int> av = MakeConstructedAsyncValueRef<int>(&host, 42);

  std::thread thread{[&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    host.EnqueueWork([&] { av.SetStateConcrete(); });
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
  AsyncValueRef<int> av = MakeConstructedAsyncValueRef<int>(&host, 42);

  std::thread thread{[&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    bool success = host.EnqueueBlockingWork([&] { av.SetStateConcrete(); });
    EXPECT_TRUE(success);
  }};

  host.Await(av.CopyRCRef());
  EXPECT_TRUE(av.IsAvailable());
  thread.join();
}

}  // namespace
}  // namespace tfrt
