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

// Unit test for TFRT RequestDeadlineTracker.

#include "tfrt/host_context/request_deadline_tracker.h"

#include <chrono>

#include "gtest/gtest.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {
namespace {

using namespace std::chrono_literals;  // NOLINT

std::unique_ptr<HostContext> CreateTestHostContext(int num_threads) {
  return std::make_unique<HostContext>(
      [](const DecodedDiagnostic&) {}, CreateMallocAllocator(),
      CreateMultiThreadedWorkQueue(num_threads, num_threads));
}

// Test RequestDeadlineTracker.
TEST(RequestDeadlineTrackerTest, CancelRequest) {
  std::unique_ptr<HostContext> host = CreateTestHostContext(1);
  RequestDeadlineTracker req_deadline_tracker{host.get()};
  Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(host.get(), /*resource_context=*/nullptr).build();
  ASSERT_FALSE(!req_ctx);

  std::chrono::system_clock::time_point deadline =
      std::chrono::system_clock::now() + 1s;
  req_deadline_tracker.CancelRequestOnDeadline(deadline, *req_ctx);

  std::this_thread::sleep_for(2s);

  // Check if RequestContext's is_cancelled_ flag is set.
  ASSERT_TRUE((*req_ctx)->IsCancelled());
}

}  // namespace
}  // namespace tfrt
