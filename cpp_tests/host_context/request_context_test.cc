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

// Unit test for TFRT RequestContext.

#include "gtest/gtest.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {
namespace {

std::unique_ptr<HostContext> CreateTestHostContext() {
  return std::make_unique<HostContext>([](const DecodedDiagnostic&) {},
                                       CreateMallocAllocator(),
                                       CreateSingleThreadedWorkQueue());
}

TEST(RequestContextTest, Id) {
  auto host = CreateTestHostContext();
  ResourceContext resource_context;
  auto expected_request_context =
      RequestContextBuilder(host.get(), &resource_context, /*id=*/0xdeadbeef)
          .build();
  ASSERT_FALSE(!expected_request_context);
  EXPECT_EQ(expected_request_context.get()->id(), 0xdeadbeef);
}

TEST(RequestContextTest, ContextData) {
  auto host = CreateTestHostContext();
  ResourceContext resource_context;
  RequestContextBuilder request_context_builder(host.get(), &resource_context,
                                                /*id=*/0xdeadbeef);
  struct TestData {
    int value;
  };
  request_context_builder.context_data().insert(TestData{100});

  auto expected_request_context = std::move(request_context_builder).build();
  ASSERT_FALSE(!expected_request_context);

  EXPECT_NE(expected_request_context.get()->GetDataIfExists<TestData>(),
            nullptr);
  EXPECT_EQ(expected_request_context.get()->GetData<TestData>().value, 100);
  EXPECT_EQ(expected_request_context.get()->GetDataIfExists<int>(), nullptr);
}

}  // namespace
}  // namespace tfrt
