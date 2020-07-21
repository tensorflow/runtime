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

//===- distributed_context_test.cc ------------------------------*- C++ -*-===//
//
// Unit test for DistributedContext.
//
//===----------------------------------------------------------------------===//

#include "tfrt/distributed_runtime/distributed_context.h"

#include <atomic>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {

namespace {

const char* kMockCommunicatorType = "mock_type";
const char* kMockCommunicatorName = "mock_name";

class MockCommunicator : public FabricCommunicator {
 public:
  explicit MockCommunicator(llvm::StringRef name)
      : FabricCommunicator(name,
                           /*distributed_context=*/nullptr) {}
  MOCK_METHOD(void, Send,
              (InstanceKey instance_key, HostId destination,
               llvm::StringRef payload),
              (override));
};

DistributedContextConfiguration GetSampleConfiguration() {
  HostConfiguration host_config{
      /*addresses=*/{"addr0", "addr1", "addr2", "addr3"}, /*rank=*/1};
  FabricCommunicatorConfiguration mock_communicator_config{
      kMockCommunicatorType, host_config};
  CollectiveGroup group0{/*name=*/"group0", /*members=*/{0, 1}};
  CollectiveGroup group1{/*name=*/"group1", /*members=*/{1, 2, 3}};
  DistributedContextConfiguration context_config{
      mock_communicator_config,
      /*collective_groups=*/{group0, group1}};
  return context_config;
}

TEST(DistributedContext, CreateFabricCommunicator) {
  auto configuration = GetSampleConfiguration();
  DistributedContext dist_context(/*host_context=*/nullptr, configuration);

  EXPECT_EQ(dist_context.GetOrCreateFabricCommunicator(), nullptr);

  auto mock_factory = [](DistributedContext* distributed_context,
                         const FabricCommunicatorConfiguration& configuration)
      -> FabricCommunicator* {
    return new MockCommunicator(kMockCommunicatorName);
  };
  DistributedContext::RegisterFabricCommunicatorType(kMockCommunicatorType,
                                                     mock_factory);

  FabricCommunicator* fabric_communicator =
      dist_context.GetOrCreateFabricCommunicator();
  EXPECT_NE(fabric_communicator, nullptr);
  EXPECT_EQ(dist_context.GetOrCreateFabricCommunicator(), fabric_communicator);
}

TEST(DistributedContext, GetHostContext) {
  auto diag_handler = [](const DecodedDiagnostic&) {};
  HostContext host_context(diag_handler, tfrt::CreateMallocAllocator(),
                           tfrt::CreateMultiThreadedWorkQueue(
                               /*num_threads=*/4,
                               /*num_blocking_threads=*/64));

  auto configuration = GetSampleConfiguration();
  DistributedContext dist_context(&host_context, configuration);

  ASSERT_EQ(dist_context.GetHostContext(), &host_context);
}

TEST(DistributedContext, GetCollectiveGroup) {
  auto diag_handler = [](const DecodedDiagnostic&) {};
  HostContext host_context(diag_handler, tfrt::CreateMallocAllocator(),
                           tfrt::CreateMultiThreadedWorkQueue(
                               /*num_threads=*/4,
                               /*num_blocking_threads=*/64));

  auto configuration = GetSampleConfiguration();
  DistributedContext dist_context(&host_context, configuration);

  auto group1 = dist_context.GetCollectiveGroup("group1");
  EXPECT_EQ(group1.name, "group1");
}

}  // namespace
}  // namespace tfrt
