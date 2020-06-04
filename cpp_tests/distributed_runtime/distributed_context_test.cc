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
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {

namespace {

const char* kMockCommunicatorType = "mock_type";
const char* kMockCommunicatorName1 = "mock_1";
const char* kMockCommunicatorName2 = "mock_2";
const char* kInvalidCommunicatorName = "invalid";

class MockCommunicator : public FabricCommunicator {
 public:
  explicit MockCommunicator(llvm::StringRef name)
      : FabricCommunicator(name,
                           /*distributed_context=*/nullptr) {}
  MOCK_METHOD(void, Send,
              (InstanceKey instance_key, Rank destination,
               llvm::ArrayRef<uint8_t> payload),
              (override));
};

DistributedContextConfiguration GetSampleConfiguration() {
  FabricCommunicatorConfiguration mock_communicator_config{
      kMockCommunicatorType, {}};
  FabricCommunicatorConfiguration invalid_communicator_config{"invalid_type",
                                                              {}};
  DistributedContextConfiguration context_config;
  context_config.communicators.insert(
      {kMockCommunicatorName1, mock_communicator_config});
  context_config.communicators.insert(
      {kMockCommunicatorName2, mock_communicator_config});
  context_config.communicators.insert(
      {kInvalidCommunicatorName, invalid_communicator_config});
  return context_config;
}

TEST(DistributedContext, CreateFabricCommunicator) {
  auto configuration = GetSampleConfiguration();
  DistributedContext dist_context(/*host_context=*/nullptr, configuration);

  EXPECT_EQ(dist_context.GetOrCreateFabricCommunicator("wrong_name"), nullptr);
  EXPECT_EQ(dist_context.GetOrCreateFabricCommunicator(kMockCommunicatorName1),
            nullptr);

  auto mock_factory = [](llvm::StringRef name,
                         DistributedContext* distributed_context,
                         const FabricCommunicatorConfiguration& configuration)
      -> FabricCommunicator* { return new MockCommunicator(name); };
  DistributedContext::RegisterFabricCommunicatorType(kMockCommunicatorType,
                                                     mock_factory);

  auto communicator_mock_1 =
      dist_context.GetOrCreateFabricCommunicator(kMockCommunicatorName1);
  EXPECT_NE(communicator_mock_1, nullptr);
  EXPECT_EQ(communicator_mock_1->GetFabricCommunicatorName(),
            kMockCommunicatorName1);
  EXPECT_EQ(dist_context.GetOrCreateFabricCommunicator(kMockCommunicatorName1),
            communicator_mock_1);

  EXPECT_EQ(
      dist_context.GetOrCreateFabricCommunicator(kInvalidCommunicatorName),
      nullptr);

  auto communicator_mock_2 =
      dist_context.GetOrCreateFabricCommunicator(kMockCommunicatorName2);
  EXPECT_NE(communicator_mock_2, nullptr);
  EXPECT_EQ(communicator_mock_2->GetFabricCommunicatorName(),
            kMockCommunicatorName2);
}

TEST(DistributedContext, GetHostContext) {
  auto configuration = GetSampleConfiguration();

  auto diag_handler = [](const DecodedDiagnostic&) {};
  HostContext host_context(diag_handler, tfrt::CreateMallocAllocator(),
                           tfrt::CreateMultiThreadedWorkQueue(
                               /*num_threads=*/4,
                               /*num_blocking_threads=*/64));

  DistributedContext dist_context(&host_context, configuration);

  ASSERT_EQ(dist_context.GetHostContext(), &host_context);
}

}  // namespace
}  // namespace tfrt
