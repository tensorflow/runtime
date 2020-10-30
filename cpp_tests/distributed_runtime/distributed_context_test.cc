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
#include "tfrt/distributed_runtime/proto/remote_message.pb.h"
#include "tfrt/distributed_runtime/remote_client.h"
#include "tfrt/distributed_runtime/server_context.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"

namespace tfrt {

namespace {

const char* kMockCommunicatorType = "mock_type";
const char* kMockCommunicatorName = "mock_name";

class MockCommunicator : public FabricCommunicator {
 public:
  explicit MockCommunicator(llvm::StringRef name)
      : FabricCommunicator(name, /*server_context=*/nullptr) {}

  MOCK_METHOD(std::unique_ptr<RemoteClientInterface>, CreateRemoteClient,
              (DistributedContext * dist_context, HostId remote_host_id),
              (override));
};

DistributedContextConfiguration GetSampleDistributedConfiguration() {
  ClusterConfiguration cluster_config{/*addresses=*/{{"worker0", "addr0"},
                                                     {"worker1", "addr1"},
                                                     {"worker2", "addr2"},
                                                     {"worker3", "addr3"}},
                                      /*id=*/1};
  FabricCommunicatorConfiguration mock_communicator_config{
      kMockCommunicatorType,
      cluster_config.addresses[cluster_config.id].address};
  CollectiveGroup group0{/*name=*/"group0", /*members=*/{0, 1}};
  CollectiveGroup group1{/*name=*/"group1", /*members=*/{1, 2, 3}};
  DistributedContextConfiguration dist_config{
      cluster_config,
      /*collective_groups=*/{group0, group1}};
  return dist_config;
}

ServerContextConfiguration GetSampleServerConfiguration(
    DistributedContextConfiguration& dist_config) {
  const auto& cluster_config = dist_config.cluster_config;
  const auto& server_address =
      cluster_config.addresses[cluster_config.id].address;
  FabricCommunicatorConfiguration mock_communicator_config{
      kMockCommunicatorType, server_address};
  ServerContextConfiguration server_config{mock_communicator_config};
  return server_config;
}

TEST(ServerContext, CreateFabricCommunicator) {
  auto diag_handler = [](const DecodedDiagnostic&) {};
  HostContext host_context(diag_handler, tfrt::CreateMallocAllocator(),
                           tfrt::CreateMultiThreadedWorkQueue(
                               /*num_threads=*/4,
                               /*num_blocking_threads=*/64));

  auto dist_config = GetSampleDistributedConfiguration();
  auto server_config = GetSampleServerConfiguration(dist_config);
  ServerContext server_context(&host_context, server_config);

  EXPECT_EQ(server_context.GetHostContext(), &host_context);
  EXPECT_EQ(server_context.GetOrCreateFabricCommunicator(), nullptr);

  auto mock_factory = [](ServerContext* server_context) -> FabricCommunicator* {
    return new MockCommunicator(kMockCommunicatorName);
  };
  FabricCommunicator::RegisterFabricCommunicatorType(kMockCommunicatorType,
                                                     mock_factory);

  FabricCommunicator* fabric_communicator =
      server_context.GetOrCreateFabricCommunicator();
  EXPECT_NE(fabric_communicator, nullptr);
  EXPECT_EQ(server_context.GetOrCreateFabricCommunicator(),
            fabric_communicator);
}

TEST(ServerContext, CreateDistributedContext) {
  auto diag_handler = [](const DecodedDiagnostic&) {};
  HostContext host_context(diag_handler, tfrt::CreateMallocAllocator(),
                           tfrt::CreateMultiThreadedWorkQueue(
                               /*num_threads=*/4,
                               /*num_blocking_threads=*/64));
  auto dist_config = GetSampleDistributedConfiguration();
  auto server_config = GetSampleServerConfiguration(dist_config);
  ServerContext server_context(&host_context, server_config);
  const uint64_t ctx_id0 = 0;
  const uint64_t ctx_id1 = 1;
  dist_config.cluster_config.id = 0;
  EXPECT_FALSE(server_context.CreateDistributedContext(ctx_id0, dist_config));
  dist_config.cluster_config.id = 1;
  EXPECT_FALSE(server_context.CreateDistributedContext(ctx_id1, dist_config));

  DistributedContext* dist_ctx0 =
      server_context.GetDistributedContext(ctx_id0).get();
  EXPECT_NE(dist_ctx0, nullptr);
  EXPECT_EQ(dist_ctx0->GetHostContext(), &host_context);
  EXPECT_EQ(dist_ctx0->GetContextId(), ctx_id0);
  EXPECT_EQ(dist_ctx0->GetHostId(), 0);
  DistributedContext* dist_ctx1 =
      server_context.GetDistributedContext(ctx_id1).get();
  EXPECT_NE(dist_ctx1, nullptr);
  EXPECT_EQ(dist_ctx1->GetHostContext(), &host_context);
  EXPECT_EQ(dist_ctx1->GetContextId(), ctx_id1);
  EXPECT_EQ(dist_ctx1->GetHostId(), 1);

  EXPECT_NE(dist_ctx0, dist_ctx1);
  // Creating DistributedContext with existing context id will lead to error
  EXPECT_TRUE(server_context.CreateDistributedContext(ctx_id0, dist_config)
                  .isA<UnknownErrorInfo>());
}

TEST(DistributedContext, GetCollectiveGroup) {
  auto diag_handler = [](const DecodedDiagnostic&) {};
  HostContext host_context(diag_handler, tfrt::CreateMallocAllocator(),
                           tfrt::CreateMultiThreadedWorkQueue(
                               /*num_threads=*/4,
                               /*num_blocking_threads=*/64));

  auto dist_config = GetSampleDistributedConfiguration();
  auto server_config = GetSampleServerConfiguration(dist_config);
  ServerContext server(&host_context, server_config);
  const uint64_t context_id = 0;
  EXPECT_FALSE(server.CreateDistributedContext(context_id, dist_config));
  DistributedContext* dist_context =
      server.GetDistributedContext(context_id).get();

  auto group1 = dist_context->GetCollectiveGroup("group1");
  EXPECT_EQ(group1.name, "group1");
}

}  // namespace
}  // namespace tfrt
