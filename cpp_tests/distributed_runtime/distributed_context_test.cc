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

// Unit test for DistributedContext.

#include "tfrt/distributed_runtime/distributed_context.h"

#include <atomic>
#include <memory>

#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
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
              (DistributedContext * dist_context, TaskHandle task_handle),
              (override));
};

DistributedContextConfiguration GetSampleDistributedConfiguration() {
  const std::string dist_config_str =
      "  cluster_config {"
      "    jobs {"
      "      name: 'worker_a'"
      "      tasks: { key: 0 value: 'addr0' }"
      "      tasks: { key: 1 value: 'addr1' }"
      "    }"
      "    jobs {"
      "      name: 'worker_b'"
      "      tasks: { key: 2 value: 'addr2' }"
      "      tasks: { key: 3 value: 'addr3' }"
      "    }"
      "  }"
      "  job_name: 'worker_a'"
      "  task_id: 1"
      "  collective_groups {"
      "    name: 'group0'"
      "    members: '/job:worker_a/task:0'"
      "    members: '/job:worker_a/task:1'"
      "  }"
      "  collective_groups {"
      "    name: 'group1'"
      "    members: '/job:worker_a/task:1'"
      "    members: '/job:worker_b/task:2'"
      "    members: '/job:worker_b/task:3'"
      "  }";

  DistributedContextConfiguration config;
  EXPECT_TRUE(::google::protobuf::TextFormat::ParseFromString(dist_config_str,
                                                              &config));
  return config;
}

ServerContextConfiguration GetSampleServerConfiguration(
    DistributedContextConfiguration& dist_config) {
  const auto& cluster_config = dist_config.cluster_config();
  string_view server_address;
  for (const auto& job_config : cluster_config.jobs()) {
    if (job_config.name() == dist_config.job_name()) {
      server_address = job_config.tasks().at(dist_config.task_id());
      break;
    }
  }
  FabricCommunicatorConfiguration mock_communicator_config{
      kMockCommunicatorType, server_address.str()};
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
  dist_config.set_task_id(0);
  EXPECT_FALSE(!server_context.CreateDistributedContext(ctx_id0, dist_config));
  dist_config.set_task_id(1);
  EXPECT_FALSE(!server_context.CreateDistributedContext(ctx_id1, dist_config));

  DistributedContext* dist_ctx0 =
      server_context.GetDistributedContext(ctx_id0).get();
  EXPECT_NE(dist_ctx0, nullptr);
  EXPECT_EQ(dist_ctx0->GetHostContext(), &host_context);
  EXPECT_EQ(dist_ctx0->GetContextId(), ctx_id0);
  EXPECT_EQ(dist_ctx0->GetTaskName(), "/job:worker_a/task:0");
  DistributedContext* dist_ctx1 =
      server_context.GetDistributedContext(ctx_id1).get();
  EXPECT_NE(dist_ctx1, nullptr);
  EXPECT_EQ(dist_ctx1->GetHostContext(), &host_context);
  EXPECT_EQ(dist_ctx1->GetContextId(), ctx_id1);
  EXPECT_EQ(dist_ctx1->GetTaskName(), "/job:worker_a/task:1");

  EXPECT_NE(dist_ctx0, dist_ctx1);
  // Creating DistributedContext with existing context id will lead to error
  EXPECT_TRUE(server_context.CreateDistributedContext(ctx_id0, dist_config)
                  .takeError()
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
  EXPECT_FALSE(!server.CreateDistributedContext(context_id, dist_config));
  DistributedContext* dist_context =
      server.GetDistributedContext(context_id).get();

  auto group1 = dist_context->GetCollectiveGroup("group1");
  EXPECT_EQ(group1.name, "group1");
  EXPECT_EQ(group1.members.size(), 3);
}

}  // namespace
}  // namespace tfrt
