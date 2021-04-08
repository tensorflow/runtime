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

// Unit test for ClusterInfo.

#include "tfrt/distributed_runtime/cluster_info.h"

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace {
TEST(ClusterInfo, GetTaskHandle) {
  const std::string dist_config_str =
      "  cluster_config {"
      "    jobs {"
      "      name: 'worker_x'"
      "      tasks: { key: 0 value: 'addr0' }"
      "      tasks: { key: 1 value: 'addr1' }"
      "    }"
      "    jobs {"
      "      name: 'worker2'"
      "      tasks: { key: 2 value: 'addr2' }"
      "    }"
      "    jobs {"
      "      name: 'worker3'"
      "      tasks: { key: 3 value: 'addr3' }"
      "    }"
      "  }"
      "  job_name: 'worker_x'"
      "  task_id: 1";
  const std::vector<std::string> tasks{
      "/job:worker_x/task:0", "/job:worker_x/task:1", "/job:worker2/task:2",
      "/job:worker3/task:3"};
  const std::vector<std::string> addresses{"addr0", "addr1", "addr2", "addr3"};

  DistributedContextConfiguration config;
  EXPECT_TRUE(::google::protobuf::TextFormat::ParseFromString(dist_config_str,
                                                              &config));
  ClusterInfo cluster_info(config);

  // Get task information of the task itself
  auto expected_handle = cluster_info.GetTaskHandle(tasks[1]);
  EXPECT_EQ(cluster_info.GetTaskHandle(), expected_handle.get());
  expected_handle = cluster_info.GetTaskHandle("worker_x", 1);
  EXPECT_EQ(cluster_info.GetTaskHandle(), expected_handle.get());
  // Valid to look up with full device name
  expected_handle =
      cluster_info.GetTaskHandle(StrCat(tasks[1], "/device:CPU:2"));
  EXPECT_EQ(cluster_info.GetTaskHandle(), expected_handle.get());
  auto expected_name = cluster_info.GetTaskName(expected_handle.get());
  EXPECT_EQ(tasks[1], expected_name.get());
  auto expected_addr = cluster_info.GetTaskAddress(expected_handle.get());
  EXPECT_EQ(addresses[1], expected_addr.get());

  // Get task information of task[3]
  auto expected_handle_3 = cluster_info.GetTaskHandle(tasks[3]);
  EXPECT_NE(cluster_info.GetTaskHandle(), expected_handle_3.get());
  auto expected_handle_3_job_taskid = cluster_info.GetTaskHandle("worker3", 3);
  EXPECT_EQ(expected_handle_3.get(), expected_handle_3_job_taskid.get());
  auto expected_handle_3_with_device =
      cluster_info.GetTaskHandle(StrCat(tasks[3], "device:GPU:0"));
  EXPECT_EQ(expected_handle_3.get(), expected_handle_3_with_device.get());
  auto expected_name_3 = cluster_info.GetTaskName(expected_handle_3.get());
  EXPECT_EQ(tasks[3], expected_name_3.get());
  auto expected_addr_3 = cluster_info.GetTaskAddress(expected_handle_3.get());
  EXPECT_EQ(addresses[3], expected_addr_3.get());

  // Get task information for non-existing task
  const std::string unknown_task = "/job:worker2/task:0";
  auto unknown_handle = cluster_info.GetTaskHandle(unknown_task);
  EXPECT_TRUE(unknown_handle.errorIsA<TaskNotFoundErrorInfo>());
  unknown_handle = cluster_info.GetTaskHandle("worker_y", 0);
  EXPECT_TRUE(unknown_handle.errorIsA<TaskNotFoundErrorInfo>());
  const TaskHandle invalid_task_handle(123);
  auto unknown_task_name = cluster_info.GetTaskName(invalid_task_handle);
  EXPECT_TRUE(unknown_task_name.errorIsA<TaskNotFoundErrorInfo>());
  auto unknown_address = cluster_info.GetTaskName(invalid_task_handle);
  EXPECT_TRUE(unknown_address.errorIsA<TaskNotFoundErrorInfo>());
}
}  // namespace
}  // namespace tfrt
