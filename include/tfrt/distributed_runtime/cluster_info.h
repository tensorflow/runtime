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

// Declares ClusterInfo which holds the tasks and their name and address info.

#ifndef TFRT_DISTRIBUTED_RUNTIME_TASK_UTIL_H_
#define TFRT_DISTRIBUTED_RUNTIME_TASK_UTIL_H_

#include <cstdint>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "tfrt/distributed_runtime/task_handle.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
class DistributedContextConfiguration;

// Hold the job and task information of the cluster within a DistributedContext.
class ClusterInfo {
 public:
  explicit ClusterInfo(const DistributedContextConfiguration& dist_config);

  ClusterInfo(ClusterInfo&&) = delete;
  ClusterInfo& operator=(ClusterInfo&&) = delete;

  ClusterInfo(const ClusterInfo&) = delete;
  ClusterInfo& operator=(const ClusterInfo&) = delete;

  // Get the local task's TaskHandle.
  TaskHandle GetTaskHandle() const { return task_handle_; }

  // Get handle with the task's full name, e.g., "/job:worker/task:1"
  Expected<TaskHandle> GetTaskHandle(string_view task_name) const;

  Expected<TaskHandle> GetTaskHandle(string_view job_name, int task_id) const;

  Expected<string_view> GetTaskName(TaskHandle task_handle) const;

  Expected<string_view> GetTaskAddress(TaskHandle task_handle) const;

 private:
  struct JobInfo;
  struct TaskInfo {
    const TaskHandle handle;
    const JobInfo* job;
    const int task_id;
    // Full name of the task, e.g., "/job:worker/task:1"
    const std::string name;
    const std::string address;
    TaskInfo() = delete;
    explicit TaskInfo(JobInfo* job_info, int task_id, string_view addr);
  };

  struct JobInfo {
    // Name of the job, e.g., "worker"
    const std::string name;
    llvm::DenseMap<int, TaskInfo> tasks;
    JobInfo() = delete;
    explicit JobInfo(string_view job) : name(job) {}
  };

  // Task handle representing the current host.
  TaskHandle task_handle_;

  llvm::StringMap<JobInfo> jobs_;
  llvm::DenseMap<TaskHandle, TaskInfo*> tasks_;
};
}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_TASK_UTIL_H_
