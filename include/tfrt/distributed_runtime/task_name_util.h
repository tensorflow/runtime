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

// Declares utils for concatenating and parsing task names.

#ifndef TFRT_DISTRIBUTED_RUNTIME_TASK_NAME_UTIL_H_
#define TFRT_DISTRIBUTED_RUNTIME_TASK_NAME_UTIL_H_

#include <string>

#include "tfrt/support/forward_decls.h"

namespace tfrt {
class TaskNameUtil {
 public:
  // Concatenate job name and task id to get full task name
  // e.g., "/job:worker/task:1"
  static std::string ConcatTaskName(string_view job_name, int task_id);

  // Concatenate job name, task id, and device name to get fully specified
  // device name, e.g., "/job:worker/task:1/device:CPU:0"
  static std::string ConcatDeviceName(string_view job_name, int task_id,
                                      string_view device_name);

  // Maybe strip the job name and task name from device name.
  static std::string StripDevicePrefix(string_view device_name);

  // Parse a full task name (e.g., "/job:worker/task:1") to extract the job name
  // (e.g., "worker") and task id (e.g., 1)
  //
  // An acceptable task name should be in the following format:
  //   /job:<job-name>/task:<task-id>[/device:<device-type>:<device-id>]
  //
  // * Job name: starts with letter, may have letters, digits, underscores
  // * Task id: non-negative integer (no leading zeros)
  // * Device info is optional and will be ignored in parsing task names
  static Error ParseTaskName(string_view task_name, std::string* out_job_name,
                             int* out_task_id);

  // If this is called, "/replica:0" will be added as prefix of task_name.
  static void SetUseReplicaInTaskName();
};
}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_TASK_NAME_UTIL_H_
