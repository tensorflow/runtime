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

// Defines the cluster info type.

#include "tfrt/distributed_runtime/task_handle.h"

#include "tfrt/support/string_util.h"

namespace tfrt {
const TaskHandle TaskHandle::kInvalidTaskHandle(0);

raw_ostream& operator<<(raw_ostream& os, const TaskHandle& value) {
  return os << StrCat("task_handle[", value.get_value(), "]");
}
}  // namespace tfrt
