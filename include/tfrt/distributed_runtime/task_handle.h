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

// Fabric Communicator
//
// This file declares TaskHandle, an opaque handle representing a task.

#ifndef TFRT_DISTRIBUTED_RUNTIME_TASK_HANDLE_H_
#define TFRT_DISTRIBUTED_RUNTIME_TASK_HANDLE_H_

#include <cstdint>

#include "llvm/ADT/DenseMapInfo.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
// An opaque handle to represent a task.
// Wrapped inside a class to prevent unintentional auto cast to uint64_t.
// NOTE: Keep this class simple to minimize overhead of frequent copies.
class TaskHandle {
 public:
  // A value that won't be representing any task.
  static const TaskHandle kInvalidTaskHandle;

  // Allow default constructor, copy constructor, and explicit conversion
  TaskHandle() = default;
  explicit TaskHandle(uint64_t value) : value_(value) {}

  bool operator==(const TaskHandle& other) const {
    return value_ == other.get_value();
  }
  bool operator!=(const TaskHandle& other) const {
    return value_ != other.get_value();
  }

  uint64_t get_value() const { return value_; }

 private:
  uint64_t value_;
};

raw_ostream& operator<<(raw_ostream& os, const TaskHandle& value);
}  // namespace tfrt

namespace llvm {
template <>
struct DenseMapInfo<tfrt::TaskHandle> {
  static tfrt::TaskHandle getEmptyKey() {
    return tfrt::TaskHandle::kInvalidTaskHandle;
  }
  static tfrt::TaskHandle getTombstoneKey() {
    return tfrt::TaskHandle::kInvalidTaskHandle;
  }
  static unsigned getHashValue(const tfrt::TaskHandle& handle) {
    return handle.get_value();
  }
  static bool isEqual(const tfrt::TaskHandle& left,
                      const tfrt::TaskHandle& right) {
    return left.get_value() == right.get_value();
  }
};
}  // namespace llvm

#endif  // TFRT_DISTRIBUTED_RUNTIME_TASK_HANDLE_H_
