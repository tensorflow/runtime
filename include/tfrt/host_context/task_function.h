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

// Task Function Abstraction
//
// This file defines the TaskFunction class for representing work queue tasks.

#ifndef TFRT_HOST_CONTEXT_TASK_FUNCTION_H_
#define TFRT_HOST_CONTEXT_TASK_FUNCTION_H_

#include "llvm/ADT/FunctionExtras.h"

namespace tfrt {

class TaskFunction {
 public:
  TaskFunction() = default;
  explicit TaskFunction(llvm::unique_function<void()> work)
      : work_(std::move(work)) {}

  TaskFunction(TaskFunction&&) = default;
  TaskFunction& operator=(TaskFunction&&) = default;
  TaskFunction(const TaskFunction&) = delete;
  TaskFunction& operator=(const TaskFunction&) = delete;

  void operator()() { work_(); }
  explicit operator bool() const { return static_cast<bool>(work_); }
  void reset() { work_ = nullptr; }

 private:
  llvm::unique_function<void()> work_;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_TASK_FUNCTION_H_
