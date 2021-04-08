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

// Executor Test Driver Library
//
// This file declares the interface for test driver library for the bef
// executor.

#ifndef TFRT_BEF_EXECUTOR_DRIVER_BEF_EXECUTOR_DRIVER_H_
#define TFRT_BEF_EXECUTOR_DRIVER_BEF_EXECUTOR_DRIVER_H_

#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

enum class HostAllocatorType {
  // Allocator that just calls malloc/free.
  kMalloc,

  // Allocator with limited capacity.
  kTestFixedSizeMalloc,

  // Allocator wrapped around kMalloc but profiles memory usage.
  kProfiledMalloc,

  // Allocator wrapped around profiled malloc and exit(1) on detecting memory
  // leak.
  kLeakCheckMalloc,
};

struct RunBefConfig {
  string_view program_name;
  // Use '-' to take input from stdin.
  string_view input_filename;
  ArrayRef<std::string> shared_libs;
  ArrayRef<std::string> functions;
  std::string test_init_function;
  std::string work_queue_type;
  tfrt::HostAllocatorType host_allocator_type;
  bool print_error_code = false;
};

// Run the BEF program with default execution context.
int RunBefExecutor(const RunBefConfig& run_config);

// Run the BEF program with the specified execution context. For each entry
// function, one execution context will be created and used.
int RunBefExecutor(
    const RunBefConfig& run_config,
    const std::function<llvm::Expected<ExecutionContext>(
        HostContext*, ResourceContext*)>& create_execution_context);

}  // namespace tfrt
#endif  // TFRT_BEF_EXECUTOR_DRIVER_BEF_EXECUTOR_DRIVER_H_
