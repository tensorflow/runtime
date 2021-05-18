// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- Code Size Test App -------------------------------------------------===//
//
// This file builds a minimal-sized binary that links trivial kernels and the
// simplest core components to run a fibonacci computation. It is used to track
// the code size of a deployed binary on mobile environment.

#include <cstdio>

#include "tfrt/basic_kernels/basic_kernels.h"
#include "tfrt/bef/bef_buffer.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/location.h"

//===----------------------------------------------------------------------===//
// Driver main
//===----------------------------------------------------------------------===//

// Read from STDIN, and remove EOF.
tfrt::BefBuffer ReadFromStdInToBuffer() {
  tfrt::BefBuffer buffer;
  buffer.reserve(1024);
  int c;
  while ((c = getchar()) != EOF) buffer.push_back(c);
  return buffer;
}

int main(int argc, char** argv) {
  // decoded_diagnostic_handler does nothing.
  auto decoded_diagnostic_handler = [&](const tfrt::DecodedDiagnostic& diag) {};

  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue =
      tfrt::CreateSingleThreadedWorkQueue();
  std::unique_ptr<tfrt::HostAllocator> host_allocator =
      tfrt::CreateMallocAllocator();
  tfrt::HostContext host(decoded_diagnostic_handler, std::move(host_allocator),
                         std::move(work_queue));

  // Register kernels used for fib.mlir test inline to avoid _alwayslink target
  // in this minimal test driver.
  tfrt::RegisterIntegerKernels(host.GetMutableRegistry());
  tfrt::RegisterControlFlowKernels(host.GetMutableRegistry());

  auto buffer = ReadFromStdInToBuffer();
  auto bef(tfrt::BEFFile::Open(buffer, host.GetKernelRegistry(),
                               decoded_diagnostic_handler,
                               host_allocator.get()));
  if (!bef) {
    printf("Could not BEFFile::Open fib.bef.\n");
    fflush(stdout);
    abort();
  }

  tfrt::SmallVector<const tfrt::Function*, 8> function_list;
  // Run all functions in the input BEF file.
  bef->GetFunctionList(&function_list);

  // Loop over each of the functions, running each as a standalone testcase.
  for (auto* fn : function_list) {
    // If the function takes arguments, then we can't run it from this driver.
    if (!fn->argument_types().empty()) {
      printf("--- Not running '%s' because it has arguments.\n",
             fn->name().str().c_str());
      fflush(stdout);
      continue;
    }

    // Skip anonymous functions.
    if (fn->name().empty()) {
      continue;
    }

    printf("--- Running '%s':\n", fn->name().str().c_str());
    fflush(stdout);

    // Kick off an execution of the function body.
    llvm::SmallVector<tfrt::RCReference<tfrt::AsyncValue>, 4> results;
    results.resize(fn->result_types().size());

    tfrt::Expected<tfrt::RCReference<tfrt::RequestContext>> req_ctx =
        tfrt::RequestContextBuilder(&host, /*resource_context=*/nullptr)
            .build();
    if (!req_ctx) {
      fprintf(stderr, "Failed to build a RequestContext.\n");
      abort();
    }
    tfrt::ExecutionContext exec_ctx{std::move(*req_ctx)};
    fn->Execute(exec_ctx, /*arguments=*/{}, results);

    // Block until the function results are fully resolved.
    host.Await(results);

    // Go ahead and print out the function results that we know about.
    if (!results.empty()) {
      printf("'%s' returned ", fn->name().str().c_str());
      auto result_types = fn->result_types();

      for (int i = 0, e = results.size(); i != e; ++i) {
        auto type_name = result_types[i];
        if (auto* error = results[i]->GetErrorIfPresent()) {
          printf("<<error: %s>>", error->message.c_str());
        } else if (type_name.GetName() == "i32") {
          printf("%d", results[i]->get<int32_t>());
        } else {
          printf("%s value", type_name.GetName().str().c_str());
        }

        // Print comma except for the last one.
        if (i != results.size() - 1) {
          printf(",");
        }
      }

      printf("\n");
      fflush(stdout);
    }

    // In this test driver, we want to make sure that every function completes
    // all execution before moving on to the next one.  This makes the leak
    // checker work better in the face of side effecting kernels that aren't
    // properly chained together (which is useful for testing).
    host.Quiesce();
  }

  bef.reset();
  return 0;
}
