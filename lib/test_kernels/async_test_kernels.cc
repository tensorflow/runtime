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

//===- async_test_kernels.cc ----------------------------------------------===//
//
// This library contains test kernels needed by example_kernels unit tests.
//
//===----------------------------------------------------------------------===//

#include <thread>

#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/rc_array.h"
#include "tfrt/test_kernels.h"

namespace tfrt {

/// The tfrt_test.do.async kernel runs an arbitrary function on a background
/// task.
static void TestDoAsync(RemainingArguments args, RemainingResults results,
                        Attribute<Function> body_fn,
                        const ExecutionContext& exec_ctx) {
  assert(body_fn->argument_types().size() == args.size() &&
         "argument count mismatch");
  assert(body_fn->result_types().size() == results.size() &&
         "result count mismatch");

  // Copy `args` and add a ref to each arg. These refs will be dropped when the
  // RCArray is destroyed. arg_refs is captured by the lambda so the kernel's
  // arguments will be available when the closure runs.
  RCArray<AsyncValue> arg_refs(args.values());

  // We need to create all the result values eagerly so we can return them
  // from the TestDoAsync function, even though we don't know their types.  Use
  // an IndirectAsyncValue for this, because it can lazily get resolved.
  SmallVector<RCReference<IndirectAsyncValue>, 4> result_refs;
  result_refs.reserve(results.size());
  for (int i = 0, e = results.size(); i != e; ++i) {
    auto result = results.AllocateIndirectResultAt(i);
    // To ensure the results live long enough to be filled in by our deferred
    // evaluation, we keep the RCReferences holding the results.
    result_refs.push_back(std::move(result));
  }

  HostContext* host = exec_ctx.host();
  host->EnqueueWork([exec_ctx, body = FormRef(&body_fn.get()),
                     arg_refs = std::move(arg_refs),
                     result_refs = std::move(result_refs)]() {
    SmallVector<RCReference<AsyncValue>, 8> results;
    results.resize(result_refs.size());

    body->Execute(exec_ctx, arg_refs.values(), results);

    // Resolve our temporary result values into the call results.  This
    // transfers the +1 results returned by Execute to the ForwardTo call.
    for (int i = 0, e = result_refs.size(); i != e; ++i) {
      result_refs[i]->ForwardTo(std::move(results[i]));
    }
  });
}

static void TestUSleep(Argument<int32_t> sleep_time_us_arg,
                       const ExecutionContext& exec_ctx) {
  int32_t sleep_time_us = *sleep_time_us_arg;
  bool work_enqueued = exec_ctx.host()->EnqueueBlockingWork([sleep_time_us] {
    std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_us));
    printf("Slept for %d microseconds\n", sleep_time_us);
    fflush(stdout);
  });

  if (!work_enqueued) {
    // We can't ReportError here because this kernel has no outputs.
    llvm::errs() << "Failed to enqueue blocking work. Maximum number of "
                    "pending blocking tasks reached.\n";
  }
}

static void TestBlockingUSleep(Argument<int32_t> sleep_time_us_arg,
                               Result<Chain> sleeping_done,
                               const ExecutionContext& exec_ctx) {
  int32_t sleep_time_us = *sleep_time_us_arg;
  bool work_enqueued = exec_ctx.host()->EnqueueBlockingWork(
      [sleep_time_us, sleeping_done = sleeping_done.Allocate()] {
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_us));
        printf("Slept for %d microseconds\n", sleep_time_us);
        fflush(stdout);
        sleeping_done.emplace();
      });

  if (!work_enqueued) {
    // We can't ReportError here because this kernel has no outputs.
    llvm::errs() << "Failed to enqueue blocking work. Maximum number of "
                    "pending blocking tasks reached.\n";
  }
}

// Test-only.
static void TestQuiesce(const ExecutionContext& exec_ctx) {
  exec_ctx.host()->Quiesce();
}

static void TestReportErrorConcreteAsync(Argument<int32_t> in,
                                         Result<int32_t> out,
                                         const ExecutionContext& exec_ctx,
                                         AsyncKernelFrame* frame) {
  AsyncValueRef<int32_t> result_ref = out.Allocate();
  exec_ctx.host()->EnqueueWork(
      [in = *in, result_ref = std::move(result_ref), frame = *frame]() mutable {
        if (in == 0) {
          result_ref.emplace(in);
        } else {
          // ReportError sets unavailable ConcreteAsyncValue to error.
          frame.ReportError("something bad happened asynchronously");
        }
      });
}

static void TestReportIndirectErrorAsync(Argument<int32_t> in,
                                         Result<int32_t> out,
                                         const ExecutionContext& exec_ctx,
                                         AsyncKernelFrame* frame) {
  HostContext* host = exec_ctx.host();
  auto result_ref = out.AllocateIndirect();
  host->EnqueueWork([in = *in, result_ref = std::move(result_ref),
                     frame = *frame, host]() mutable {
    if (in == 0) {
      auto concrete_av = host->MakeAvailableAsyncValueRef<int32_t>();
      result_ref->ForwardTo(std::move(concrete_av));
    } else {
      // ReportError creates a ConcreteAsyncValue in error state and
      // forwards any IndirectAsyncValue to it.
      frame.ReportError("something bad happened asynchronously");
    }
  });
}

void RegisterAsyncTestKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_test.do.async", TFRT_KERNEL(TestDoAsync));
  registry->AddKernel("tfrt_test.quiesce", TFRT_KERNEL(TestQuiesce));
  registry->AddKernel("tfrt_test.usleep", TFRT_KERNEL(TestUSleep));
  registry->AddKernel("tfrt_test.blocking.usleep",
                      TFRT_KERNEL(TestBlockingUSleep));
  registry->AddKernel("tfrt_test.report_error_concrete_async",
                      TFRT_KERNEL(TestReportErrorConcreteAsync));
  registry->AddKernel("tfrt_test.report_error_indirect_async",
                      TFRT_KERNEL(TestReportIndirectErrorAsync));
}
}  // namespace tfrt
