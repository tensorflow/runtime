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

//===- parallel_kernels.cc ------------------------------------------------===//
//
// This file implements core parallel kernels.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>

#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/host_context/parallel_for.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace {

static AsyncValueRef<Chain> HexParallelFor(const ExecutionContext& exec_ctx,
                                           Argument<int32_t> start,
                                           Argument<int32_t> end,
                                           Argument<int32_t> block_size,
                                           RemainingArguments args,
                                           Attribute<Function> body_fn_const) {
  const Function* body_fn = &(*body_fn_const);

  const size_t total_size = *end - *start;
  const size_t fixed_block_size = *block_size;
  const size_t offset = *start;

  //------------------------------------------------------------------------- //
  // Parallel for block function.
  auto compute = [exec_ctx, offset, body_fn,
                  args = RCArray<AsyncValue>(args.values())](size_t start,
                                                             size_t end) {
    HostContext* host = exec_ctx.host();

    // Pack parallel block arguments into async values.
    auto start_arg = host->MakeAvailableAsyncValueRef<int32_t>(start + offset);
    auto end_arg = host->MakeAvailableAsyncValueRef<int32_t>(end + offset);

    SmallVector<AsyncValue*, 6> fn_args = {start_arg.GetAsyncValue(),
                                           end_arg.GetAsyncValue()};
    for (AsyncValue* arg : args.values()) fn_args.push_back(arg);

    // Function returns single AsyncValueRef<Chain>.
    SmallVector<RCReference<AsyncValue>, 1> fn_results;
    fn_results.resize(1);

    // Call parallel for body.
    body_fn->Execute(exec_ctx, fn_args, fn_results);
    assert(fn_results.size() == 1);

    return AsyncValueRef<Chain>(std::move(fn_results[0]));
  };

  //------------------------------------------------------------------------- //
  // Return ready chain when all parallel for blocks are completed.
  auto on_done = [](ArrayRef<AsyncValueRef<Chain>> _) -> Chain {
    return Chain();
  };

  //------------------------------------------------------------------------- //
  // Launch parallel for operation.
  ParallelFor parallel_for(exec_ctx.host());
  return parallel_for.Execute<Chain, Chain>(
      total_size, ParallelFor::BlockSizes::Fixed(fixed_block_size),
      std::move(compute), std::move(on_done));
}

}  // namespace

void RegisterParallelKernels(KernelRegistry* registry) {
  registry->AddKernel("hex.parallel_for.i32", TFRT_KERNEL(HexParallelFor));
}

}  // namespace tfrt
