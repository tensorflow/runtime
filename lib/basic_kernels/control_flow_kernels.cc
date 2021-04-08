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

// This file implements core control flow related kernels.

#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/kernel_frame.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/rc_array.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

static Chain TFRTNewChain() { return Chain(); }

static void TFRTMergeChains(Argument<Chain> chain_in,
                            RemainingArguments remaining_arguments,
                            Result<Chain> chain_out) {
  // We can return an arbitrary chain argument - we know that all are ready.
  chain_out.Set(chain_in);
}

static void TFRTCall(RemainingArguments args, RemainingResults results,
                     Attribute<Function> fn, const ExecutionContext& exec_ctx) {
  assert(fn->argument_types().size() == args.size() &&
         "argument count mismatch");
  assert(fn->result_types().size() == results.size() &&
         "result count mismatch");

  fn->Execute(exec_ctx, args.values(), results.values());
}

static void TFRTCase(RemainingArguments args, RemainingResults results,
                     RemainingFunctions branches,
                     const ExecutionContext& exec_ctx) {
  // The first argument is branch index, which must present.
  assert(args.size() >= 1);

  auto case_impl = [exec_ctx](
                       SmallVector<const Function*, 4> branches,
                       AsyncValue* branch_index_av, ArrayRef<AsyncValue*> args,
                       MutableArrayRef<RCReference<AsyncValue>> results) {
    assert(branch_index_av->IsAvailable());
    // If we have an error, propagate to all the results.
    if (branch_index_av->IsError()) {
      for (auto& result : results) result = FormRef(branch_index_av);
      return;
    }

    // Otherwise obtain the branch index and execute the branch.
    int branch_index = branch_index_av->get<int>();
    // Check if index is valid.
    if (branch_index < 0 || branch_index >= branches.size()) {
      auto error = EmitErrorAsync(
          exec_ctx,
          tfrt::StrCat("branch_index invalid. branch index: ", branch_index,
                       " # branches: ", branches.size()),
          tfrt::ErrorCode::kInvalidArgument);
      for (auto& result : results) result = error.CopyRef();
    }
    branches[branch_index]->Execute(exec_ctx, args.drop_front(), results);
  };

  // If branch_index is available, try to dispatch the branch right away.
  AsyncValue* branch_index_av = args[0];
  if (branch_index_av->IsAvailable()) {
    // Obtain the branches from RemainingAttributes.
    SmallVector<const Function*, 4> branch_vector;
    branch_vector.reserve(branches.size());
    for (int i = 0, e = branches.size(); i != e; ++i)
      branch_vector.push_back(&(*branches.Get(i)));
    case_impl(branch_vector, branch_index_av, args.values(), results.values());
    return;
  }

  // Copy `args` and add a ref to each arg. These refs will be dropped when the
  // RCArray is destroyed. arg_refs is captured by the lambda so the kernel's
  // arguments will be available when the closure runs.
  RCArray<AsyncValue> arg_refs(args.values());

  SmallVector<RCReference<IndirectAsyncValue>, 4> result_refs;
  result_refs.reserve(results.size());
  for (int i = 0, e = results.size(); i != e; ++i) {
    auto result = results.AllocateIndirectResultAt(i);
    // To ensure the results live long enough to be filled in by our deferred
    // evaluation, we keep the RCReferences holding the results.
    result_refs.push_back(std::move(result));
  }

  // Copy `branches` and add a ref to each branch, which is captured by the
  // lambda so the function pointers to the branches will be available when the
  // closure runs.
  SmallVector<RCReference<const Function>, 4> branch_refs;
  branch_refs.reserve(branches.size());
  for (int i = 0, e = branches.size(); i != e; ++i) {
    const Function* branch = &(*branches.Get(i));
    branch_refs.push_back(FormRef(branch));
  }

  // Dispatch when the branch index becomes available.
  branch_index_av->AndThen([case_impl, branch_refs = std::move(branch_refs),
                            branch_index_av = std::move(branch_index_av),
                            arg_refs = std::move(arg_refs),
                            result_refs = std::move(result_refs)] {
    assert(arg_refs[0]->IsAvailable() &&
           "We must have the branch index by now");

    SmallVector<RCReference<AsyncValue>, 8> results;
    results.resize(result_refs.size());

    SmallVector<const Function*, 4> branch_vector;
    branch_vector.reserve(branch_refs.size());
    for (int i = 0, e = branch_refs.size(); i != e; ++i)
      branch_vector.push_back(branch_refs[i].get());
    case_impl(branch_vector, branch_index_av, arg_refs.values(), results);

    // Forward result_refs to results. This transfers the +1 results returned by
    // Execute to the ForwardTo call.
    for (int i = 0, e = result_refs.size(); i != e; ++i) {
      result_refs[i]->ForwardTo(std::move(results[i]));
    }
  });
}

// tfrt.if dispatches to a 'true' or 'false' function based on a condition.
//
// Arguments: The first argument is the condition, with type i1, and any
// additional arguments are passed to the selected function.
//
// Attributes: The first attribute is the true_fn, and the second attribute is
// the false_fn. The functions must have matching signatures, and their
// signatures must match tfrt.if's signature, exempting the extra i1 for the
// condition.
//
// tfrt.if supports "non-strict" invocation: it is safe to invoke before all its
// arguments are ready. The caller must set the bef.nonstrict attribute on
// tfrt.if to make an invocation non-strict.
static void TFRTIf(RemainingArguments args, RemainingResults results,
                   Attribute<Function> true_fn_const,
                   Attribute<Function> false_fn_const,
                   const ExecutionContext& exec_ctx) {
  assert(args.size() > 0);

  const Function* true_fn = &(*true_fn_const);
  const Function* false_fn = &(*false_fn_const);

  assert(true_fn->argument_types().size() == args.size() - 1 &&
         "argument count mismatch");
  assert(true_fn->result_types().size() == results.size() &&
         "result count mismatch");
  assert(true_fn->argument_types() == false_fn->argument_types() &&
         true_fn->result_types() == false_fn->result_types() &&
         "true and false function types need to line up");

  auto if_impl = [exec_ctx](const Function* true_fn, const Function* false_fn,
                            ArrayRef<AsyncValue*> args,
                            MutableArrayRef<RCReference<AsyncValue>> results) {
    AsyncValue* condition = args[0];
    // If we have an error, then we can force propagate errors to all the
    // results.
    if (condition->IsError()) {
      for (auto& result : results) result = FormRef(condition);
      return;
    }

    // Otherwise, we know which way to go.
    const Function* fn = condition->get<bool>() ? true_fn : false_fn;
    fn->Execute(exec_ctx, args.drop_front(), results);
  };

  // If the condition is already available, we can immediately dispatch the
  // call.
  AsyncValue* condition = args[0];
  if (condition->IsAvailable()) {
    if_impl(true_fn, false_fn, args.values(), results.values());
    return;
  }

  // Note: At this point, the condition's availability is unknown. It was
  // unavailable when we checked above, but it may become available at any time.

  // Copy `args` and add a ref to each arg. These refs will be dropped when the
  // RCArray is destroyed. arg_refs is captured by the lambda so the kernel's
  // arguments will be available when the closure runs.
  RCArray<AsyncValue> arg_refs(args.values());

  // We need to create all the result values eagerly so we can return them
  // from the TFRTIf function, even though we don't know their types.  Use
  // an IndirectAsyncValue for this, because it can lazily get resolved.
  SmallVector<RCReference<IndirectAsyncValue>, 4> result_refs;
  result_refs.reserve(results.size());
  for (int i = 0, e = results.size(); i != e; ++i) {
    auto result = results.AllocateIndirectResultAt(i);
    // To ensure the results live long enough to be filled in by our deferred
    // evaluation, we keep the RCReferences holding the results.
    result_refs.push_back(std::move(result));
  }

  // Dispatch when the condition becomes available.
  condition->AndThen([if_impl, true_fn_ref = FormRef(true_fn),
                      false_fn_ref = FormRef(false_fn),
                      arg_refs = std::move(arg_refs),
                      result_refs = std::move(result_refs)] {
    assert(arg_refs[0]->IsAvailable() && "We must have the condition by now");

    SmallVector<RCReference<AsyncValue>, 8> results;
    results.resize(result_refs.size());
    if_impl(true_fn_ref.get(), false_fn_ref.get(), arg_refs.values(), results);

    // Forward result_refs to results. This transfers the +1 results returned by
    // Execute to the ForwardTo call.
    for (int i = 0, e = result_refs.size(); i != e; ++i) {
      result_refs[i]->ForwardTo(std::move(results[i]));
    }
  });
}

// This is a helper function that runs a block of iterations and sets up a
// callback to run the next block at the end.
static void TFRTRepeatI32Block(
    int32_t start, int32_t block_size, int32_t count_value,
    const ExecutionContext& exec_ctx, RCReference<const Function> body_fn_ref,
    RCArray<AsyncValue> args,
    SmallVector<RCReference<IndirectAsyncValue>, 4>&& result_refs) {
  // Temporary buffers to store intermediate arguments and results.
  SmallVector<AsyncValue*, 8> passed_args(args.values().begin(),
                                          args.values().end());

  SmallVector<RCReference<AsyncValue>, 4> results;
  results.resize(result_refs.size());
  auto num_fn_args = args.size();

  auto end = std::min(start + block_size, count_value);

  for (int i = start; i < end; ++i) {
    if (auto cancel_av = exec_ctx.GetCancelAsyncValue()) {
      // Cancellation detected. DropRef on args if needed, set results to
      // the cancel async value, and break out.
      for (int arg = 0; arg != num_fn_args; ++arg) {
        // If this is not the first iteration, destroy the loop-carried
        // args. The first iteration uses TFRTRepeatI32's args, which we
        // can't destroy.
        if (i > 0) passed_args[arg]->DropRef();
      }

      for (auto& result : result_refs) {
        result->ForwardTo(FormRef(cancel_av));
      }
      return;
    }

    body_fn_ref->Execute(exec_ctx, passed_args, results);

    for (int arg = 0; arg != num_fn_args; ++arg) {
      // If this is not the first iteration, destroy the loop-carried
      // args. The first iteration uses TFRTRepeatI32's args, which we
      // can't destroy.
      if (i > 0) passed_args[arg]->DropRef();

      // If this is not the last iteration, set up for the next
      // iteration by copying this iteration's results to the next
      // iteration's args.
      if (i + 1 != count_value) {
        passed_args[arg] = results[arg].release();
      }
    }
  }

  // Forward result_refs to the actual result values from the last iteration.
  if (end >= count_value) {
    for (int i = 0, e = result_refs.size(); i != e; ++i) {
      result_refs[i]->ForwardTo(std::move(results[i]));
    }
    return;
  } else {
    assert(num_fn_args > 0);
    passed_args[0]->AndThen(
        [end, block_size, count_value, exec_ctx,
         body_fn_ref = std::move(body_fn_ref),
         arg_refs = RCArray<AsyncValue>(llvm::makeArrayRef(passed_args)),
         result_refs = std::move(result_refs)]() mutable {
          TFRTRepeatI32Block(end, block_size, count_value, exec_ctx,
                             std::move(body_fn_ref), std::move(arg_refs),
                             std::move(result_refs));
        });
  }
}

// This takes a single i32 iteration count, plus arguments that are passed to
// the body_fn and eventually returned.
static void TFRTRepeatI32(RemainingArguments args, RemainingResults results,
                          Attribute<Function> body_fn_const,
                          const ExecutionContext& exec_ctx) {
  assert(args.size() > 0 && args.size() - 1 == results.size());

  const Function* body_fn = &(*body_fn_const);

  // Repeat gets a single function constant for its body_fn.
  assert(body_fn->argument_types() == body_fn->result_types() &&
         "Argument and result types of repeat body_fn must match");

  auto while_impl =
      [exec_ctx](
          RCReference<const Function> body_fn_ref, RCArray<AsyncValue> arg_refs,
          SmallVector<RCReference<IndirectAsyncValue>, 4> result_refs) mutable {
        // TODO(xldrx,jingdong): Get the block_size from an optional attribute.
        int32_t block_size = 32;
        auto args = arg_refs.values();
        auto* count = args[0];
        args = args.drop_front();

        // If we have an error, then we can force propagate errors to all the
        // results.
        if (count->IsError()) {
          for (auto& result : result_refs) {
            result->ForwardTo(FormRef(count));
          }
          return;
        }

        auto count_value = count->get<int32_t>();

        auto num_fn_args = args.size();

        // If the function does not returns any results, it is not feasible to
        // divide the loop into multiple blocks
        if (num_fn_args == 0 || block_size <= 0) block_size = count_value;

        // Special case: "Repeat 0" just copies args to results.
        if (count_value == 0) {
          for (int arg = 0; arg != num_fn_args; ++arg) {
            result_refs[arg]->ForwardTo(FormRef(args[arg]));
          }
          return;
        }

        assert(result_refs.size() == num_fn_args);
        // Run 'body_fn' at least once.
        assert(count_value > 0);

        TFRTRepeatI32Block(0, block_size, count_value, exec_ctx,
                           std::move(body_fn_ref), RCArray<AsyncValue>(args),
                           std::move(result_refs));
      };

  // If the count is already available, we can immediately dispatch the bodies.
  AsyncValue* count = args[0];

  // Copy `args` and add a ref to each arg. These refs will be dropped when the
  // RCArray is destroyed. arg_refs is captured by the lambda so the kernel's
  // arguments will be available when the closure runs.
  RCArray<AsyncValue> arg_refs(args.values());

  // Create a RCRef of Function to extend its lifetime into the lambda.
  RCReference<const Function> body_fn_ref = FormRef(body_fn);

  // Define results as IndirectAsync values. The actual results is set in the
  // last iteration of the loop.
  SmallVector<RCReference<IndirectAsyncValue>, 4> result_refs;
  result_refs.reserve(results.size());
  for (int i = 0, e = results.size(); i != e; ++i) {
    auto result = results.AllocateIndirectResultAt(i);
    result_refs.push_back(std::move(result));
  }

  // Dispatch when the condition becomes available.
  if (count->IsAvailable()) {
    while_impl(std::move(body_fn_ref), std::move(arg_refs),
               std::move(result_refs));

  } else {
    count->AndThen([while_impl, body_fn_ref = std::move(body_fn_ref),
                    arg_refs = std::move(arg_refs),
                    result_refs = std::move(result_refs)]() mutable {
      while_impl(std::move(body_fn_ref), std::move(arg_refs),
                 std::move(result_refs));
    });
  }
}

// This kernel takes a Chain and an AsyncValue. Then it returns the same
// AsyncValue. A function can use this kernel to return a value that depends on
// a given chain.
static void TFRTAliasValue(Chain chain, RemainingArguments args,
                           RemainingResults results) {
  assert((args.size() == 1) && "args should contain one AsyncValue");
  assert((results.size() == 1) && "results should contain one AsyncValue");
  results[0] = FormRef(args.values()[0]);
}

// This is the entrypoint to the library.
void RegisterControlFlowKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt.new.chain", TFRT_KERNEL(TFRTNewChain));
  registry->AddKernel("tfrt.merge.chains", TFRT_KERNEL(TFRTMergeChains));
  registry->AddKernel("tfrt.alias.value", TFRT_KERNEL(TFRTAliasValue));
  registry->AddKernel("tfrt.repeat.i32", TFRT_KERNEL(TFRTRepeatI32));
  registry->AddKernel("tfrt.call", TFRT_KERNEL(TFRTCall));
  registry->AddKernel("tfrt.if", TFRT_KERNEL(TFRTIf));
  registry->AddKernel("tfrt.cond", TFRT_KERNEL(TFRTIf));
  registry->AddKernel("tfrt.case", TFRT_KERNEL(TFRTCase));
}

}  // namespace tfrt
