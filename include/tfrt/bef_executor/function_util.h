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

// Utility for invoking BEF Functions
//
// This file implements utility functions to make invoking BEF Functions easy
// when the caller knows the static signature of the function.

#ifndef TFRT_BEF_EXECUTOR_FUNCTION_UTIL_H_
#define TFRT_BEF_EXECUTOR_FUNCTION_UTIL_H_

#include <array>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/value.h"
#include "tfrt/support/error_util.h"

namespace tfrt {

// A utility function to make invoking a sync tfrt::Function, similar
// to invoking a C++ function.
//
// Example:
//
// Suppose a tfrt::Function foo takes two integer arguments and returns a float
// value, we can invoke this function as follows:
//
//   ExecutionContext exec_ctx = ...;  // Create an ExecutionContext.
//   Expected<float> result = InvokeFunction<float>(foo, exec_ctx, int1, int2);
//
//   if (!result) {
//      // an error occurred.
//      std::cout << "Error: " << result.takeError();
//   } else {
//     float result_value = *result;
//   }
//
// Similarly, for functions that have two return values, we can invoke the
// function as follows.
//   Expected<std::tuple<float, float>> result =
//                InvokeFunction<float, float>(foo, exec_ctx, int1, int2);
//
// If the function foo has no return value, we can invoke it as follows.
//   Error error = InvokeFunction<void>(foo, exec_ctx, int1, int2);

namespace internal {

template <typename... Args, size_t N>
Error InvokeSyncFunctionHelper(const Function& function,
                               const ExecutionContext& exec_ctx,
                               std::array<Value, N>* results, Args&&... args) {
  assert(function.function_kind() == tfrt::FunctionKind::kSyncBEFFunction);

  static constexpr size_t kNArgs = sizeof...(Args);
  if (function.num_arguments() != kNArgs ||
      function.num_results() != results->size()) {
    return MakeStringError("Invalid signature for function ", function.name(),
                           ". Expected ", kNArgs, " arguments and ",
                           results->size(), " results, but get ",
                           function.num_arguments(), " arguments and ",
                           function.num_results(), " results");
  }

  std::array<Value, sizeof...(args)> arg_values{
      Value(&args, Value::PointerPayload{})...};
  std::array<Value*, sizeof...(args)> arg_value_ptrs;
  std::transform(arg_values.begin(), arg_values.end(), arg_value_ptrs.begin(),
                 [](auto& value) { return &value; });

  std::array<Value*, N> result_value_ptrs;
  std::transform(results->begin(), results->end(), result_value_ptrs.begin(),
                 [](auto& value) { return &value; });

  return ExecuteSyncBEFFunction(function, exec_ctx, arg_value_ptrs,
                                result_value_ptrs);
}

// Convert an array of Values into a std::tuple.
template <typename... Results, size_t N, size_t... Is>
std::tuple<Results...> ValueArrayAsTuple(std::array<Value, N>* arr,
                                         std::index_sequence<Is...>) {
  using T = std::tuple<Results...>;
  return std::tuple<Results...>(
      std::move((*arr)[Is].template get<std::tuple_element_t<Is, T>>())...);
}

}  // namespace internal

// Case for invoking functions that return void.
template <typename Result, typename... Args,
          std::enable_if_t<std::is_void<Result>::value, bool> = true>
Error InvokeSyncFunction(const Function& function,
                         const ExecutionContext& exec_ctx, Args&&... args) {
  std::array<Value, 0> result_values;
  return internal::InvokeSyncFunctionHelper(function, exec_ctx, &result_values,
                                            std::forward<Args>(args)...);
}

// Case for invoking functions that have a single return value.
template <typename Result, typename... Args,
          std::enable_if_t<!std::is_void<Result>::value, bool> = true>
Expected<Result> InvokeSyncFunction(const Function& function,
                                    const ExecutionContext& exec_ctx,
                                    Args&&... args) {
  std::array<Value, 1> result_values;
  if (auto error = internal::InvokeSyncFunctionHelper<Args...>(
          function, exec_ctx, &result_values, std::forward<Args>(args)...)) {
    return std::move(error);
  } else {
    return std::move(result_values[0].get<Result>());
  }
}

// Case for invoking functions that have multiple return values.
template <typename... Results, typename... Args,
          std::enable_if_t<(sizeof...(Results) > 1), bool> = true>
Expected<std::tuple<Results...>> InvokeSyncFunction(
    const Function& function, const ExecutionContext& exec_ctx,
    Args&&... args) {
  static constexpr size_t NResults = sizeof...(Results);
  std::array<Value, NResults> result_values;
  if (auto error = internal::InvokeSyncFunctionHelper(
          function, exec_ctx, &result_values, std::forward<Args>(args)...)) {
    return std::move(error);
  } else {
    return internal::ValueArrayAsTuple<Results...>(
        &result_values, std::make_index_sequence<NResults>{});
  }
}

// Another utility function to make invoking a sync tfrt::Function.
//
// Example:
//
// SyncFunctionRunner<float(int, int)> func{func, host_ctx, resource_ctx};
//
// Expected<float> result = func.run(int1, int2);

template <typename>
class SyncFunctionRunner;

template <typename R, typename... Args>
class SyncFunctionRunner<R(Args...)> {
  using RunReturnT =
      std::conditional_t<std::is_void<R>::value, Error, Expected<R>>;

 public:
  SyncFunctionRunner(const Function* function, HostContext* host,
                     ResourceContext* resource_ctx)
      : function_{function}, host_{host}, resource_ctx_{resource_ctx} {
    assert(function_->function_kind() == tfrt::FunctionKind::kSyncBEFFunction);
  }

  RunReturnT run(Args... args) const {
    auto req_ctx = RequestContextBuilder(host_, resource_ctx_).build();
    if (!req_ctx) {
      return MakeStringError("Failed to make RequestContext ",
                             req_ctx.takeError());
    }

    ExecutionContext exec_ctx{std::move(*req_ctx)};
    return InvokeSyncFunction<R>(*function_, std::move(exec_ctx),
                                 std::move(args)...);
  }

 private:
  const Function* function_;
  HostContext* host_;
  ResourceContext* resource_ctx_;
};

}  // namespace tfrt

#endif  // TFRT_BEF_EXECUTOR_FUNCTION_UTIL_H_
