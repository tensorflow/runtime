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

// Helpers for tfrt_gpu kernel implementations.
#ifndef TFRT_GPU_KERNELS_KERNELS_DETAIL_H_
#define TFRT_GPU_KERNELS_KERNELS_DETAIL_H_

#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"

// TFRT_KERNEL_WITH_CHAIN_RESULT wraps the TFRT_KERNEL macro to add a return
// chain to the kernel implementation.
#define TFRT_KERNEL_WITH_CHAIN_RESULT(sync_func)                         \
  TFRT_KERNEL(tfrt::gpu::internal::WithChainResult<decltype(&sync_func), \
                                                   &sync_func>::Invoke)

namespace tfrt {
namespace gpu {

// Wraps zero-argument 'func' to destroy its captures when invoked.
//
// Use this for Run/EnqueueBlockingWork() to destroy RAII-type captures before
// the return value becomes available.
template <typename F>
auto DestroyCapturesOnInvoke(F&& func) {
  return [func = llvm::unique_function<decltype(func())()>(
              std::forward<F>(func))]() mutable {
    auto result = func();
    func = nullptr;
    return std::move(result);
  };
}

namespace internal {

// Helper template that provides an async kernel implementation from a sync one.
template <typename F, F>
struct WithChainResult;

// Return a Chain instead of void.
template <typename... Args, void (*sync_func_ptr)(Args...)>
struct WithChainResult<void (*)(Args...), sync_func_ptr> {
  static Chain Invoke(Args... args) {
    sync_func_ptr(std::forward<Args>(args)...);
    return Chain{};
  }
};

// Return an Expected<Chain> instead of an Error.
template <typename... Args, Error (*sync_func_ptr)(Args...)>
struct WithChainResult<Error (*)(Args...), sync_func_ptr> {
  static Expected<Chain> Invoke(Args... args) {
    if (auto error = sync_func_ptr(std::forward<Args>(args)...))
      return std::move(error);
    return Chain{};
  }
};

// Return Expected<tuple<T, Chain>> instead of Expected<T>.
template <typename... Args, typename Result,
          Expected<Result> (*sync_func_ptr)(Args...)>
struct WithChainResult<Expected<Result> (*)(Args...), sync_func_ptr> {
  static Expected<std::tuple<Result, Chain>> Invoke(Args... args) {
    auto result = sync_func_ptr(std::forward<Args>(args)...);
    if (!result) return result.takeError();
    return std::make_tuple(std::move(*result), Chain{});
  }
};

// Return Expected<tuple<..., Chain>> instead of Expecte<tuple<...>>.
template <typename... Args, typename... Results,
          Expected<std::tuple<Results...>> (*sync_func_ptr)(Args...)>
struct WithChainResult<Expected<std::tuple<Results...>> (*)(Args...),
                       sync_func_ptr> {
  static Expected<std::tuple<Results..., Chain>> Invoke(Args... args) {
    auto tuple = sync_func_ptr(std::forward<Args>(args)...);
    if (!tuple) return tuple.takeError();
    return std::tuple_cat(std::move(*tuple), std::make_tuple(Chain{}));
  }
};

}  // namespace internal
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_KERNELS_KERNELS_DETAIL_H_
