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

//===- thread_pool_device.h -------------------------------------*- C++ -*-===//
//
// This file implements wrapping of HostContext into an Eigen ThreadPoolDevice,
// that can be later used to parallelize Eigen expression evaluation.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_COMMON_COMPAT_EIGEN_THREAD_POOL_DEVICE_H_
#define TFRT_BACKENDS_COMMON_COMPAT_EIGEN_THREAD_POOL_DEVICE_H_

#define EIGEN_USE_THREADS

#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_frame.h"
#include "tfrt/host_context/shared_context.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace tfrt {
namespace compat {

//===----------------------------------------------------------------------===//
// Context that manages Eigen thread pool and ThreadPoolDevice lifetime.
//===----------------------------------------------------------------------===//

class EigenHostContext : public SharedContext {
 public:
  // TODO(ezhulenev): Pass a custom Eigen::Allocator that wraps HostContext
  // AllocateBytes/DeallocateBytes.
  explicit EigenHostContext(HostContext* host_context)
      : host_context_(host_context),
        thread_pool_(host_context),
        device_(&thread_pool_, thread_pool_.NumThreads()) {}

  EigenHostContext(const EigenHostContext&) = delete;
  void operator=(const EigenHostContext&) = delete;

  const Eigen::ThreadPoolInterface& ThreadPool() const { return thread_pool_; }
  const Eigen::ThreadPoolDevice& Device() const { return device_; }

  HostContext* host() const { return host_context_; };

 private:
  //===--------------------------------------------------------------------===//
  // Eigen::ThreadPoolInterface implementation that wraps HostContext.
  //===--------------------------------------------------------------------===//

  class EigenHostContextThreadPool : public Eigen::ThreadPoolInterface {
   public:
    explicit EigenHostContextThreadPool(HostContext* host_context)
        : host_context_(host_context) {}

    // Submits a closure to be run by a thread in the pool.
    void Schedule(std::function<void()> fn) override {
      host_context_->EnqueueWork(std::move(fn));
    }

    // Returns the number of threads in the pool.
    int NumThreads() const override {
      return host_context_->GetNumWorkerThreads();
    }

    // Eigen expressions that are using this thread pool must not rely on
    // CurrentThreadId(), and should use Eigen::ThreadLocal instead.
    int CurrentThreadId() const override {
      assert(false && "CurrentThreadId is not supported by HostContext");
      return -1;
    }

   private:
    HostContext* const host_context_;  // Must outlive *this.
  };

  HostContext* host_context_;
  EigenHostContextThreadPool thread_pool_;
  Eigen::ThreadPoolDevice device_;
};

namespace internal {
// std::is_invocable requires C++17.
// https://en.cppreference.com/w/cpp/types/is_invocable
template <typename F, typename... Args>
struct is_invocable
    : std::is_constructible<
          std::function<void(Args...)>,
          std::reference_wrapper<typename std::remove_reference<F>::type>> {};
}  // namespace internal

// Syntactic sugar for extending lifetime of the DenseHostTensor buffer.
//
// Example:
//
//   DenseHostTensor* arg0 = ...;
//   DenseHostTensor* arg1 = ...;
//   ...
//   auto keep_args = KeepBuffers::alive(arg0, arg1, ...);
//
struct KeepBuffers {
  template <typename... DenseHostTensors>
  static auto alive(DenseHostTensors... tensors)
      -> std::array<RCReference<HostBuffer>, sizeof...(DenseHostTensors)> {
    constexpr size_t size = sizeof...(DenseHostTensors);
    std::array<const DenseHostTensor*, size> tensors_arr{tensors...};
    std::array<RCReference<HostBuffer>, size> buffers_arr;
    for (size_t i = 0; i < size; ++i)
      buffers_arr[i] = tensors_arr[i]->buffer().CopyRef();
    return buffers_arr;
  }
};

// Evaluates Eigen expression `expr` and assigns result to the output `out`
// (which is also an Eigen expression, typically Eigen::TensorMap) using
// asynchronous execution. This function returns immediately without blocking a
// caller thread. When expression evaluation is completed, Eigen will invoke
// `done` callback from the thread that computed the last task.
//
// It is possible that small expressions will be evaluated in a caller thread,
// without enqueuing work into the HostContext. In this case callbacks will be
// called from the caller thread, after completing expression evaluation.
//
// WARNING: All values that must be alive for the duration of Eigen expression
// evaluation, must be moved into the `done` callback before passing it to the
// `AsyncAssign`. Also all reference counted types must be moved into the
// callback, to prevent accidental call to a DropRef() (and maybe deallocation)
// when kernel function finishes its execution.
template <
    typename Output, typename Expr, typename DoneCallback,
    typename = std::enable_if_t<internal::is_invocable<DoneCallback>::value>>
void AsyncAssign(const EigenHostContext& ctx, Output out, Expr expr,
                 DoneCallback done) {
  auto callback = [done = std::move(done)]() mutable { done(); };
  out.device(ctx.Device(), std::move(callback)) = expr;
}

// Syntactic sugar for the `AsyncAssign` defined above, that does output
// chain allocation.
//
// WARNING: Caller is responsible for capturing all reference-counted object
// required for the duration of Eigen expression evaluation (lifetime extension)
// into an instance of `ArgLifetimeExtension` (see `KeepBuffers` or
// `RAIIKernelFrame`).
template <typename Output, typename Expr, typename ArgLifetimeExtension,
          typename = std::enable_if_t<
              !internal::is_invocable<ArgLifetimeExtension>::value>>
AsyncValueRef<Chain> AsyncAssign(const EigenHostContext& ctx, Output out,
                                 Expr expr, ArgLifetimeExtension args) {
  auto chain = ctx.host()->MakeUnconstructedAsyncValueRef<Chain>();
  auto callback = [args = std::move(args), chain = chain.CopyRef()]() {
    chain.emplace();
  };
  out.device(ctx.Device(), std::move(callback)) = expr;
  return chain;
}

}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_COMPAT_EIGEN_THREAD_POOL_DEVICE_H_
