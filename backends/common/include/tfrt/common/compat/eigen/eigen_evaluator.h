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

//===- eigen_evaluator.h ----------------------------------------*- C++ -*-===//
//
// This file implements AsyncEigenEvaluator and SyncEigenEvaluator.
// AsyncEigenEvaluator and SyncEigenEvaluator have the same interface and are
// intended to be used a template argument to give the user function both the
// async and sync evaluation semantitc.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_COMMON_COMPAT_EIGEN_EVAULATOR_H_
#define TFRT_BACKENDS_COMMON_COMPAT_EIGEN_EVAULATOR_H_

#include <cassert>
#include <type_traits>

#include "./thread_pool_device.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace compat {

// AsyncEigenEvaluator use the thread pool to do the eigen operation.
class AsyncEigenEvaluator {
 public:
  using DependencyToken = AsyncValueRef<Chain>;

  explicit AsyncEigenEvaluator(HostContext* host)
      : ctx_(host->GetOrCreateSharedContext<compat::EigenHostContext>()) {}

  template <typename... DenseHostTensors>
  auto KeepAlive(DenseHostTensors&&... tensors)
      -> std::array<RCReference<HostBuffer>, sizeof...(DenseHostTensors)> {
    return KeepBuffers::alive(std::forward<DenseHostTensors>(tensors)...);
  }

  template <
      typename Output, typename Expr, typename DoneCallback,
      typename = std::enable_if_t<internal::is_invocable<DoneCallback>::value>>
  void Evaluate(Output out, Expr expr, DoneCallback done) {
    return AsyncAssign(ctx_, std::move(out), std::move(expr), std::move(done));
  }

  template <typename Output, typename Expr, typename ArgLifetimeExtension,
            typename = std::enable_if_t<
                !internal::is_invocable<ArgLifetimeExtension>::value>>
  AsyncValueRef<Chain> Evaluate(Output out, Expr expr,
                                ArgLifetimeExtension args) {
    return AsyncAssign(ctx_, std::move(out), std::move(expr), std::move(args));
  }

  template <typename Output, typename Expr, typename ArgLifetimeExtension,
            typename = std::enable_if_t<
                !internal::is_invocable<ArgLifetimeExtension>::value>>
  AsyncValueRef<Chain> Evaluate(const AsyncValueRef<Chain>& chain, Output out,
                                Expr expr, ArgLifetimeExtension args) {
    return AsyncAssign(ctx_, chain, std::move(out), std::move(expr),
                       std::move(args));
  }

  template <typename... Args>
  DependencyToken MakeError(Args&&... args) {
    return MakeErrorAsyncValueRef(ctx_.host(),
                                  StrCat(std::forward<Args>(args)...));
  }

 private:
  const EigenHostContext& ctx_;
};

// AsyncEigenEvaluator does the eigen operation inline in the same thread.
class SyncEigenEvaluator {
  // This is not exposed to the client. The client should use `auto` to refer to
  // this type, so that the code is generic for both AsyncEigenEvaluator and
  // SyncEigenEvaluator.
  struct NoKeepAlive {};

 public:
  // This plays the same role as the AsyncValueRef<Chain> in the
  // AsyncEigenEvaluator.
  using DependencyToken = Error;

  explicit SyncEigenEvaluator(HostContext* host) {}

  // KeepAlive is a no-op for the sync evaluation.
  template <typename... DenseHostTensors>
  auto KeepAlive(DenseHostTensors&&... tensors) {
    return NoKeepAlive{};
  }

  template <
      typename Output, typename Expr, typename DoneCallback,
      typename = std::enable_if_t<internal::is_invocable<DoneCallback>::value>>
  void Evaluate(Output out, Expr expr, DoneCallback done) {
    out = expr;
    done();
  }

  template <typename Output, typename Expr>
  Error Evaluate(Output out, Expr expr, NoKeepAlive) {
    out = expr;
    return Error::success();
  }

  template <typename Output, typename Expr>
  Error Evaluate(Error& error, Output out, Expr expr, NoKeepAlive) {
    // Similar to the AsyncValueRef `chain` in the AsyncEigenEvalutor, the
    // `error` parameter is used to carry dependency only.
    assert(!error);
    out = expr;
    return Error::success();
  }

  template <typename... Args>
  Error MakeError(Args&&... args) {
    return MakeStringError(std::forward<Args>(args)...);
  }
};

}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_COMPAT_EIGEN_EVAULATOR_H_
