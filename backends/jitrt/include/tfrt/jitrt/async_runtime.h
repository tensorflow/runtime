/*
 * Copyright 2022 The TensorFlow Runtime Authors
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

// MLIR Async Runtime implemented on top of TFRT HostContext and host
// concurrency primitives.

#ifndef TFRT_BACKENDS_JITRT_ASYNC_RUNTIME_H_
#define TFRT_BACKENDS_JITRT_ASYNC_RUNTIME_H_

#define EIGEN_USE_THREADS

#include <cstddef>
#include <functional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/ExecutionEngine/AsyncRuntime.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/support/forward_decls.h"
#include "unsupported/Eigen/CXX11/ThreadPool"  // from @eigen_archive

namespace tfrt {
namespace jitrt {

// Forward declare a base class for async runtime objects.
class AsyncRuntimeObject;

// Async task runner abstracts over the underlying thread pool (or concurrent
// work queue) implementation.
class AsyncTaskRunner {
 public:
  using Task = std::function<void()>;
  virtual ~AsyncTaskRunner() = default;
  virtual void Schedule(Task task) = 0;
};

class AsyncRuntime {
 public:
  using Token = ::mlir::runtime::AsyncToken;
  using Value = ::mlir::runtime::AsyncValue;
  using Group = ::mlir::runtime::AsyncGroup;

  explicit AsyncRuntime(AsyncTaskRunner* runner) : runner_(runner) {
    assert(runner != nullptr && "async task runner must be not null");
  }

  // We need a default constructor to define a thread local variable for async
  // runtime passing between tasks (see implementation in async_runtime_api.cc).
  AsyncRuntime() : runner_(nullptr) {}

  // ------------------------------------------------------------------------ //
  // Async Token API.
  // ------------------------------------------------------------------------ //

  // Creates a new token in not-ready state.
  Token* CreateToken();

  // Switches the token to the available state and runs all the awaiters.
  void SetAvailable(Token* token);

  // Switches the token to the error state and runs all the awaiters.
  void SetError(Token* token);

  // Returns `true` if the token is in the error state.
  bool IsError(Token* token);

  // Blocks the caller thread until the token becomes ready.
  void AwaitToken(Token* token);

  // ------------------------------------------------------------------------ //
  // Async Value API.
  // ------------------------------------------------------------------------ //

  // Creates a new value in not-ready state with a storage of the given size.
  Value* CreateValue(size_t size, size_t alignment);

  // Switches the value to the available state and runs all the awaiters.
  void SetAvailable(Value* value);

  // Switches the value to the error state and runs all the awaiters.
  void SetError(Value* value);

  // Returns `true` if the value is in the error state.
  bool IsError(Value* value);

  // Blocks the caller thread until the value becomes ready.
  void AwaitValue(Value* value);

  // ------------------------------------------------------------------------ //
  // Async Group API.
  // ------------------------------------------------------------------------ //

  // Creates a new empty group.
  Group* CreateGroup(int64_t size);

  // Adds `token` to the `group`.
  size_t AddTokenToGroup(Group* group, Token* token);

  // Returns `true` if the group is in the error state (any of the tokens or
  // values added to the group is in the error state).
  bool IsError(Group* group);

  // Blocks the caller thread until the group becomes ready (all tokens that
  // were added to the group are emplaced).
  void AwaitGroup(Group* group);

  // ------------------------------------------------------------------------ //
  // Execution and continuation based resumption API.
  // ------------------------------------------------------------------------ //

  // Execute the callable `f` on a thread managed by the runtime.
  template <typename F>
  void Execute(F&& f);

  // Await operation that do not block the caller thread, but instead execute
  // the callable `F` when the token/group become ready.
  template <typename F>
  void AwaitToken(Token* token, F&& f);
  template <typename F>
  void AwaitValue(Value* value, F&& f);
  template <typename F>
  void AwaitGroup(Group* group, F&& f);

  // ------------------------------------------------------------------------ //

  // Returns a pointer to the async value storage.
  static void* GetStorage(Value* value);

  // Extracts async value that holds a chain owned by the value.
  static AsyncValue* GetAsyncValue(Value* value);

  // Extracts async value that is owned by the token.
  static AsyncValue* GetAsyncValue(Token* token);

  // Extracts async value that signals group completion.
  static AsyncValue* GetAsyncValue(Group* group);

  // Reference counting operations for the runtime objects.
  static void AddRef(AsyncRuntimeObject* obj, unsigned count = 1);
  static void DropRef(AsyncRuntimeObject* obj, unsigned count = 1);

  // Convert Token/Value/Group to AsyncRuntimeObject*;
  static AsyncRuntimeObject* ToAsyncRuntimeObject(Token* token);
  static AsyncRuntimeObject* ToAsyncRuntimeObject(Value* value);
  static AsyncRuntimeObject* ToAsyncRuntimeObject(Group* group);

  AsyncTaskRunner* runner() const { return runner_; }

 private:
  // Blocks the caller thread until awaitable async value becomes available.
  void Await(AsyncValue* awaitable);

  AsyncTaskRunner* runner_;  // must outlive *this
};

// A base class for all Async dialect types reference counted at runtime.
class AsyncRuntimeObject : public ::tfrt::ReferenceCounted<AsyncRuntimeObject> {
 public:
  using ReferenceCounted::ReferenceCounted;  // inherit constructors
  virtual ~AsyncRuntimeObject() = default;
};

template <typename F>
void AsyncRuntime::Execute(F&& f) {
  runner_->Schedule(std::forward<F>(f));
}

template <typename F>
void AsyncRuntime::AwaitToken(Token* token, F&& f) {
  AsyncRuntime::GetAsyncValue(token)->AndThen(std::forward<F>(f));
}

template <typename F>
void AsyncRuntime::AwaitValue(Value* value, F&& f) {
  AsyncRuntime::GetAsyncValue(value)->AndThen(std::forward<F>(f));
}

template <typename F>
void AsyncRuntime::AwaitGroup(Group* group, F&& f) {
  AsyncRuntime::GetAsyncValue(group)->AndThen(std::forward<F>(f));
}

// Runs async tasks by enqueing them into the host context work queue.
class HostContextAsyncTaskRunner : public AsyncTaskRunner {
 public:
  explicit HostContextAsyncTaskRunner(HostContext* host) : host_(host) {}
  void Schedule(Task task) override { EnqueueWork(host_, std::move(task)); }

 private:
  HostContext* host_;
};

// Runs async tasks by scheduling them into the Eigen thread pool.
class EigenThreadPoolAsyncTaskRunner : public AsyncTaskRunner {
 public:
  explicit EigenThreadPoolAsyncTaskRunner(
      Eigen::ThreadPoolInterface* thread_pool)
      : thread_pool_(thread_pool) {}
  void Schedule(Task task) override { thread_pool_->Schedule(std::move(task)); }

 private:
  Eigen::ThreadPoolInterface* thread_pool_;
};

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_ASYNC_RUNTIME_H_
