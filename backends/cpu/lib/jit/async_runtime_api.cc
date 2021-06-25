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

// MLIR Async Runtime API integration with TFRT based AsyncRuntime.

#include "tfrt/cpu/jit/async_runtime_api.h"

#include <cstddef>
#include <iostream>

#include "tfrt/cpu/jit/async_runtime.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {
namespace cpu {
namespace jit {

namespace {
// Always keep the current active async runtime in a thread local variable.
static thread_local AsyncRuntime async_runtime_context;

static_assert(std::is_trivially_destructible<AsyncRuntime>::value,
              "AsyncRuntime must be trivially destructible");

// This is an arbitrary limitation, to make sure that AsyncRuntime would not
// become expensive to copy unnoticed.
static_assert(sizeof(AsyncRuntime) == sizeof(void *),
              "AsyncRuntime must hold only a host context pointer");

AsyncRuntime &GetAsyncRuntimeContext() {
  assert(async_runtime_context.host_context() != nullptr);
  return async_runtime_context;
}
}  // namespace

void SetAsyncRuntimeHostContext(HostContext *host_context) {
  assert(host_context != nullptr);
  async_runtime_context = AsyncRuntime(host_context);
}

AsyncValueRef<Chain> ConvertAsyncTokenToChain(AsyncRuntime::Token *token) {
  auto *async_value = AsyncRuntime::GetAsyncValue(token);
  auto out_chain = AsyncValueRef<Chain>(FormRef(async_value));
  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token));
  return out_chain;
}

void ExtractAsyncValue(
    AsyncRuntime::Value *value, AsyncValue *dst,
    llvm::function_ref<void(void *storage, AsyncValue *dst)> emplace_fn) {
  auto *async_value = AsyncRuntime::GetAsyncValue(value);

  // Fast path if async value is already available.
  if (async_value->IsAvailable()) {
    void *storage = AsyncRuntime::GetStorage(value);
    emplace_fn(storage, dst);
    AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value));
    return;
  }

  // Wait for the async value completion, and emplace the `dst`.
  async_value->AndThen([value, emplace_fn, dst = FormRef(dst)]() {
    void *storage = AsyncRuntime::GetStorage(value);
    emplace_fn(storage, dst.get());
    AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value));
  });
}

void ExtractAsyncValue(
    AsyncRuntime::Value *value, AsyncValue *dst, void *context,
    llvm::function_ref<void(void *storage, AsyncValue *dst, void *context)>
        emplace_fn) {
  auto *async_value = AsyncRuntime::GetAsyncValue(value);

  // Fast path if async value is already available.
  if (async_value->IsAvailable()) {
    void *storage = AsyncRuntime::GetStorage(value);
    emplace_fn(storage, dst, context);
    AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value));
    return;
  }

  // Wait for the async value completion, and emplace the `dst`.
  async_value->AndThen([value, emplace_fn, context, dst = FormRef(dst)]() {
    void *storage = AsyncRuntime::GetStorage(value);
    emplace_fn(storage, dst.get(), context);
    AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value));
  });
}

llvm::orc::SymbolMap AsyncRuntimeApiSymbolMap(
    llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;

  auto bind = [&](llvm::StringRef name, auto symbol_ptr) {
    symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(symbol_ptr), llvm::JITSymbolFlags());
  };

  bind("mlirAsyncRuntimeAddRef", &mlir::runtime::mlirAsyncRuntimeAddRef);
  bind("mlirAsyncRuntimeDropRef", &mlir::runtime::mlirAsyncRuntimeDropRef);
  bind("mlirAsyncRuntimeExecute", &mlir::runtime::mlirAsyncRuntimeExecute);
  bind("mlirAsyncRuntimeGetValueStorage",
       &mlir::runtime::mlirAsyncRuntimeGetValueStorage);
  bind("mlirAsyncRuntimeCreateToken",
       &mlir::runtime::mlirAsyncRuntimeCreateToken);
  bind("mlirAsyncRuntimeCreateValue",
       &mlir::runtime::mlirAsyncRuntimeCreateValue);
  bind("mlirAsyncRuntimeEmplaceToken",
       &mlir::runtime::mlirAsyncRuntimeEmplaceToken);
  bind("mlirAsyncRuntimeSetTokenError",
       &mlir::runtime::mlirAsyncRuntimeSetTokenError);
  bind("mlirAsyncRuntimeIsTokenError",
       &mlir::runtime::mlirAsyncRuntimeIsTokenError);
  bind("mlirAsyncRuntimeEmplaceValue",
       &mlir::runtime::mlirAsyncRuntimeEmplaceValue);
  bind("mlirAsyncRuntimeSetValueError",
       &mlir::runtime::mlirAsyncRuntimeSetValueError);
  bind("mlirAsyncRuntimeIsValueError",
       &mlir::runtime::mlirAsyncRuntimeIsValueError);
  bind("mlirAsyncRuntimeIsGroupError",
       &mlir::runtime::mlirAsyncRuntimeIsGroupError);
  bind("mlirAsyncRuntimeAwaitToken",
       &mlir::runtime::mlirAsyncRuntimeAwaitToken);
  bind("mlirAsyncRuntimeAwaitValue",
       &mlir::runtime::mlirAsyncRuntimeAwaitValue);
  bind("mlirAsyncRuntimeAwaitTokenAndExecute",
       &mlir::runtime::mlirAsyncRuntimeAwaitTokenAndExecute);
  bind("mlirAsyncRuntimeAwaitValueAndExecute",
       &mlir::runtime::mlirAsyncRuntimeAwaitValueAndExecute);
  bind("mlirAsyncRuntimeCreateGroup",
       &mlir::runtime::mlirAsyncRuntimeCreateGroup);
  bind("mlirAsyncRuntimeAddTokenToGroup",
       &mlir::runtime::mlirAsyncRuntimeAddTokenToGroup);
  bind("mlirAsyncRuntimeAwaitAllInGroup",
       &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroup);
  bind("mlirAsyncRuntimeAwaitAllInGroupAndExecute",
       &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroupAndExecute);
  bind("mlirAsyncRuntimePrintCurrentThreadId",
       &mlir::runtime::mlirAsyncRuntimePrintCurrentThreadId);

  return symbol_map;
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace runtime {

using AsyncRuntime = ::tfrt::cpu::jit::AsyncRuntime;
using AsyncRuntimeObject = ::tfrt::cpu::jit::AsyncRuntimeObject;

// Adds references to reference counted runtime object.
void mlirAsyncRuntimeAddRef(RefCountedObjPtr ptr, int32_t count) {
  AsyncRuntimeObject *obj = static_cast<AsyncRuntimeObject *>(ptr);
  AsyncRuntime::AddRef(obj, count);
}

// Drops references from reference counted runtime object.
void mlirAsyncRuntimeDropRef(RefCountedObjPtr ptr, int32_t count) {
  AsyncRuntimeObject *obj = static_cast<AsyncRuntimeObject *>(ptr);
  AsyncRuntime::DropRef(obj, count);
}

// Create a new `async.token` in not-ready state.
AsyncToken *mlirAsyncRuntimeCreateToken() {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  return runtime.CreateToken();
}

// Creates a new `async.value` in not-ready state.
AsyncValue *mlirAsyncRuntimeCreateValue(int32_t size) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  return runtime.CreateValue(size, /*alignment=*/alignof(std::max_align_t));
}

// Create a new `async.group` in empty state.
AsyncGroup *mlirAsyncRuntimeCreateGroup(int64_t size) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  return runtime.CreateGroup(size);
}

int64_t mlirAsyncRuntimeAddTokenToGroup(AsyncToken *token, AsyncGroup *group) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  return runtime.AddTokenToGroup(group, token);
}

bool mlirAsyncRuntimeIsGroupError(AsyncGroup *group) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  return runtime.IsError(group);
}

void mlirAsyncRuntimeEmplaceToken(AsyncToken *token) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime.SetAvailable(token);
}

void mlirAsyncRuntimeSetTokenError(AsyncToken *token) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime.SetError(token);
}

bool mlirAsyncRuntimeIsTokenError(AsyncToken *token) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  return runtime.IsError(token);
}

void mlirAsyncRuntimeAwaitToken(AsyncToken *token) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime.AwaitToken(token);
}

void mlirAsyncRuntimeAwaitAllInGroup(AsyncGroup *group) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime.AwaitGroup(group);
}

ValueStorage mlirAsyncRuntimeGetValueStorage(AsyncValue *value) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  return runtime.GetStorage(value);
}

void mlirAsyncRuntimeEmplaceValue(AsyncValue *value) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime.SetAvailable(value);
}

void mlirAsyncRuntimeSetValueError(AsyncValue *value) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime.SetError(value);
}

bool mlirAsyncRuntimeIsValueError(AsyncValue *value) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  return runtime.IsError(value);
}

void mlirAsyncRuntimeAwaitValue(AsyncValue *value) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime.AwaitValue(value);
}

void mlirAsyncRuntimeExecute(CoroHandle handle, CoroResume resume) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime.Execute([resume, handle, host = runtime.host_context()]() {
    ::tfrt::cpu::jit::SetAsyncRuntimeHostContext(host);
    (*resume)(handle);
  });
}

void mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *token, CoroHandle handle,
                                          CoroResume resume) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime.AwaitToken(token, [handle, resume, host = runtime.host_context()]() {
    ::tfrt::cpu::jit::SetAsyncRuntimeHostContext(host);
    (*resume)(handle);
  });
}

void mlirAsyncRuntimeAwaitValueAndExecute(AsyncValue *value, CoroHandle handle,
                                          CoroResume resume) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime.AwaitValue(value, [handle, resume, host = runtime.host_context()]() {
    ::tfrt::cpu::jit::SetAsyncRuntimeHostContext(host);
    (*resume)(handle);
  });
}

void mlirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *group,
                                               CoroHandle handle,
                                               CoroResume resume) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime.AwaitGroup(group, [handle, resume, host = runtime.host_context()]() {
    ::tfrt::cpu::jit::SetAsyncRuntimeHostContext(host);
    (*resume)(handle);
  });
}

//===----------------------------------------------------------------------===//
// Small async runtime support library for testing.
//===----------------------------------------------------------------------===//

void mlirAsyncRuntimePrintCurrentThreadId() {
  static thread_local std::thread::id thisId = std::this_thread::get_id();
  std::cout << "Current thread id: " << thisId << std::endl;
}

}  // namespace runtime
}  // namespace mlir
