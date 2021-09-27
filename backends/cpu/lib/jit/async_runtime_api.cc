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

#include <stdlib.h>

#include <cstddef>
#include <iostream>
#include <type_traits>

#include "tfrt/cpu/jit/async_runtime.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/alloc.h"
#include "tfrt/support/msan.h"

namespace tfrt {
namespace cpu {
namespace jit {

namespace {
// Always keep the current active async runtime in a thread local variable.
static thread_local AsyncRuntime async_runtime;

static_assert(std::is_trivially_destructible<AsyncRuntime>::value,
              "AsyncRuntime must be trivially destructible");

static_assert(std::is_trivially_copy_assignable<AsyncRuntime>::value,
              "AsyncRuntime must be trivially copy assignable");

static_assert(std::is_trivially_copy_constructible<AsyncRuntime>::value,
              "AsyncRuntime must be trivially copy constructible");

// This is an arbitrary limitation, to make sure that AsyncRuntime would not
// become expensive to copy unnoticed.
static_assert(sizeof(AsyncRuntime) == 2 * sizeof(void *),
              "AsyncRuntime must hold only two pointers");

AsyncRuntime &GetAsyncRuntime() {
  assert(async_runtime.host_context() != nullptr);
  return async_runtime;
}
}  // namespace

void SetAsyncRuntime(AsyncRuntime runtime) {
  assert(runtime.host_context() != nullptr);
  async_runtime = runtime;
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

namespace {

void *RuntimeAlignedAlloc(size_t alignment, size_t size) {
  return AlignedAlloc(alignment, size);
}

void *RuntimeMalloc(size_t size) {
  // AlignedAlloc() requires results to be deallocated with AlignedFree().
  // Make all allocations aligned because there is only one RuntimeFree().
  // Align to the size of a pointer by default.
  return RuntimeAlignedAlloc(sizeof(void *), size);
}

void RuntimeFree(void *ptr) { return AlignedFree(ptr); }

}  // namespace

llvm::orc::SymbolMap AsyncRuntimeMemoryAllocationSymbolMap(
    llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;

  auto bind = [&](llvm::StringRef name, auto symbol_ptr) {
    symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(symbol_ptr), llvm::JITSymbolFlags());
  };

  bind("malloc", &RuntimeMalloc);
  bind("free", &RuntimeFree);
  bind("aligned_alloc", &RuntimeAlignedAlloc);

  return symbol_map;
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

// TODO(b/192775419): All pointers passed from the JIT compiled code to the
// runtime API must be marked initialized when running with msan enabled,
// because currently we do not have a way to enable sanitizer in the compiled
// code, and msan does not have any visibility into that code at runtime.

namespace mlir {
namespace runtime {

using AsyncRuntime = ::tfrt::cpu::jit::AsyncRuntime;
using AsyncRuntimeObject = ::tfrt::cpu::jit::AsyncRuntimeObject;

// Adds references to reference counted runtime object.
void mlirAsyncRuntimeAddRef(RefCountedObjPtr ptr, int64_t count) {
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&ptr, sizeof(RefCountedObjPtr));
  AsyncRuntimeObject *obj = static_cast<AsyncRuntimeObject *>(ptr);
  AsyncRuntime::AddRef(obj, count);
}

// Drops references from reference counted runtime object.
void mlirAsyncRuntimeDropRef(RefCountedObjPtr ptr, int64_t count) {
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&ptr, sizeof(RefCountedObjPtr));
  AsyncRuntimeObject *obj = static_cast<AsyncRuntimeObject *>(ptr);
  AsyncRuntime::DropRef(obj, count);
}

// Create a new `async.token` in not-ready state.
AsyncToken *mlirAsyncRuntimeCreateToken() {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  return runtime.CreateToken();
}

// Creates a new `async.value` in not-ready state.
AsyncValue *mlirAsyncRuntimeCreateValue(int64_t size) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  return runtime.CreateValue(size, /*alignment=*/alignof(std::max_align_t));
}

// Create a new `async.group` in empty state.
AsyncGroup *mlirAsyncRuntimeCreateGroup(int64_t size) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  return runtime.CreateGroup(size);
}

int64_t mlirAsyncRuntimeAddTokenToGroup(AsyncToken *token, AsyncGroup *group) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&group, sizeof(void *));
  return runtime.AddTokenToGroup(group, token);
}

bool mlirAsyncRuntimeIsGroupError(AsyncGroup *group) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&group, sizeof(void *));
  return runtime.IsError(group);
}

void mlirAsyncRuntimeEmplaceToken(AsyncToken *token) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  runtime.SetAvailable(token);
}

void mlirAsyncRuntimeSetTokenError(AsyncToken *token) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  runtime.SetError(token);
}

bool mlirAsyncRuntimeIsTokenError(AsyncToken *token) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  return runtime.IsError(token);
}

void mlirAsyncRuntimeAwaitToken(AsyncToken *token) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  runtime.AwaitToken(token);
}

void mlirAsyncRuntimeAwaitAllInGroup(AsyncGroup *group) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&group, sizeof(void *));
  runtime.AwaitGroup(group);
}

ValueStorage mlirAsyncRuntimeGetValueStorage(AsyncValue *value) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  return runtime.GetStorage(value);
}

void mlirAsyncRuntimeEmplaceValue(AsyncValue *value) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  runtime.SetAvailable(value);
}

void mlirAsyncRuntimeSetValueError(AsyncValue *value) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  runtime.SetError(value);
}

bool mlirAsyncRuntimeIsValueError(AsyncValue *value) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  return runtime.IsError(value);
}

void mlirAsyncRuntimeAwaitValue(AsyncValue *value) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  runtime.AwaitValue(value);
}

void mlirAsyncRuntimeExecute(CoroHandle handle, CoroResume resume) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  runtime.Execute([resume, handle, runtime]() {
    ::tfrt::cpu::jit::SetAsyncRuntime(runtime);
    (*resume)(handle);
  });
}

void mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *token, CoroHandle handle,
                                          CoroResume resume) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  runtime.AwaitToken(token, [handle, resume, runtime]() {
    ::tfrt::cpu::jit::SetAsyncRuntime(runtime);
    (*resume)(handle);
  });
}

void mlirAsyncRuntimeAwaitValueAndExecute(AsyncValue *value, CoroHandle handle,
                                          CoroResume resume) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  runtime.AwaitValue(value, [handle, resume, runtime]() {
    ::tfrt::cpu::jit::SetAsyncRuntime(runtime);

    (*resume)(handle);
  });
}

void mlirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *group,
                                               CoroHandle handle,
                                               CoroResume resume) {
  AsyncRuntime &runtime = ::tfrt::cpu::jit::GetAsyncRuntime();
  TFRT_MSAN_MEMORY_IS_INITIALIZED(&group, sizeof(void *));
  runtime.AwaitGroup(group, [handle, resume, runtime]() {
    ::tfrt::cpu::jit::SetAsyncRuntime(runtime);
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
