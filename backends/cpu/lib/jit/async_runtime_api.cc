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

//===- async_runtime_api.cc -------------------------------------*- C++ -*-===//
//
// MLIR Async Runtime API integration with TFRT based AsyncRuntime.
//
//===----------------------------------------------------------------------===//

#include "tfrt/cpu/jit/async_runtime_api.h"

#ifdef MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS

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
static thread_local AsyncRuntime *async_runtime_context = nullptr;

AsyncRuntime *GetAsyncRuntimeContext() {
  assert(async_runtime_context != nullptr);
  return async_runtime_context;
}
}  // namespace

void SetAsyncRuntimeContext(AsyncRuntime *runtime) {
  assert(runtime != nullptr);
  async_runtime_context = runtime;
}

// Converts MLIR Async Runtime token into the TFRT async chain.
AsyncValueRef<Chain> ConverAsyncTokenToChain(AsyncRuntime::AsyncToken *token) {
  auto *async_value = AsyncRuntime::GetAsyncValue(token);
  auto out_chain = AsyncValueRef<Chain>(FormRef(async_value));
  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token));
  return out_chain;
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
  bind("mlirAsyncRuntimeCreateToken",
       &mlir::runtime::mlirAsyncRuntimeCreateToken);
  bind("mlirAsyncRuntimeEmplaceToken",
       &mlir::runtime::mlirAsyncRuntimeEmplaceToken);
  bind("mlirAsyncRuntimeAwaitToken",
       &mlir::runtime::mlirAsyncRuntimeAwaitToken);
  bind("mlirAsyncRuntimeAwaitTokenAndExecute",
       &mlir::runtime::mlirAsyncRuntimeAwaitTokenAndExecute);
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
  AsyncRuntime *runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  AsyncToken *token = runtime->CreateToken();
  return token;
}

// Create a new `async.group` in empty state.
AsyncGroup *mlirAsyncRuntimeCreateGroup() {
  AsyncRuntime *runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  AsyncGroup *group = runtime->CreateGroup();
  return group;
}

int64_t mlirAsyncRuntimeAddTokenToGroup(AsyncToken *token, AsyncGroup *group) {
  AsyncRuntime *runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  return runtime->AddTokenToGroup(group, token);
}

// Switches `async.token` to ready state and runs all awaiters.
void mlirAsyncRuntimeEmplaceToken(AsyncToken *token) {
  AsyncRuntime *runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime->EmplaceToken(token);
}

void mlirAsyncRuntimeAwaitToken(AsyncToken *token) {
  AsyncRuntime *runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime->AwaitToken(token);
}

void mlirAsyncRuntimeAwaitAllInGroup(AsyncGroup *group) {
  AsyncRuntime *runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime->AwaitGroup(group);
}

void mlirAsyncRuntimeExecute(CoroHandle handle, CoroResume resume) {
  AsyncRuntime *runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime->Execute([resume, handle, runtime]() {
    ::tfrt::cpu::jit::SetAsyncRuntimeContext(runtime);
    (*resume)(handle);
  });
}

void mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *token, CoroHandle handle,
                                          CoroResume resume) {
  AsyncRuntime *runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime->AwaitToken(token, [handle, resume, runtime]() {
    ::tfrt::cpu::jit::SetAsyncRuntimeContext(runtime);
    (*resume)(handle);
  });
}

void mlirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *group,
                                               CoroHandle handle,
                                               CoroResume resume) {
  AsyncRuntime *runtime = ::tfrt::cpu::jit::GetAsyncRuntimeContext();
  runtime->AwaitGroup(group, [handle, resume, runtime]() {
    ::tfrt::cpu::jit::SetAsyncRuntimeContext(runtime);
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

#endif  // MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
