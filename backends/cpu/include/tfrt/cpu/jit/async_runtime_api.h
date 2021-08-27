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

#ifndef TFRT_BACKENDS_CPU_JIT_ASYNC_RUNTIME_API_H_
#define TFRT_BACKENDS_CPU_JIT_ASYNC_RUNTIME_API_H_

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "tfrt/cpu/jit/async_runtime.h"
#include "tfrt/host_context/async_value_ref.h"

namespace tfrt {

class Chain;

namespace cpu {
namespace jit {

// Set the AsyncRuntime that will be implicitly propagated to all async tasks.
//
// On every launch of an async task, current async runtime will be captured, and
// restored when the task will start its execution on a different thread.
void SetAsyncRuntime(AsyncRuntime runtime);

// Converts MLIR Async Runtime token into the TFRT async chain, and drops the
// reference count on the token.
AsyncValueRef<Chain> ConvertAsyncTokenToChain(AsyncRuntime::Token* token);

// Extracts a payload from the MLIR Async Runtime `value` and emplaces it into
// the TFRT async value `dst` using a user provided emplace function. Drops the
// reference on the runtime value after it is no longer needed.
void ExtractAsyncValue(
    AsyncRuntime::Value* value, AsyncValue* dst,
    llvm::function_ref<void(void* storage, AsyncValue* dst)> emplace_fn);

// A version of the `ExtractAsyncValue` function defined above that takes an
// additional opaque pointer that will be passed to the emplace function when
// async value will become ready. It is the caller responsibility to ensure that
// the pointed object will stay alive.
void ExtractAsyncValue(
    AsyncRuntime::Value* value, AsyncValue* dst, void* context,
    llvm::function_ref<void(void* storage, AsyncValue* dst, void*)> emplace_fn);

// Builds a symbol map from the Async Runtime API functions.
llvm::orc::SymbolMap AsyncRuntimeApiSymbolMap(
    llvm::orc::MangleAndInterner mangle);

// Builds a symbol map to override malloc/alligned_alloc/free function with a
// calls to the host runtime.
// TODO(ezhulenev): Currently we just directly forward to stdlib functions,
// because HostContext can only do sized deallocation.
// TODO(ezhulenev): Strictly speaking memory allocation is not a part of async
// runtime, however currently the only way to get to the implicitly propagated
// host context from the JIT compiled functions is async runtime. Restructure
// codegen<->TFRT interaction API to separate different aspects.
llvm::orc::SymbolMap AsyncRuntimeMemoryAllocationSymbolMap(
    llvm::orc::MangleAndInterner mangle);

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_JIT_ASYNC_RUNTIME_API_H_
