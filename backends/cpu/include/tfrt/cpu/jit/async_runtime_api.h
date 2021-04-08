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

// Set the AsyncRuntime host context that will be implicitly propagated to all
// async tasks.
//
// On every launch of an async task, current context will be captured, and
// restored when the task will start its execution on a different thread.
void SetAsyncRuntimeHostContext(HostContext* host_context);

// Converts MLIR Async Runtime token into the TFRT async chain, and drops the
// reference count on the token.
AsyncValueRef<Chain> ConvertAsyncTokenToChain(AsyncRuntime::Token* token);

// Extracts a payload from the MLIR Async Runtime `value` and emplaces it into
// the TFRT async value `dst` using a user provided emplace function. Drops the
// reference on the runtime value after it is no longer needed.
void ExtractAsyncValue(
    AsyncRuntime::Value* value, AsyncValue* dst,
    llvm::function_ref<void(void* storage, AsyncValue* dst)> emplace_fn);

// Builds a symbol map from the Async Runtime API functions/
llvm::orc::SymbolMap AsyncRuntimeApiSymbolMap(
    llvm::orc::MangleAndInterner mangle);

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_JIT_ASYNC_RUNTIME_API_H_
