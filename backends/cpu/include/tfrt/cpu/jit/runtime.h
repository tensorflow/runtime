/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

#ifndef TFRT_BACKENDS_CPU_JIT_RUNTIME_H_
#define TFRT_BACKENDS_CPU_JIT_RUNTIME_H_

#include <stdint.h>

namespace tfrt {
namespace cpu {
namespace jit {
namespace runtime {

//===----------------------------------------------------------------------===//
// Runtime <-> Codegen integration API.
//===----------------------------------------------------------------------===//

// This API enables JIT compiled code (kernels) to call back into the underlying
// runtime for:
//
//  - User friendly error reporting integrated with the high level execution
//    model (errors do not crash the compiled binary).
//  - Memory allocation and buffer forwarding that relies on fine grained
//    information available only to the runtime (e.g. reference count).
//
// CPURT compilation pipeline sets up passes to convert the regular functions
// to the so called "kernels" integrated with the runtime using the API defined
// below, e.g. instead of conventional returns all results are returned via the
// runtimeGetResultStorage API. At MLIR level these operations correspond to the
// `rt` dialect, and converted to LLVM using the `rt-to-llvm` conversion pass.
//
// Runtime errors are reported back to the runtime via the runtimeSetError API.
// The compilation pipeline will automatically convert assertions in the kernel
// function into run time errors.

// Opaque runtime kernel context passed as the first operand to compiled kernels
// and passed back to all runtime API methods.
typedef struct KernelContext KernelContext;

// Returns a pointer to the memory location of the result with the given index.
extern "C" void *runtimeGetResultStorage(KernelContext *, int64_t);

// Sets kernel context to an error state.
extern "C" void runtimeSetError(KernelContext *, const char *);

}  // namespace runtime
}  // namespace jit
}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_JIT_RUNTIME_H_
