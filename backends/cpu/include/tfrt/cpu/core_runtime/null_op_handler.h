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

// This file declares the CreateNullOpHandler. NullOpHandler always fails on op
// execution. It is the default fallback op handler.

#ifndef TFRT_BACKENDS_CPU_CORE_RUNTIME_NULL_OP_HANDLER_H_
#define TFRT_BACKENDS_CPU_CORE_RUNTIME_NULL_OP_HANDLER_H_

#include "tfrt/core_runtime/op_handler.h"

namespace tfrt {

// The NullOpHandler can be chained to a CpuOpHandler as a fallback. It always
// fails on op execution.

llvm::Expected<OpHandler*> CreateNullOpHandler(CoreRuntime* runtime);

}  // namespace tfrt
#endif  // TFRT_BACKENDS_CPU_CORE_RUNTIME_NULL_OP_HANDLER_H_
