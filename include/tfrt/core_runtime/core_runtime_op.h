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

//===- core_runtime_op.h ----------------------------------------*- C++ -*-===//
//
// This file declares CoreRuntimeOp, responsible for executing ops.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_CORE_RUNTIME_CORE_RUNTIME_OP_H_
#define TFRT_CORE_RUNTIME_CORE_RUNTIME_OP_H_

#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class TensorHandle;
class OpAttrsRef;
class Chain;
class ExecutionContext;

// A callable op handle prepared by a specific OpHandler.
class CoreRuntimeOp {
 public:
  // Default initialized CoreRuntimeOp's are in the invalid state.
  explicit CoreRuntimeOp();
  explicit CoreRuntimeOp(llvm::unique_function<void(const OpInvocation&)>&& fn);
  CoreRuntimeOp(llvm::unique_function<void(const OpInvocation&)>&& fn,
                bool is_fallback);

  // Execute the prepared op.
  void operator()(const ExecutionContext& exec_ctx,
                  MutableArrayRef<TensorHandle> arguments,
                  const OpAttrsRef& attrs,
                  MutableArrayRef<TensorHandle> results,
                  AsyncValueRef<Chain>* chain);

  void operator()(const OpInvocation& invocation);

  explicit operator bool() const { return static_cast<bool>(fn_); }

  bool IsFallback() const { return is_fallback_; }

 private:
  llvm::unique_function<void(const OpInvocation&)> fn_;
  bool is_fallback_;
};

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_CORE_RUNTIME_OP_H_
