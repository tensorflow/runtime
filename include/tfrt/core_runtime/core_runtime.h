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

//===- core_runtime.h -------------------------------------------*- C++ -*-===//
//
// This defines CoreRuntime, the primary op-level interface to TFRT.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_CORE_RUNTIME_CORE_RUNTIME_H_
#define TFRT_CORE_RUNTIME_CORE_RUNTIME_H_

#include <functional>

#include "tfrt/support/forward_decls.h"

namespace tfrt {

template <typename T>
class AsyncValueRef;
class Chain;
class ConcurrentWorkQueue;
class DecodedDiagnostic;
class ExecutionContext;
class Function;
class HostAllocator;
class HostContext;
class OpAttrsRef;
class CoreRuntimeOp;
class OpHandler;
class TensorHandle;

class CoreRuntime final {
 public:
  // TODO(fishx): Avoid hard-coded type string.
  static const char* kTensorHandleType;

  // Create a CoreRuntime object. `op_handler_chains` is an array of strings
  // that specifies devices to register including their fallback op handlers.
  // For example, if we need a cpu device that falls back to tf device, we can
  // use "cpu|tf".
  static llvm::Expected<std::unique_ptr<CoreRuntime>> Create(
      std::function<void(const DecodedDiagnostic&)> diag_handler,
      std::unique_ptr<HostAllocator> allocator,
      std::unique_ptr<ConcurrentWorkQueue> work_queue,
      ArrayRef<std::string> op_handler_chains);

  CoreRuntime(std::function<void(const DecodedDiagnostic&)> diag_handler,
              std::unique_ptr<HostAllocator> allocator,
              std::unique_ptr<ConcurrentWorkQueue> work_queue);

  ~CoreRuntime();

  HostContext* GetHostContext();

  // Return the CoreRuntime instance that owns the specified HostContext.  This
  // returns null if the specified HostContext isn't owned by a CoreRuntime.
  static CoreRuntime* GetFromHostContext(HostContext* context);

  // Return the non-owning reference to the op_handler with `name`. Return
  // nullptr if no such op_handler is found.
  OpHandler* GetOpHandler(string_view name) const;

  // Execute the op specified by 'op_name' with this op_handler, filling in
  // `results` with any TensorHandle results.
  //
  // Executing an op is generally an asynchronous operation that will produce
  // TensorHandle's for the results of executing the operation (which can be
  // asynchronous errors), but op dispatch can also synchronously fail.  This
  // happens when the specified operation is not known, or when synchronously
  // executed logic (e.g. metadata functions) detect an error.  When this
  // happens, this method emits the corresponding error using the normal
  // diagnostic machinery and fills the `results` TensorHandle's and result
  // chain with the error value.
  //
  // Note that this takes the input argument TensorHandle's and is allowed to
  // destructively mutate them.  This is useful in the common case of expression
  // trees like (x+y)*z, but there are cases where the caller will need to
  // duplicate the TensorHandle (using CopyRef()) method if it needs the
  // TensorHandle to be valid after the execute call.
  //
  // If the client does not need the location information in error messages, the
  // client can set `loc` to a default constructed Location, Loation().
  void Execute(const ExecutionContext& exec_ctx, string_view op_name,
               OpHandler* op_handler, MutableArrayRef<TensorHandle> arguments,
               const OpAttrsRef& attrs, MutableArrayRef<TensorHandle> results,
               AsyncValueRef<Chain>* chain);

  // [Experimental]
  // Return an CoreRuntimeOp (a callable) that clients can use to execute an op
  // directly, or an error if it cannot find the op in the op registry.
  Expected<CoreRuntimeOp> MakeOp(string_view op_name, OpHandler* op_handler);

  // [Experimental]
  // Construct and return a CoreRuntimeOp (a callable) from a Function. To
  // handle side effects, the first argument must be an input chain, and the
  // first result an output chain.
  //
  // This Function must take TensorHandle as inputs and produce TensorHandle
  // as output. Right now the Function cannot have side effect since it cannot
  // handle chain properly.
  Expected<CoreRuntimeOp> MakeCompositeOp(const Function* fn);

  // Similar to the above API, but this function takes and returns AsyncValues
  // of any underlying types, and is not tied to TensorHandle.
  // TODO(b/161062313): Assess whether / how to unify these 2 forms of composite
  // ops.
  Expected<CoreRuntimeOp> MakeNativeCompositeOp(const Function* fn);

  // [Experimental]
  // Transfer the ownership of an OpHandler to this core runtime object.
  // This function is intended to be called by OpHandler factory functions.
  void TakeOpHandler(std::unique_ptr<OpHandler> op_handler);

  // [Experimental]
  // Assign a name to a given OpHandler and make it accessible through
  // GetOpHandler. All OpHandlers should owned by the core runtime object or
  // stay alive during the execution.
  void RegisterOpHandler(string_view name, OpHandler* op_handler);

 private:
  friend class OpHandler;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_CORE_RUNTIME_H_
