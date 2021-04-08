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

// This file declares and implements OpInvocation.

#ifndef TFRT_CORE_RUNTIME_OP_INVOCATION_H_
#define TFRT_CORE_RUNTIME_OP_INVOCATION_H_

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class Chain;
class OpAttrsRef;
class TensorHandle;

// This struct encapsulates the data needed to represent an op invocation,
// essentially a bundle of information passed to CoreRuntime::Execute.  This is
// an internal implementation detail of Device and AggregateOpHandler, it isn't
// intended to be used by normal clients of the CoreRuntime API.
//
// This struct is intentionally non-copyable and non-movable because it contains
// MutableArrayRef's and string_views that often point into the caller's stack.
struct OpInvocation {
  // This is the name of the op to invoke, e.g. "tf.Add".
  string_view op_name;

  // The ExecutionContext of the op invocation.
  ExecutionContext exec_ctx;

  // This is the input arguments to an op invocation.  Note that this is a
  // MutableArrayRef because invocation of the op will generally take (and
  // null out) the input TensorHandles in order to implement in-place
  // optimizations.
  MutableArrayRef<TensorHandle> arguments;

  // The attributes for the op invocation.
  const OpAttrsRef& attrs;

  // Result TensorHandles to be provided by the op invocation.
  MutableArrayRef<TensorHandle> results;

  // This points to a chain value that the op execution should depend on if it
  // has side effects.  Invocation of a side effecting op will read from this
  // pointer, and then write the result chain which completes when the op side
  // effects are done.
  AsyncValueRef<Chain>* chain;

  // Non-copyable and non-movable because this includes unsafe pointers to the
  // caller stack.
  OpInvocation(const OpInvocation&) = delete;
  OpInvocation(OpInvocation&&) = delete;
  OpInvocation& operator=(const OpInvocation&) = delete;
  OpInvocation& operator=(OpInvocation&&) = delete;
};

// TODO(b/161062314): Assess whether to use this struct consistently for
// composite op handling.
struct CompositeOpInvocation {
  // The ExecutionContext of the op invocation.
  ExecutionContext exec_ctx;

  // This is the input arguments to an op invocation.  Note that this is a
  // MutableArrayRef because invocation of the op will generally take (and
  // null out) the input TensorHandles in order to implement in-place
  // optimizations.
  ArrayRef<RCReference<AsyncValue>> arguments;

  // Result TensorHandles to be provided by the op invocation.
  MutableArrayRef<RCReference<AsyncValue>> results;

  // This points to a chain value that the op execution should depend on if it
  // has side effects.  Invocation of a side effecting op will read from this
  // pointer, and then write the result chain which completes when the op side
  // effects are done.
  AsyncValueRef<Chain>* chain;

  // Non-copyable and non-movable because this includes unsafe pointers to the
  // caller stack.
  CompositeOpInvocation(const CompositeOpInvocation&) = delete;
  CompositeOpInvocation(CompositeOpInvocation&&) = delete;
  CompositeOpInvocation& operator=(const CompositeOpInvocation&) = delete;
  CompositeOpInvocation& operator=(CompositeOpInvocation&&) = delete;
};

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_OP_INVOCATION_H_
