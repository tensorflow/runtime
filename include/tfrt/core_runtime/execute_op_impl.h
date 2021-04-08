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

// This library contains the implementations of corert.executeop.

#ifndef TFRT_CORE_RUNTIME_EXECUTE_OP_IMPL_H_
#define TFRT_CORE_RUNTIME_EXECUTE_OP_IMPL_H_

#include "tfrt/support/forward_decls.h"

namespace tfrt {

class Chain;
class AggregateAttr;
class ExecutionContext;
class AsyncValue;
class CoreRuntimeOp;
class OpAttrs;
class Value;
class TensorHandle;
template <typename T>
class RepeatedSyncArguments;
class SyncKernelFrame;

// Set up `op_attrs` with binary attributes in `op_attr_array`.
// TypedAttributeTrait is required in ops' ODS to use this function on their
// attributes.
void SetUpOpAttrs(AggregateAttr op_attr_array, OpAttrs *op_attrs);

// Set up `op_attrs` with binary attributes in `op_attr_func_array`.
// `op_attr_func_array` is an array of string that denotes function attributes.
void SetUpOpFuncAttrs(AggregateAttr op_func_attr_array, OpAttrs *op_attrs);

// ExecuteOpImpl is the common implementation used by ExecuteOpSeq and
// ExecuteOp. The `op_chain` is the input/output parameter for sequencing op
// execution. `op_chain` can be nullptr, which means it need not be sequenced.
// `op_func_attr_array` is an optional array that contains function attributes.
void ExecuteOpImpl(CoreRuntimeOp op, ArrayRef<AsyncValue *> args,
                   AsyncValueRef<Chain> *op_chain,
                   MutableArrayRef<RCReference<AsyncValue>> results,
                   AggregateAttr op_attr_array,
                   AggregateAttr op_func_attr_array,
                   const ExecutionContext &exec_ctx);

void ExecuteOpImplSync(const CoreRuntimeOp &op,
                       RepeatedSyncArguments<TensorHandle> args,
                       AsyncValueRef<Chain> *op_chain, SyncKernelFrame *frame,
                       AggregateAttr op_attr_array,
                       const ExecutionContext &exec_ctx);

void AsyncWaitForResultsFromTensorHandles(
    MutableArrayRef<RCReference<AsyncValue>> results,
    MutableArrayRef<TensorHandle> result_ths);

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_EXECUTE_OP_IMPL_H_
