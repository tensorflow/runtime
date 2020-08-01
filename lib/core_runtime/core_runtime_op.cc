// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- core_runtime_op.cc -------------------------------------------------===//
//
// This file implements the CoreRuntimeOp class.
//
//===----------------------------------------------------------------------===//
#include "tfrt/core_runtime/core_runtime_op.h"

#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/tensor_handle.h"

namespace tfrt {

CoreRuntimeOp::CoreRuntimeOp() : fn_(nullptr), is_fallback_(false) {}

CoreRuntimeOp::CoreRuntimeOp(
    llvm::unique_function<void(const OpInvocation&)>&& fn, bool is_fallback)
    : fn_(std::move(fn)), is_fallback_(is_fallback) {}

// is_fallback_ is not relevant.
CoreRuntimeOp::CoreRuntimeOp(
    llvm::unique_function<void(const CompositeOpInvocation&)>&& fn)
    : native_fn_(std::move(fn)), is_fallback_(false) {}

void CoreRuntimeOp::operator()(const ExecutionContext& exec_ctx,
                               MutableArrayRef<TensorHandle> arguments,
                               const OpAttrsRef& attrs,
                               MutableArrayRef<TensorHandle> results,
                               AsyncValueRef<Chain>* chain) const {
  // TODO(fishx): Consider removing op_name from OpInvocation after migrating
  // all clients to Prepare API.
  OpInvocation invocation{string_view{}, exec_ctx, arguments,
                          attrs,         results,  chain};
  fn_(invocation);
}

void CoreRuntimeOp::operator()(const OpInvocation& invocation) const {
  // The caller must provide a chain or at least one result (or both).  Ops with
  // zero results must have a chain because they are side effecting.  This
  // ensures that we have a way to report errors to the caller.
  assert((invocation.chain || !invocation.results.empty()) &&
         "Op invocation must have results or a chain");

  fn_(invocation);
}

void CoreRuntimeOp::operator()(const CompositeOpInvocation& invocation) const {
  // The caller must provide a chain or at least one result (or both).  Ops with
  // zero results must have a chain because they are side effecting.  This
  // ensures that we have a way to report errors to the caller.
  assert((invocation.chain || !invocation.results.empty()) &&
         "Op invocation must have results or a chain");

  native_fn_(invocation);
}

}  // namespace tfrt
