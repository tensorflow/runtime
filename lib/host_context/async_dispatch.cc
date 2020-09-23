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

//===- async_dispatch.cc --------------------------------------------------===//
//
// This file implements functions related to asynchronous work dispatching.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/async_dispatch.h"

namespace tfrt {

void Await(const ExecutionContext& exec_ctx,
           ArrayRef<RCReference<AsyncValue>> values) {
  exec_ctx.host()->Await(values);
}

void EnqueueWork(const ExecutionContext& exec_ctx,
                 llvm::unique_function<void()> work) {
  EnqueueWork(exec_ctx.host(), std::move(work));
}

bool EnqueueBlockingWork(const ExecutionContext& exec_ctx,
                         llvm::unique_function<void()> work) {
  return exec_ctx.host()->EnqueueBlockingWork(std::move(work));
}

bool RunBlockingWork(const ExecutionContext& exec_ctx,
                     llvm::unique_function<void()> work) {
  return exec_ctx.host()->RunBlockingWork(std::move(work));
}

void RunWhenReady(const ExecutionContext& exec_ctx,
                  ArrayRef<AsyncValue*> values,
                  llvm::unique_function<void()> callee) {
  exec_ctx.host()->RunWhenReady(values, std::move(callee));
}

void RunWhenReady(const ExecutionContext& exec_ctx,
                  ArrayRef<RCReference<AsyncValue>> values,
                  llvm::unique_function<void()> callee) {
  exec_ctx.host()->RunWhenReady(values, std::move(callee));
}

}  // namespace tfrt
