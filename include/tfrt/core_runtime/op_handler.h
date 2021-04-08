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

// This file declares OpHandler, responsible for eagerly executing ops.

#ifndef TFRT_CORE_RUNTIME_OP_HANDLER_H_
#define TFRT_CORE_RUNTIME_OP_HANDLER_H_

#include "llvm/Support/Error.h"
#include "tfrt/core_runtime/core_runtime_op.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class OpInvocation;
class CoreRuntime;

template <typename T>
class AsyncValueRef;
class DenseHostTensor;
class ExecutionContext;
class HostTensor;
class Tensor;
class TensorHandle;
class Location;
class OpAttrsRef;
class Chain;

class OpHandler {
 public:
  explicit OpHandler(string_view name, CoreRuntime *runtime,
                     OpHandler *fallback);

  // Return the CoreRuntime object that this op_handler is associated with.
  CoreRuntime *GetRuntime() const { return runtime_; }

  string_view GetName() const { return name_; }

  OpHandler *GetFallback() const { return fallback_; }

  virtual Expected<CoreRuntimeOp> MakeOp(string_view op_name) = 0;

  virtual ~OpHandler();

 private:
  const std::string name_;
  CoreRuntime *const runtime_;
  OpHandler *const fallback_;
};

//===----------------------------------------------------------------------===//
// Inline implementation details of OpHandler
//===----------------------------------------------------------------------===//

inline OpHandler::OpHandler(string_view name, CoreRuntime *runtime,
                            OpHandler *fallback)
    : name_(name), runtime_(runtime), fallback_(fallback) {}

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_OP_HANDLER_H_
