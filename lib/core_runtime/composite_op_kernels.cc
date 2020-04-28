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

//===- composite_op_kernels.cc -----------------------------------------===//
//
// This library contains kernels that allows the bef_executor to run composite
// ops.
//
//===----------------------------------------------------------------------===//

#include "composite_op_handler.h"
#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/kernels.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {

// TODO(fishx): Let it take CompositeOpHandler* as input.
static Chain RegisterCompositeOp(Argument<OpHandler *> op_handler,
                                 StringAttribute name,
                                 Attribute<Function> fn_const,
                                 const ExecutionContext &exec_ctx) {
  auto *fn_op_handler = static_cast<CompositeOpHandler *>(op_handler.get());
  RCReference<Function> fn = FormRef(const_cast<Function *>(&(*fn_const)));

  fn_op_handler->RegisterCompositeOp(name, std::move(fn));

  return {};
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void RegisterCompositeOpHandlerKernels(KernelRegistry *registry) {
  registry->AddKernel("corert.register_composite_op",
                      TFRT_KERNEL(RegisterCompositeOp));
}

}  // namespace tfrt
