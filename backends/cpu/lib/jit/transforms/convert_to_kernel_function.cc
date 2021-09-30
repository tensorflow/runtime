/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "tfrt/cpu/jit/opdefs/rt_ops.h"
#include "tfrt/cpu/jit/transforms/rt_passes.h"

namespace tfrt {
namespace cpu {
namespace jit {
namespace {

using mlir::FuncOp;
using mlir::FunctionType;
using mlir::ImplicitLocOpBuilder;
using mlir::ModuleOp;
using mlir::ReturnOp;
using mlir::Value;

#define GEN_PASS_CLASSES
#include "tfrt/cpu/jit/transforms/rt_gen_passes.h.inc"

class ConvertToKernelFunctionPass
    : public ConvertToKernelFunctionBase<ConvertToKernelFunctionPass> {
  void runOnOperation() override;
};

}  // namespace

static void ConvertToKernelFunction(FuncOp func) {
  // We only convert functions with kernel context as the first argument.
  Value kernel_ctx = func.getNumArguments() ? func.getArgument(0) : Value();
  if (!kernel_ctx || !kernel_ctx.getType().isa<KernelContextType>()) return;

  // Rewrite all returns to the Runtime API calls.
  func.walk([&](ReturnOp ret) {
    ImplicitLocOpBuilder builder(ret.getLoc(), ret);

    // Return all outputs via the `rt.set_output` operation.
    for (auto pair : llvm::enumerate(ret.operands())) {
      builder.create<SetOutput>(kernel_ctx, pair.index(), pair.value());
    }

    // Replace original return with an empty one.
    builder.create<ReturnOp>();
    ret.erase();
  });

  // Update function type to the function with empty results.
  auto type = FunctionType::get(func.getContext(), func.getArgumentTypes(), {});
  func.setType(type);
}

void ConvertToKernelFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  module.walk(ConvertToKernelFunction);
}

std::unique_ptr<mlir::OperationPass<ModuleOp>> CreateConvertToKernelFunction() {
  return std::make_unique<ConvertToKernelFunctionPass>();
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
