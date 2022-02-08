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
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "tfrt/jitrt/opdefs/rt_ops.h"
#include "tfrt/jitrt/transforms/rt_passes.h"

namespace tfrt {
namespace jitrt {
namespace {

using mlir::Block;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::ImplicitLocOpBuilder;
using mlir::ModuleOp;
using mlir::ReturnOp;
using mlir::Value;
using mlir::cf::AssertOp;
using mlir::cf::CondBranchOp;

#define GEN_PASS_CLASSES
#include "tfrt/jitrt/transforms/rt_gen_passes.h.inc"

class ConvertToKernelFunctionPass
    : public ConvertToKernelFunctionBase<ConvertToKernelFunctionPass> {
  void runOnOperation() override;
};

}  // namespace

static void ConvertReturnOperations(FuncOp func, Value kernel_ctx) {
  // Convert all returns to the Runtime API calls.
  func.walk([&](ReturnOp ret) {
    ImplicitLocOpBuilder b(ret.getLoc(), ret);

    // Return all outputs via the `rt.set_output` operation.
    for (auto& pair : llvm::enumerate(ret.getOperands())) {
      b.create<SetOutputOp>(kernel_ctx, pair.index(), pair.value());
    }

    // Replace original return with an empty one.
    b.create<ReturnOp>();
    ret.erase();
  });

  // Update function type to the function with empty results.
  auto type = FunctionType::get(func.getContext(), func.getArgumentTypes(), {});
  func.setType(type);
}

static void ConvertAssertOperations(FuncOp func, Value kernel_ctx) {
  // Collect all assert operations in the function body.
  llvm::SmallVector<AssertOp> asserts;
  func.walk([&](AssertOp op) { asserts.push_back(op); });

  // Rewrite all asserts to the Runtime API calls.
  for (AssertOp assert : asserts) {
    ImplicitLocOpBuilder b(assert.getLoc(), assert);

    // Split the block at the assert operation.
    Block* block = assert->getBlock();
    Block* ok = block->splitBlock(assert);

    // Set up block for returning error.
    Block* err = func.addBlock();
    b.setInsertionPointToStart(err);
    b.create<SetErrorOp>(kernel_ctx, assert.getMsg());
    b.create<ReturnOp>();

    // Branch into the error block if assertion failed.
    b.setInsertionPointToEnd(block);
    b.create<CondBranchOp>(assert.getArg(), ok, err);

    // Erase the original assert operation.
    assert.erase();
  }
}

static Value PrependKernelContextArgument(mlir::FuncOp func) {
  mlir::Type new_type = KernelContextType::get(func.getContext());
  mlir::DictionaryAttr attr = mlir::DictionaryAttr::get(func.getContext());
  func.insertArguments({0}, {new_type}, {attr}, {func.getLoc()});
  return func.getArgument(0);
}

static void ConvertToKernelFunction(FuncOp func) {
  // Skip functions that are not JitRt entrypoints.
  if (!func->hasAttr(kJitRtEntrypointAttrName)) return;

  Value kernel_ctx = PrependKernelContextArgument(func);
  ConvertReturnOperations(func, kernel_ctx);
  ConvertAssertOperations(func, kernel_ctx);

  // After conversion !rt.kernel_context is a marker of an entrypoint function.
  func->removeAttr(kJitRtEntrypointAttrName);
}

void ConvertToKernelFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  module.walk(ConvertToKernelFunction);
}

std::unique_ptr<mlir::OperationPass<ModuleOp>> CreateConvertToKernelFunction() {
  return std::make_unique<ConvertToKernelFunctionPass>();
}

}  // namespace jitrt
}  // namespace tfrt
