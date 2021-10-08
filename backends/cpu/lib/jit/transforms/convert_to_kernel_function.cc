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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "tfrt/cpu/jit/opdefs/rt_ops.h"
#include "tfrt/cpu/jit/transforms/rt_passes.h"

namespace tfrt {
namespace cpu {
namespace jit {
namespace {

using mlir::AssertOp;
using mlir::Block;
using mlir::CondBranchOp;
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
 public:
  explicit ConvertToKernelFunctionPass(bool convert_assert)
      : convert_assert_(convert_assert) {}
  void runOnOperation() override;

 private:
  bool convert_assert_;
};

}  // namespace

static void ConvertReturnOperations(FuncOp func, Value kernel_ctx) {
  // Convert all returns to the Runtime API calls.
  func.walk([&](ReturnOp ret) {
    ImplicitLocOpBuilder b(ret.getLoc(), ret);

    // Return all outputs via the `rt.set_output` operation.
    for (auto pair : llvm::enumerate(ret.operands())) {
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
    b.create<SetErrorOp>(kernel_ctx, assert.msg());
    b.create<ReturnOp>();

    // Branch into the error block if assertion failed.
    b.setInsertionPointToEnd(block);
    b.create<CondBranchOp>(assert.arg(), ok, err);

    // Erase the original assert operation.
    assert.erase();
  }
}

static void ConvertToKernelFunction(FuncOp func, bool convert_assert) {
  // We only convert functions with kernel context as the first argument.
  bool is_candidate = !func.isDeclaration() && func.getNumArguments();
  Value kernel_ctx = is_candidate ? func.getArgument(0) : Value();
  if (!kernel_ctx || !kernel_ctx.getType().isa<KernelContextType>()) return;

  ConvertReturnOperations(func, kernel_ctx);
  if (convert_assert) ConvertAssertOperations(func, kernel_ctx);
}

void ConvertToKernelFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  module.walk(
      [&](FuncOp func) { ConvertToKernelFunction(func, convert_assert_); });
}

std::unique_ptr<mlir::OperationPass<ModuleOp>> CreateConvertToKernelFunction(
    bool convert_assert) {
  return std::make_unique<ConvertToKernelFunctionPass>(convert_assert);
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
