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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "tfrt/jitrt/opdefs/rt_ops.h"
#include "tfrt/jitrt/transforms/rt_passes.h"

namespace tfrt {
namespace jitrt {
namespace {

using llvm::StringRef;

using mlir::Block;
using mlir::dyn_cast;
using mlir::FunctionType;
using mlir::ImplicitLocOpBuilder;
using mlir::isa_and_nonnull;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::StringAttr;
using mlir::SymbolTable;
using mlir::Type;
using mlir::TypeRange;
using mlir::Value;

using mlir::cf::AssertOp;
using mlir::cf::CondBranchOp;

using mlir::func::CallOp;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;

#define GEN_PASS_CLASSES
#include "tfrt/jitrt/transforms/rt_gen_passes.h.inc"

class ConvertToKernelFunctionPass
    : public ConvertToKernelFunctionBase<ConvertToKernelFunctionPass> {
  void runOnOperation() override;
};

}  // namespace

static void ConvertCustomCallOperations(FuncOp func, Value kernel_ctx) {
  MLIRContext* ctx = func->getContext();

  SymbolTable sym_table(func->getParentOfType<ModuleOp>());

  struct CustomCall {
    CallOp call;
    FuncOp callee;
    StringRef target;
    bool direct;
  };

  // Collect function calls that have to be converted to custom calls.
  llvm::SmallVector<CustomCall> custom_calls;
  func.walk([&](CallOp op) {
    if (auto callee = dyn_cast<FuncOp>(sym_table.lookup(op.getCallee()))) {
      // Check if the call is an indirect custom call ...
      StringAttr target = callee->getAttrOfType<StringAttr>("rt.custom_call");
      if (target) custom_calls.push_back({op, callee, target.strref(), false});

      // ... or a direct custom call.
      target = callee->getAttrOfType<StringAttr>("rt.direct_custom_call");
      if (target) custom_calls.push_back({op, callee, target.strref(), true});
    }
  });

  // After converting to custom call we need to clean up all declarations.
  llvm::DenseSet<FuncOp> erase_declarations;

  // Rewrite function calls to `rt.custom_call` operations.
  for (CustomCall custom_call : custom_calls) {
    ImplicitLocOpBuilder b(custom_call.call.getLoc(), custom_call.call);

    // Custom call intrinsic always returns the status flag.
    llvm::SmallVector<Type> results = {StatusType::get(ctx)};
    results.append(custom_call.call->getResultTypes().begin(),
                   custom_call.call->getResultTypes().end());

    // Rewrite function call with a custom call, and check the return status.
    auto call = b.create<CustomCallOp>(results, kernel_ctx, custom_call.target,
                                       custom_call.direct,
                                       custom_call.call.getOperands());

    // Copy optional attributes from the custom call function declaration.
    llvm::ArrayRef<llvm::StringRef> callee_attrs =
        custom_call.callee.getAttributeNames();
    for (auto& attr : custom_call.callee->getAttrs()) {
      if (isa_and_nonnull<RuntimeDialect>(attr.getNameDialect())) continue;
      if (llvm::find(callee_attrs, attr.getName()) == callee_attrs.end())
        call->setAttr(attr.getName(), attr.getValue());
    }

    // Copy optional attributes from the call operation to the custom call.
    llvm::ArrayRef<llvm::StringRef> orig_attrs =
        custom_call.call.getAttributeNames();
    for (auto& attr : custom_call.call->getAttrs()) {
      if (llvm::find(orig_attrs, attr.getName()) == orig_attrs.end())
        call->setAttr(attr.getName(), attr.getValue());
    }

    b.create<AssertOp>(
        b.create<IsOkOp>(TypeRange(b.getI1Type()), call.status()),
        b.getStringAttr("custom call '" + custom_call.target + "' failed"));

    // Forward users of the original results to custom call results.
    auto rets = llvm::zip(custom_call.call.getResults(),
                          llvm::drop_begin(call.getResults()));
    llvm::for_each(rets, [](auto ret) {
      std::get<0>(ret).replaceAllUsesWith(std::get<1>(ret));
    });

    // Keep track of custom call declaration to erase.
    erase_declarations.insert(custom_call.callee);

    // Erase the original function call operation.
    custom_call.call.erase();
  }

  // Erase all converted custom calls declarations.
  for (auto func : erase_declarations) sym_table.erase(func);
}

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

static Value PrependKernelContextArgument(mlir::func::FuncOp func) {
  mlir::Type new_type = KernelContextType::get(func.getContext());
  mlir::DictionaryAttr attr = mlir::DictionaryAttr::get(func.getContext());
  func.insertArguments({0}, {new_type}, {attr}, {func.getLoc()});
  return func.getArgument(0);
}

static void ConvertToKernelFunction(FuncOp func) {
  assert(func->hasAttr(kJitRtEntrypointAttrName));

  Value kernel_ctx = PrependKernelContextArgument(func);
  ConvertCustomCallOperations(func, kernel_ctx);
  ConvertReturnOperations(func, kernel_ctx);
  ConvertAssertOperations(func, kernel_ctx);

  // After conversion !rt.kernel_context is a marker of an entrypoint function.
  func->removeAttr(kJitRtEntrypointAttrName);
}

void ConvertToKernelFunctionPass::runOnOperation() {
  llvm::SmallVector<FuncOp> entry_points;

  // Collect JitRt entrypoint functions.
  getOperation().walk([&](FuncOp op) {
    if (op->hasAttr(kJitRtEntrypointAttrName)) entry_points.push_back(op);
  });

  llvm::for_each(entry_points, ConvertToKernelFunction);
}

std::unique_ptr<mlir::OperationPass<ModuleOp>> CreateConvertToKernelFunction() {
  return std::make_unique<ConvertToKernelFunctionPass>();
}

}  // namespace jitrt
}  // namespace tfrt
