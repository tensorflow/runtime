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

#include "tfrt/gpu/pass/pass.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"
#include "tfrt/gpu/kernels/gpu_ops.h"

namespace tfrt {
namespace gpu {

Value internal::GpuAsyncOpConversionGetStream(Operation *parent) {
  if (auto exec_op = dyn_cast_or_null<conversion::AsyncExecuteOp>(parent))
    return exec_op.body().getArgument(1);
  return Value();
}
Value internal::GpuAsyncOpConversionGetChain(Operation *parent) {
  if (auto exec_op = dyn_cast_or_null<conversion::AsyncExecuteOp>(parent))
    return exec_op.body().back().getTerminator()->getOperand(0);
  return Value();
}
void internal::GpuAsyncOpConversionSetChain(Value chain,
                                            PatternRewriter &rewriter) {
  Operation *terminator = chain.getParentRegion()->back().getTerminator();
  rewriter.updateRootInPlace(
      terminator, [&] { terminator->setOperands(ValueRange(chain)); });
}

namespace {
struct WrapInAsyncExecPattern : public OpRewritePattern<FuncOp> {
  WrapInAsyncExecPattern(MLIRContext *context, ConversionTarget &target);

 private:
  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override;
  LogicalResult matchAndRewriteBlock(Block *block,
                                     PatternRewriter &rewriter) const;
  ConversionTarget &target;
};
}  // namespace

WrapInAsyncExecPattern::WrapInAsyncExecPattern(MLIRContext *context,
                                               ConversionTarget &target)
    : OpRewritePattern(context), target(target) {}

LogicalResult WrapInAsyncExecPattern::matchAndRewrite(
    FuncOp op, PatternRewriter &rewriter) const {
  rewriter.startRootUpdate(op);
  LogicalResult result = failure();
  op.walk([&](Block *block) {
    if (dyn_cast<conversion::AsyncExecuteOp>(block->getParentOp()))
      return WalkResult::skip();
    if (succeeded(matchAndRewriteBlock(block, rewriter)))
      result = success();  //
    return WalkResult::advance();
  });
  succeeded(result) ? rewriter.finalizeRootUpdate(op)
                    : rewriter.cancelRootUpdate(op);
  return result;
}

// Iterate over ops in block, and whenever we transition from a legal to an
// illegal op, wrap preceding legal ops in !tfrt_gpu.conversion.async.execute.
LogicalResult WrapInAsyncExecPattern::matchAndRewriteBlock(
    Block *block, PatternRewriter &rewriter) const {
  LogicalResult result = failure();
  Operation *legal_begin = nullptr;
  for (Operation *op : llvm::make_pointer_range(block->getOperations())) {
    if (target.isLegal(op)) {
      if (!legal_begin)  // Start of legal op sequence.
        legal_begin = op;
      continue;
    }
    if (!legal_begin)  // Continue in illegal op sequence.
      continue;

    rewriter.setInsertionPoint(legal_begin);
    auto loc = legal_begin->getLoc();
    auto *body = rewriter.create<conversion::AsyncExecuteOp>(loc).getBody();
    // Move sequence of legal ops into !tfrt_gpu.conversion.async.execute body.
    body->getOperations().splice(body->begin(), op->getBlock()->getOperations(),
                                 legal_begin->getIterator(), op->getIterator());
    legal_begin = nullptr;  // Start of illegal op sequence.
    result = success();
  }
  return result;
}

void populateGpuAsyncConversionPatterns(RewritePatternSet &patterns,
                                        mlir::ConversionTarget &target) {
  patterns.add<WrapInAsyncExecPattern>(patterns.getContext(), target);
}

}  // namespace gpu
}  // namespace tfrt
