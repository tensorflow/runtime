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

#include <iterator>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"
#include "tfrt/gpu/kernels/gpu_ops.h"
#include "tfrt/gpu/passes/passes.h"

namespace tfrt {
namespace gpu {

Value internal::GpuAsyncOpConversionGetStream(Operation *parent) {
  if (auto exec_op = dyn_cast_or_null<conversion::AsyncExecuteOp>(parent))
    return exec_op.getRegion().getArgument(1);
  return Value();
}
Value internal::GpuAsyncOpConversionGetChain(Operation *parent) {
  if (auto exec_op = dyn_cast_or_null<conversion::AsyncExecuteOp>(parent))
    return exec_op.getRegion().back().getTerminator()->getOperand(0);
  return Value();
}
void internal::GpuAsyncOpConversionSetChain(Value chain,
                                            PatternRewriter &rewriter) {
  Operation *terminator = chain.getParentRegion()->back().getTerminator();
  rewriter.updateRootInPlace(
      terminator, [&] { terminator->setOperands(ValueRange(chain)); });
}

namespace {

// Wraps consecutive legal ops within a block into a
// tfrt_gpu_conversion.async.execute op.
struct NestLegalOpsInConversionAsyncExecPattern
    : public OpRewritePattern<FuncOp> {
  NestLegalOpsInConversionAsyncExecPattern(MLIRContext *context,
                                           ConversionTarget &target)
      : OpRewritePattern(context), target(target) {}

 private:
  LogicalResult matchAndRewrite(FuncOp func_op,
                                PatternRewriter &rewriter) const override;
  LogicalResult matchAndRewriteBlock(Block *block,
                                     PatternRewriter &rewriter) const;
  ConversionTarget &target;
};

// Folds a memref.view of !tfrt_gpu.buffer with zero byte_shift.
struct FoldMemrefViewPattern : public OpConversionPattern<memref::ViewOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      memref::ViewOp view_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// Folds a memref.reinterpret_cast of !tfrt_gpu.buffer with zero static offsets.
struct FoldMemrefReinterpretCastPattern
    : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      memref::ReinterpretCastOp cast_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

}  // namespace

LogicalResult NestLegalOpsInConversionAsyncExecPattern::matchAndRewrite(
    FuncOp func_op, PatternRewriter &rewriter) const {
  rewriter.startRootUpdate(func_op);
  LogicalResult result = failure();
  func_op.walk([&](Block *block) {
    if (isa<conversion::AsyncExecuteOp>(block->getParentOp()))
      return WalkResult::skip();
    if (succeeded(matchAndRewriteBlock(block, rewriter)))
      result = success();  //
    return WalkResult::advance();
  });
  succeeded(result) ? rewriter.finalizeRootUpdate(func_op)
                    : rewriter.cancelRootUpdate(func_op);
  return result;
}

// Iterate over ops in block, and whenever we transition from a legal to an
// illegal op, wrap preceding legal ops in !tfrt_gpu_conversion.async.execute.
LogicalResult NestLegalOpsInConversionAsyncExecPattern::matchAndRewriteBlock(
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
    // Move sequence of legal ops into !tfrt_gpu_conversion.async.execute
    // body.
    body->getOperations().splice(body->begin(), op->getBlock()->getOperations(),
                                 legal_begin->getIterator(), op->getIterator());
    legal_begin = nullptr;  // Start of illegal op sequence.
    result = success();
  }
  return result;
}

LogicalResult FoldMemrefViewPattern::matchAndRewrite(
    memref::ViewOp view_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!adaptor.source().getType().isa<BufferType>())
    return rewriter.notifyMatchFailure(view_op, "expected BufferType source");
  if (!adaptor.sizes().empty())
    return rewriter.notifyMatchFailure(view_op, "expected no sizes");
  auto const_offset =
      adaptor.byte_shift().getDefiningOp<arith::ConstantIndexOp>();
  auto size_bits = view_op.getType().getSizeInBits();
  auto source_type = view_op.source().getType().cast<MemRefType>();
  if (const_offset && const_offset.value() == 0 &&
      source_type.getSizeInBits() == size_bits) {
    rewriter.replaceOp(view_op, {adaptor.source()});
    return success();
  }
  auto loc = view_op->getLoc();
  auto offset = rewriter.create<UnrealizedConversionCastOp>(
      loc, rewriter.getIntegerType(64, false), adaptor.byte_shift());
  auto size_bytes = (size_bits + 7) / 8;
  auto size = rewriter.create<compiler::ConstantUI64Op>(loc, size_bytes);
  rewriter.replaceOpWithNewOp<MemViewOp>(view_op, adaptor.source(),
                                         offset.getResult(0), size);
  return success();
}

LogicalResult FoldMemrefReinterpretCastPattern::matchAndRewrite(
    memref::ReinterpretCastOp cast_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!adaptor.source().getType().isa<BufferType>())
    return rewriter.notifyMatchFailure(cast_op, "expected BufferType source");
  if (!adaptor.offsets().empty() ||
      llvm::any_of(adaptor.static_offsets(), [](Attribute attribute) {
        return attribute.cast<IntegerAttr>().getInt() != 0;
      }))
    return rewriter.notifyMatchFailure(cast_op, "expected static zero offsets");
  rewriter.replaceOp(cast_op, {adaptor.source()});
  return success();
}

void populateGpuAsyncConversionPatterns(RewritePatternSet &patterns,
                                        TypeConverter &converter,
                                        ConversionTarget &target) {
  patterns.add<NestLegalOpsInConversionAsyncExecPattern>(patterns.getContext(),
                                                         target);
  patterns.add<FoldMemrefViewPattern, FoldMemrefReinterpretCastPattern>(
      converter, patterns.getContext());
}

}  // namespace gpu
}  // namespace tfrt
