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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"
#include "tfrt/basic_kernels/opdefs/types.h"
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
// Wraps consecutive legal ops within a block into a
// tfrt_gpu_conversion.async.execute op.
struct WrapInAsyncExecPattern : public OpRewritePattern<FuncOp> {
  WrapInAsyncExecPattern(MLIRContext *context, ConversionTarget &target);

 private:
  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override;
  LogicalResult matchAndRewriteBlock(Block *block,
                                     PatternRewriter &rewriter) const;
  ConversionTarget &target;
};

// Folds a memref.view of !tfrt_gpu.buffer with zero byte_shift.
struct FoldMemrefViewPattern
    : public OpConversionPattern<mlir::memref::ViewOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      mlir::memref::ViewOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

// Moves the body of a tfrt_gpu_conversion.async.execute op into the parent
// block and removes the op.
//
//     %t0 = tfrt_gpu.cast %ch0, %stream : !gpu.async.token
//     %t1 = tfrt_gpu_conversion.async.execute [%t0] {
//       ^bb(%0 : !tfrt.chain, %1 : !tfrt_gpu.stream)
//       ... ops using %0 and %1 ...
//       tfrt.return %n : !tfrt.chain
//     }
//
// will be rewritten to
//
//     %t0 = tfrt_gpu.cast %ch0, %stream : !gpu.async.token
//     ... ops using %ch0 and %stream ...
//     %t1 = tfrt_gpu.cast %n, %stream : !gpu.async.token
//
struct UnwrapAsyncExecPattern
    : public OpConversionPattern<conversion::AsyncExecuteOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      conversion::AsyncExecuteOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

// Rewrites a function with two gpu.wait ops to take extra !tfrt.chain and
// !tfrt_gpu.stream arguments and return a !tfrt.chain.
//
//     func @main(...) {
//       %0 = gpu.wait async
//       ...
//       gpu.wait [%n]
//       return
//     }
//
// will be rewritten to
//
//     func @main(%chain : !tfrt:chain, %stream : !tfrt_gpu.stream, ...) ->
//     !tfrt.chain {
//       %0 = tfrt_gpu_conversion.cast %chain, %stream : !gpu.async.token
//       ...
//       %result = tfrt_gpu_conversion.cast %n : !tfrt.chain
//       tfrt.return %result
//     }
//
struct HoistGpuWaitsPattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern::OpRewritePattern;

 private:
  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override;
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
// illegal op, wrap preceding legal ops in !tfrt_gpu_conversion.async.execute.
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
    // Move sequence of legal ops into !tfrt_gpu_conversion.async.execute body.
    body->getOperations().splice(body->begin(), op->getBlock()->getOperations(),
                                 legal_begin->getIterator(), op->getIterator());
    legal_begin = nullptr;  // Start of illegal op sequence.
    result = success();
  }
  return result;
}

LogicalResult FoldMemrefViewPattern::matchAndRewrite(
    mlir::memref::ViewOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  mlir::memref::ViewOpAdaptor adaptor(operands);
  if (!adaptor.source().getType().isa<BufferType>())
    return rewriter.notifyMatchFailure(op, "expected gpu::BufferType source");
  auto byte_shift = adaptor.byte_shift().getDefiningOp<mlir::ConstantIndexOp>();
  if (!byte_shift || byte_shift.getValue() != 0)
    return rewriter.notifyMatchFailure(op, "expected const zero byte_shift");
  if (!adaptor.sizes().empty())
    return rewriter.notifyMatchFailure(op, "expected no sizes");
  rewriter.replaceOp(op, {adaptor.source()});
  return success();
}

LogicalResult UnwrapAsyncExecPattern::matchAndRewrite(
    conversion::AsyncExecuteOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!op->getNumResults())
    return rewriter.notifyMatchFailure(op, "has no result");
  auto cast_op = [&]() -> Operation * {
    if (operands.empty()) return nullptr;
    return operands.front().getDefiningOp<conversion::CastOp>();
  }();
  if (!cast_op || cast_op->getNumOperands() != 2 ||
      !cast_op->getOperand(0).getType().isa<ChainType>() ||
      !cast_op->getOperand(1).getType().isa<StreamType>())
    return rewriter.notifyMatchFailure(op, "no !tfrt_gpu_conversion.cast user");

  // Merge !tfrt_gpu_conversion.async.execute body into parent block.
  Operation *terminator = op.getBody()->getTerminator();
  rewriter.mergeBlockBefore(op.getBody(), op, cast_op->getOperands());
  rewriter.replaceOpWithNewOp<conversion::CastOp>(
      op, rewriter.getType<mlir::gpu::AsyncTokenType>(),
      ValueRange{terminator->getOperand(0), cast_op->getOperand(1)});
  rewriter.eraseOp(terminator);
  return success();
}

LogicalResult HoistGpuWaitsPattern::matchAndRewrite(
    FuncOp op, PatternRewriter &rewriter) const {
  auto range = op.body().getOps<mlir::gpu::WaitOp>();
  SmallVector<Operation *, 2> wait_ops(range.begin(), range.end());

  // Require 2 gpu.async.wait ops:
  // wait_ops[0] needs to be of the form `%token = gpu.wait async`.
  // wait_ops[1] needs to be of the form `gpu.wait [%token]`.
  if (wait_ops.size() != 2)
    return rewriter.notifyMatchFailure(op, "expected 2 !gpu.async.wait ops");
  for (int i : {0, 1}) {
    if (wait_ops[i]->getNumResults() == i || wait_ops[i]->getNumOperands() != i)
      return rewriter.notifyMatchFailure(op, "unexpected !gpu.async.wait form");
  }

  // Add !tfrt.chain, !tfrt_gpu.stream arguments and !tfrt.chain result.
  auto chain_type = rewriter.getType<ChainType>();
  SmallVector<Type, 8> input_types;
  input_types.reserve(op.getNumArguments() + 2);
  input_types.push_back(chain_type);
  input_types.push_back(rewriter.getType<StreamType>());
  copy(op.getArgumentTypes(), std::back_inserter(input_types));
  rewriter.updateRootInPlace(op, [&] {
    op.setType(rewriter.getType<mlir::FunctionType>(input_types,
                                                    TypeRange(chain_type)));
  });

  // Add new function arguments to entry block. This is a bit of a dance
  // so that it could be rolled back in case of conversion failure.
  Block *block = &op.body().front();
  Block *entry = rewriter.createBlock(block, input_types);
  auto block_args = entry->getArguments();
  rewriter.mergeBlocks(block, entry, block_args.drop_front(2));

  // Replace wait_ops[0] with cast of new block arguments to !gpu.async.token.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&op.body().front());
  rewriter.replaceOpWithNewOp<conversion::CastOp>(
      wait_ops[0], rewriter.getType<mlir::gpu::AsyncTokenType>(),
      block_args.take_front(2));

  // Replace wait_ops[1] with cast of its token operand to !tfrt.chain.
  Operation *terminator = op.body().back().getTerminator();
  rewriter.setInsertionPointAfter(terminator);
  auto cast = rewriter.create<conversion::CastOp>(
      wait_ops[1]->getLoc(), chain_type, wait_ops[1]->getOperands());
  rewriter.eraseOp(wait_ops[1]);

  // Return casted !tfrt.chain.
  rewriter.replaceOpWithNewOp<tfrt::ReturnOp>(terminator, cast->getResults());

  return success();
}

void populateGpuAsyncConversionPatterns(RewritePatternSet &patterns,
                                        mlir::TypeConverter &converter,
                                        mlir::ConversionTarget &target) {
  patterns.add<WrapInAsyncExecPattern>(patterns.getContext(), target);
  patterns.add<FoldMemrefViewPattern>(converter, patterns.getContext());
}

void populateTfrtConversionPatterns(mlir::RewritePatternSet &patterns,
                                    mlir::ConversionTarget &target) {
  patterns.add<UnwrapAsyncExecPattern, HoistGpuWaitsPattern>(
      patterns.getContext());

  // Cast ops are unused after conversion, but DCE needs to be run separately.
  target.addLegalOp<conversion::CastOp>();

  // Signature needs to be `(!tfrt.chain, !tfrt.stream, ...) -> (!tfrt.chain)`.
  target.addDynamicallyLegalOp<FuncOp>([](FuncOp op) {
    auto type = op.getType();
    return type.getNumResults() == 1 && type.getResult(0).isa<ChainType>() &&
           type.getNumInputs() >= 2 && type.getInput(0).isa<ChainType>() &&
           type.getInput(1).isa<StreamType>();
  });
}

}  // namespace gpu
}  // namespace tfrt
