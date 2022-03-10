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
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/gpu/kernels/gpu_ops.h"
#include "tfrt/gpu/passes/passes.h"
#include "tfrt/tensor/opdefs/dense_host_tensor.h"
#include "tfrt/test_kernels/opdefs/test_kernels.h"

namespace tfrt {
namespace gpu {

using CastOp = UnrealizedConversionCastOp;

namespace {

// Helper for 1-N conversion, similar to materializeSource/TargetConversion().
// Creates CastOps from illegal source types to legal target types and back.
class OneToAnyConversion {
 public:
  static FailureOr<OneToAnyConversion> Get(TypeConverter *converter,
                                           TypeRange source_types);

  ArrayRef<Type> GetTargetTypes() const {
    return conversion_.getConvertedTypes();
  }

  // Inserts casts of legal-typed target_values back to source_types.
  SmallVector<Value, 4> CastToSourceTypes(OpBuilder &builder, Location loc,
                                          ValueRange target_values);

  // Inserts casts of illegal-typed source_values to converted types.
  SmallVector<Value, 4> CastToTargetTypes(OpBuilder &builder, Location loc,
                                          ValueRange source_values);

 private:
  OneToAnyConversion(TypeRange source_types,
                     TypeConverter::SignatureConversion conversion)
      : source_types_(source_types), conversion_(conversion) {}

  TypeRange source_types_;
  TypeConverter::SignatureConversion conversion_;
};

// Rewrites a function to take extra !tfrt.chain and !tfrt_gpu.stream arguments
// and return a !tfrt.chain. Adds gpu.wait dependencies where there aren't any.
//
//     func @main(...) {
//       ...
//       %ti = gpu.wait async [/*no deps*/]  // At least one, may be nested.
//       ...
//       gpu.wait /*not async*/ [...]        // Exactly one.
//       return
//     }
//
// will be rewritten to
//
//     func @main(
//       %arg0 : !tfrt.chain, %arg1 : !tfrt_gpu.stream, ...
//     ) -> !tfrt.chain {
//       %t0 = unrealized_conversion_cast %arg0, %arg1
//               : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
//       %t1 = gpu.wait async [%t0]
//       ...
//       %ti = gpu.wait async [%t1]
//       ...
//       %tn = gpu.wait async [...]
//       %ch, %stream = unrealized_conversion_cast %tn
//               : !gpu.async.token to !tfrt.chain, !tfrt_gpu.stream
//       return %ch
//     }
//
struct AddChainAndStreamToFuncPattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern::OpRewritePattern;

 private:
  LogicalResult matchAndRewrite(FuncOp func_op,
                                PatternRewriter &rewriter) const override;
};

// Two type conversion patterns for async.execute. It inserts casts to/from the
// converted types before/after as well as at the end/beginning of the region.
//
// With type X being converted to Y, applying
// ConvertAsyncExec/YieldToChainAndEventPattern to
//
//     %a1, %f1 = async.execute [%a0] (
//       %f0 as %x0: !async.value<X>
//     ) -> !async.value<X> {
//       ...
//       async.yield %x1 : X
//     }
//
// will be rewritten to
//
//     %g0 = unrealized_conversion_cast %f0 : !async.value<X> to !async.value<Y>
//     %a1, %g1 = async.execute [%a0] (
//       %g0 as %y0: !async.value<Y>
//     ) -> (!async.value<Y>) {
//       %x0 = unrealized_conversion_cast %y0 : Y to X
//       ...
//       %y1 = unrealized_conversion_cast %x1 : X to Y
//       async.yield %y1 : Y
//     }
//     %f1 = unrealized_conversion_cast %g1 : !async.value<Y> to !async.value<X>
//
struct ConvertAsyncExecToChainAndEventPattern
    : public OpConversionPattern<async::ExecuteOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      async::ExecuteOp exec_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// A type conversion pattern for async.yield.
// See documentation of above pattern.
struct ConvertAsyncYieldToChainAndEventPattern
    : public OpConversionPattern<async::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      async::YieldOp yield_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// Swaps the `async.await` and the operand-defining cast.
//
//     %fx = unrealized_conversion_cast %fy : !async.value<Y> to !async.value<X>
//     %x  = async.await %fx : X
//
// will be rewritten to
//
//     %y  = async.await %fy : Y
//     %x  = unrealized_conversion_cast %y : Y to X
//
struct SwapAsyncAwaitOfCastPattern
    : public OpConversionPattern<async::AwaitOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      async::AwaitOp await_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// Converts `gpu.memset` op to `tfrt_gpu.mem.set` op.
struct ConvertMemsetPattern : OpConversionPattern<mlir::gpu::MemsetOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      mlir::gpu::MemsetOp memset_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// Converts `gpu.memcpy` op to `tfrt_gpu.mem.copy` op.
struct ConvertMemcpyPattern : OpConversionPattern<mlir::gpu::MemcpyOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      mlir::gpu::MemcpyOp memcpy_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// Converts `memref.get_global` to a `tfrt.once` call of the corresponding
// function.
//
//     %buffer = memref.get_global @global : memref<4xf32>
//
// will be rewritten to
//
//     %result:2 = tfrt.once @global %context
//         : (!tfrt_gpu.context) -> (!tfrt_gpu.buffer, !tfrt.chain)
//     %buffer = unrealized_conversion_cast %result#0, %result#1
//         !tfrt_gpu.buffer, !tfrt.chain to !tfrt_gpu.buffer
//
// The cast is removed later, see ConvertBufferCastPattern.
struct ConvertGetGlobalPattern : OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      memref::GetGlobalOp get_global_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// Converts `gpu.module` op to a function that loads the module.
//
//     gpu.module @module attributes { binary = "<cubin>" }
//
// will be rewritten to
//
//     func @module(%arg0: !tfrt_gpu.context) -> !tfrt_gpu.module {
//       %0 = tfrt_gpu.module.load %arg0 {data = "<cubin>\00"}
//       tfrt.return %0 : !tfrt_gpu.module
//     }
//
// If the `gpu.module` also has a `constants` attribute, the generated function
// initializes the given globals with the provided values and returns a chain.
struct ConvertGpuModulePattern : OpConversionPattern<mlir::gpu::GPUModuleOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      mlir::gpu::GPUModuleOp module_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// Converts a `memref.global` to a function that returns the corresponding
// `!tfrt_gpu.buffer` and a `tfrt.chain`.
//
//     memref.global @global attributes { [gpu_module = @module] }
//
// will be rewritten to
//
//     func @global(%arg0: !tfrt_gpu.context) -> !tfrt_gpu.buffer, !tfrt.chain {
//       ...
//     }
//
// The function returns the GPU module symbol if the `gpu_module` attribute is
// present. Otherwise it allocates memory.
struct ConvertMemrefGlobalPattern : OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      memref::GlobalOp global_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// Converts `gpu.launch_func` op to `tfrt_gpu.function.launch` op.
struct ConvertLaunchFuncPattern : OpConversionPattern<mlir::gpu::LaunchFuncOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      mlir::gpu::LaunchFuncOp launch_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// Folds `unrealized_conversion_cast(constant ? : index) : index to ui32`.
struct FoldConstCastPattern : OpConversionPattern<CastOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      CastOp cast_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// Moves the body of a tfrt_gpu_conversion.async.execute op into the parent
// block and removes the op.
//
//     %t0 = unrealized_conversion_cast %ch0, %stream : !gpu.async.token
//     %t1 = tfrt_gpu_conversion.async.execute [%t0] {
//       ^bb(%0 : !tfrt.chain, %1 : !tfrt_gpu.stream)
//       ... ops using %0 and %1 ...
//       tfrt.return %n : !tfrt.chain
//     }
//
// will be rewritten to
//
//     %t0 = unrealized_conversion_cast %ch0, %stream : !gpu.async.token
//     ... ops using %ch0 and %stream ...
//     %t1 = unrealized_conversion_cast %n, %stream : !gpu.async.token
//
struct InlineConversionAsyncExecPattern
    : public OpConversionPattern<conversion::AsyncExecuteOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      conversion::AsyncExecuteOp exec_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// A rewrite pattern to convert gpu.wait operations to streams and events.
//
//     %t0 = unrealized_conversion_cast %ch0, %stream0
//     %t1 = unrealized_converstion_cast %ch1, %event0
//
//     %t2 = gpu.wait async [%t0]
//     %t3 = gpu.wait async [%t1]
//     %t4 = gpu.wait async [%t0, %t1]
//
// will be rewritten to
//
//     %t0 = unrealized_conversion_cast %ch0, %stream0
//     %t1 = unrealized_converstion_cast %ch1, %event0
//
//     // %t2 is replaced with %t0
//     %t2      = %t0
//     // %t3 is casted from a new stream synchronized with %event0
//     %ctx     = tfrt_gpu.stream.get_context %stream0
//     %stream1 = tfrt_gpu.stream.create %ctx
//     %ch2     = tfrt_gpu.stream.wait %stream1, %event0, %ch1
//     %t3      = unrealized_conversion_cast %ch2, %stream1
//     // %t4 is casted from %stream0 synchronized with %event0
//     %ch3     = tfrt_gpu.merge_chains %ch0, %ch1
//     %ch4     = tfrt_gpu.stream.wait %stream0, %event0, %ch3
//     %t4      = unrealized_conversion_cast %ch4, %stream0
//
// All uses outside of the current block or as terminator operand are replaced
// by a cast from an event.
//
//     %t0 = unrealized_conversion_cast %ch0, %stream0
//     ... op using %t0 ...
//     return %t0
//
// will be rewritten to
//
//     %t0 = unrealized_conversion_cast %ch0, %stream
//     %ch1, %event = unrealized_conversion_cast %t0
//     %t1 = unrealized_conversion_cast %ch1, %stream
//     %t2 = unrealized_conversion_cast %ch1, %event
//     ... op using %t1 ...
//     return %t2
//
struct ConvertGpuWaitToChainAndStreamPattern
    : public OpConversionPattern<mlir::gpu::WaitOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      mlir::gpu::WaitOp wait_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// A rewriter pattern to convert a nested cast from stream to event into a
// recorded event.
//
//     %t           = unrealized_conversion_cast %ch0, %stream
//     %ch1, %event = unrealized_conversion_cast %t
//
// will be rewritten to
//
//     %ctx   = tfrt_gpu.stream.get_context %stream
//     %event = tfrt_gpu.event.create
//     %ch1   = tfrt_gpu.event.record %event, %stream, %ch0
//
struct ConvertCastToEventRecordPattern : public OpConversionPattern<CastOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      CastOp cast_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// A rewrite pattern to convert async.execute operations to tfrt_test.do.async.
// The !async.token values have no meaning with non-strict execution and we
// simply drop them. This means that side-effecting ops need to by synchronized
// through one of the !async.value<> arguments.
//
//     y0 = ... : Y
//     %a1, %f1 = async.execute [%a0] (
//       %f0 as %x0: !async.value<X>
//     ) -> !async.value<X> {
//       ... %y0
//       async.yield %x0 : X
//     }
//
// will be rewritten to
//
//     y0 = ... : Y
//     %x1 = tfrt_test.do.async %x0, %y0 : (X, Y) -> (X) {
//       ... %c0
//       tfrt.return %x0 : X
//     }
//
struct ConvertAsyncExecToDoAsyncPattern
    : public OpConversionPattern<async::ExecuteOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      async::ExecuteOp exec_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// A rewrite pattern to remove async.await operations.
struct FoldAsyncAwaitPattern : public OpConversionPattern<async::AwaitOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      async::AwaitOp await_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// A rewrite pattern to hoist tfrt_gpu.blas/dnn/solver.create operations.
struct HoistCreateHandlePattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern::OpRewritePattern;

 private:
  LogicalResult matchAndRewrite(FuncOp func_op,
                                PatternRewriter &rewriter) const override;
};

// A pass which rewrites a function to take extra !tfrt.chain and
// !tfrt_gpu.stream arguments and return a !tfrt.chain.
struct AddChainAndStreamToFuncPass
    : public PassWrapper<AddChainAndStreamToFuncPass, OperationPass<FuncOp>> {
 private:
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<GpuDialect, compiler::TFRTDialect>();
  }
  StringRef getArgument() const override { return "func-tfrt-streamify"; }
};

// A pass which rewrites !async.execute and related ops to use !tfrt.chain and
// !tfrt_gpu.stream instead of !gpu.async.token.
struct ConvertAsyncToChainAndEventPass
    : public PassWrapper<ConvertAsyncToChainAndEventPass,
                         OperationPass<FuncOp>> {
 private:
  void runOnOperation() override;
  StringRef getArgument() const override { return "async-tfrt-streamify"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tfrt::gpu::GpuDialect, func::FuncDialect,
                    tfrt::compiler::TFRTDialect>();
  }
};

// A pass which converts from gpu dialect to tfrt_gpu dialect.
struct ConvertGpuToTfrtGpuPass
    : public PassWrapper<ConvertGpuToTfrtGpuPass, OperationPass<ModuleOp>> {
 private:
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tfrt::dht::DenseHostTensorDialect>();
  }
  StringRef getArgument() const override { return "gpu-tfrt-streamify"; }
};

// A pass which outlines resource creating ops and replaces them with tfrt.once.
struct HoistingPass
    : public PassWrapper<HoistingPass, OperationPass<ModuleOp>> {
 private:
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<compiler::TFRTDialect>();
  }
  StringRef getArgument() const override { return "gpu-tfrt-hoisting"; }
};

// A pass which removes unrealized_conversion_cast ops.
struct ReconcileCastsPass
    : public PassWrapper<ReconcileCastsPass, OperationPass<ModuleOp>> {
 private:
  void runOnOperation() override;
  StringRef getArgument() const override { return "cast-tfrt-streamify"; }
};

// A pass which converts from async dialect to tfrt dialect.
struct ConvertAsyncToTfrtPass
    : public PassWrapper<ConvertAsyncToTfrtPass, OperationPass<FuncOp>> {
 private:
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<test::TestDialect>();
  }
  StringRef getArgument() const override { return "async-to-tfrt"; }
};

}  // namespace

template <typename... Ts>
static std::array<Type, sizeof...(Ts)> GetTypes(OpBuilder &builder) {
  return {builder.getType<Ts>()...};
}

// Helper functions to unrealized_conversion_cast to statically known types.
template <typename T>
static Value CastTo(OpBuilder &builder, Location loc, ValueRange values) {
  return builder.create<CastOp>(loc, builder.getType<T>(), values).getResult(0);
}
template <typename T>
static ValueRange CastToChainAnd(OpBuilder &builder, Location loc,
                                 Value value) {
  Type types[] = {builder.getType<compiler::ChainType>(), builder.getType<T>()};
  return builder.create<CastOp>(loc, types, value).getResults();
}
const auto CastToToken = CastTo<mlir::gpu::AsyncTokenType>;
const auto CastToChainAndStream = CastToChainAnd<StreamType>;
const auto CastToChainAndEvent = CastToChainAnd<EventType>;

// Helper functions test TypeRange against a list of statically known types.
template <typename... Ts, std::size_t... Is>
static bool IsTypesImpl(TypeRange types, std::index_sequence<Is...>) {
  // TODO(csigg): Replace with fold expression once we can use C++17.
  return llvm::all_of(std::initializer_list<bool>{types[Is].isa<Ts>()...},
                      [](bool result) { return result; });
}
template <typename... Ts>
static bool IsTypes(TypeRange types) {
  if (types.size() != sizeof...(Ts)) return false;
  return IsTypesImpl<Ts...>(types, std::make_index_sequence<sizeof...(Ts)>());
}
const auto IsTokenType = IsTypes<mlir::gpu::AsyncTokenType>;

// Helper function to test whether cast is between !gpu.async.token and
// !tfrt.chain plus !tfrt_gpu.stream/event.
template <typename T>
static bool IsCastFromChainAnd(CastOp cast_op) {
  return cast_op && IsTokenType(cast_op.getResultTypes()) &&
         IsTypes<compiler::ChainType, T>(cast_op.getOperandTypes());
}
const auto IsCastFromChainAndEvent = IsCastFromChainAnd<EventType>;
const auto IsCastFromChainAndStream = IsCastFromChainAnd<StreamType>;
template <typename T>
static bool IsCastToChainAnd(CastOp cast_op) {
  return cast_op && IsTokenType(cast_op.getOperandTypes()) &&
         IsTypes<compiler::ChainType, T>(cast_op.getResultTypes());
}
const auto IsCastToChainAndEvent = IsCastToChainAnd<EventType>;

// Helper function to merge two ranges into a SmallVector.
template <typename R1, typename R2>
auto MergeRanges(R1 first, R2 second) {
  using T = typename std::iterator_traits<typename R1::iterator>::value_type;
  SmallVector<T, 8> result;
  result.reserve(first.size() + second.size());
  llvm::copy(first, std::back_inserter(result));
  llvm::copy(second, std::back_inserter(result));
  return result;
}

FailureOr<OneToAnyConversion> OneToAnyConversion::Get(TypeConverter *converter,
                                                      TypeRange source_types) {
  TypeConverter::SignatureConversion conversion(source_types.size());
  if (failed(converter->convertSignatureArgs(source_types, conversion)))
    return failure();
  return OneToAnyConversion(source_types, conversion);
}

// Inserts casts of legal-typed target_values back to source_types.
SmallVector<Value, 4> OneToAnyConversion::CastToSourceTypes(
    OpBuilder &builder, Location loc, ValueRange target_values) {
  SmallVector<Value, 4> results;
  llvm::transform(
      llvm::enumerate(source_types_), std::back_inserter(results),
      [&](const auto &pair) {
        auto mapping =
            conversion_.getInputMapping(pair.index())
                .getValueOr(TypeConverter::SignatureConversion::InputMapping{});
        if (mapping.replacementValue) return mapping.replacementValue;
        auto operands = target_values.take_front(mapping.size);
        target_values = target_values.drop_front(mapping.size);
        if (mapping.size == 1 && operands.front().getType() == pair.value())
          return operands.front();
        auto cast_op = builder.create<CastOp>(loc, pair.value(), operands);
        return cast_op.getResult(0);
      });
  return results;
}

// Inserts casts of illegal-typed source_values to converted types.
SmallVector<Value, 4> OneToAnyConversion::CastToTargetTypes(
    OpBuilder &builder, Location loc, ValueRange source_values) {
  SmallVector<Value, 4> results;
  for (const auto &pair : llvm::enumerate(source_values)) {
    auto mapping = conversion_.getInputMapping(pair.index());
    if (!mapping) continue;  // Argument was dropped.
    if (mapping->replacementValue) results.push_back(mapping->replacementValue);
    assert(mapping->size != 0);
    auto types = GetTargetTypes().slice(mapping->inputNo, mapping->size);
    if (types.size() == 1 && types.front() == pair.value().getType()) {
      results.push_back(pair.value());
    } else {
      auto cast_op = builder.create<CastOp>(loc, types, pair.value());
      llvm::copy(cast_op->getResults(), std::back_inserter(results));
    }
  }
  return results;
}

LogicalResult AddChainAndStreamToFuncPattern::matchAndRewrite(
    FuncOp func_op, PatternRewriter &rewriter) const {
  // Collect `gpu.wait [...]` and `gpu.wait async []` ops.
  SmallVector<mlir::gpu::WaitOp, 4> wait_ops;
  func_op.walk([&](mlir::gpu::WaitOp op) {
    if (!op.asyncToken() || op.asyncDependencies().empty())
      wait_ops.push_back(op);
  });

  if (wait_ops.size() < 2)
    return rewriter.notifyMatchFailure(func_op, "expected at least 2 gpu.wait");
  if (llvm::find_if(wait_ops, [](mlir::gpu::WaitOp op) {
        return !op.asyncToken();
      }) != wait_ops.end() - 1) {
    return rewriter.notifyMatchFailure(
        func_op, "expected all but the last gpu.wait to be async");
  }

  // Add !tfrt.chain, !tfrt_gpu.stream arguments and !tfrt.chain result.
  rewriter.updateRootInPlace(func_op, [&] {
    auto types = GetTypes<compiler::ChainType, StreamType>(rewriter);
    func_op.insertArguments(
        {0, 0}, types, /*argAttrs=*/{},
        SmallVector<Location, 2>(types.size(), func_op.getLoc()));
    func_op.insertResult(0, types.front(), /*resultAttrs=*/nullptr);
  });

  // Cast new arguments to token and insert wait async op.
  // %t0 = unrealized_conversion_cast %arg0, %arg1 -> !gpu.async.token
  // %t1 = gpu.wait async [%t0]
  Location loc = func_op.getLoc();
  rewriter.setInsertionPointToStart(&func_op.getBody().front());
  Value token =
      CastToToken(rewriter, loc, func_op.getArguments().take_front(2));
  auto first_wait_op =
      rewriter.create<mlir::gpu::WaitOp>(loc, token.getType(), token);

  // Add %t1 from above to all `gpu.wait async []` ops.
  for (auto op : makeArrayRef(wait_ops).drop_back())
    op.addAsyncDependency(first_wait_op.asyncToken());

  // Make `gpu.wait [...]` async, cast result and add chain to returned
  // values.
  Operation *terminator = func_op.getBody().back().getTerminator();
  rewriter.setInsertionPoint(terminator);
  auto last_wait_op = rewriter.create<mlir::gpu::WaitOp>(
      wait_ops.back().getLoc(), token.getType(),
      wait_ops.back().asyncDependencies());
  rewriter.eraseOp(wait_ops.back());
  auto chain_and_stream = CastToChainAndStream(rewriter, last_wait_op.getLoc(),
                                               last_wait_op.asyncToken());
  auto results =
      MergeRanges(chain_and_stream.take_front(), terminator->getOperands());
  rewriter.replaceOpWithNewOp<compiler::ReturnOp>(terminator, results);

  return success();
}

LogicalResult ConvertAsyncExecToChainAndEventPattern::matchAndRewrite(
    async::ExecuteOp exec_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = exec_op->getLoc();

  auto operand_conversion =
      OneToAnyConversion::Get(typeConverter, TypeRange(adaptor.operands()));
  auto result_conversion =
      OneToAnyConversion::Get(typeConverter, exec_op.getResultTypes());
  auto argument_conversion = OneToAnyConversion::Get(
      typeConverter, exec_op.getRegion().getArgumentTypes());
  auto terminator_conversion = OneToAnyConversion::Get(
      typeConverter,
      exec_op.getRegion().back().getTerminator()->getOperandTypes());

  if (failed(operand_conversion) || failed(result_conversion) ||
      failed(argument_conversion) || failed(terminator_conversion))
    return rewriter.notifyMatchFailure(exec_op, "failed to convert types");

  // Create new async.execute op with converted operands.
  auto new_op = rewriter.create<mlir::async::ExecuteOp>(
      loc, terminator_conversion->GetTargetTypes(), adaptor.dependencies(),
      operand_conversion->CastToTargetTypes(rewriter, loc, adaptor.operands()));

  // Convert new results back to invalid types.
  rewriter.replaceOp(exec_op, result_conversion->CastToSourceTypes(
                                  rewriter, loc, new_op.getResults()));

  OpBuilder::InsertionGuard guard(rewriter);

  // Convert region arguments back to invalid types.
  Region *region = &new_op.getRegion();
  rewriter.setInsertionPointToEnd(&region->front());
  auto arguments = argument_conversion->CastToSourceTypes(
      rewriter, loc, region->getArguments());

  // Clone original body into the new region.
  BlockAndValueMapping mapping;
  rewriter.cloneRegionBefore(exec_op.getRegion(), *region, region->end(),
                             mapping);
  rewriter.mergeBlocks(&region->back(), &region->front(), arguments);

  return success();
}

LogicalResult ConvertAsyncYieldToChainAndEventPattern::matchAndRewrite(
    async::YieldOp yield_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto operands = adaptor.getOperands();
  auto conversion = OneToAnyConversion::Get(typeConverter, TypeRange(operands));
  if (failed(conversion))
    return rewriter.notifyMatchFailure(yield_op, "failed to convert types");
  rewriter.replaceOpWithNewOp<mlir::async::YieldOp>(
      yield_op,
      conversion->CastToTargetTypes(rewriter, yield_op->getLoc(), operands));
  return success();
}

LogicalResult SwapAsyncAwaitOfCastPattern::matchAndRewrite(
    async::AwaitOp await_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto cast_op = adaptor.operand().getDefiningOp<CastOp>();
  if (!cast_op || !llvm::all_of(cast_op->getOperandTypes(), [](Type type) {
        return type.isa<async::ValueType>();
      }))
    return rewriter.notifyMatchFailure(await_op, "operand not def by cast");

  Location loc = await_op->getLoc();
  SmallVector<Value, 4> results;
  for (auto operand : cast_op->getOperands()) {
    results.push_back(rewriter.create<async::AwaitOp>(loc, operand).result());
  }
  rewriter.replaceOp(await_op, CastToToken(rewriter, loc, results));
  return success();
}

LogicalResult ConvertMemsetPattern::matchAndRewrite(
    mlir::gpu::MemsetOp memset_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (adaptor.value().getType().getIntOrFloatBitWidth() != 32)
    return rewriter.notifyMatchFailure(memset_op, "expected 32bit value");
  if (!adaptor.dst().getType().isa<tfrt::gpu::BufferType>())
    return rewriter.notifyMatchFailure(memset_op, "expected buffer dst");
  if (adaptor.asyncDependencies().empty() || !memset_op.asyncToken())
    return rewriter.notifyMatchFailure(memset_op, "no async deps or no result");
  auto cast_op = adaptor.asyncDependencies().front().getDefiningOp<CastOp>();
  if (!IsCastFromChainAndStream(cast_op))
    return rewriter.notifyMatchFailure(memset_op, "operand not def by cast");

  auto loc = memset_op->getLoc();
  auto stream = cast_op.getOperand(1);
  auto new_op = rewriter.create<tfrt::gpu::MemSetOp>(
      loc, adaptor.dst(), adaptor.value(), stream, cast_op.getOperand(0));
  auto token = CastToToken(rewriter, loc, {new_op, stream});
  rewriter.replaceOp(memset_op, token);
  return success();
}

LogicalResult ConvertMemcpyPattern::matchAndRewrite(
    mlir::gpu::MemcpyOp memcpy_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!adaptor.src().getType().isa<tfrt::gpu::BufferType>() ||
      !adaptor.dst().getType().isa<tfrt::gpu::BufferType>())
    return rewriter.notifyMatchFailure(memcpy_op, "expected buffer operands");
  if (adaptor.asyncDependencies().empty() || !memcpy_op.asyncToken())
    return rewriter.notifyMatchFailure(memcpy_op, "no async deps or no result");
  auto cast_op = adaptor.asyncDependencies().front().getDefiningOp<CastOp>();
  if (!IsCastFromChainAndStream(cast_op))
    return rewriter.notifyMatchFailure(memcpy_op, "operand not def by cast");

  // Drop copy if memref type has zero elements.
  if (!memcpy_op.dst().getType().cast<mlir::MemRefType>().getNumElements()) {
    rewriter.replaceOp(memcpy_op, cast_op.getResults());
    return success();
  }

  Location loc = memcpy_op->getLoc();
  Value stream = cast_op.getOperand(1);
  Value chain = rewriter.create<tfrt::gpu::MemCopyOp>(
      loc, adaptor.dst(), adaptor.src(), stream, cast_op.getOperand(0));
  auto token = CastToToken(rewriter, loc, {chain, stream});
  rewriter.replaceOp(memcpy_op, token);
  return success();
}

// Returns !tfrt_gpu.context of the parent function's stream argument.
// Inserts tfrt_gpu.stream.get_context if it doesn't already exist.
Value GetContextFromParentFunc(ConversionPatternRewriter &rewriter,
                               Operation *op) {
  auto func_op = op->getParentOfType<FuncOp>();
  auto get_ctx_ops = func_op.getOps<StreamGetContextOp>();
  if (!get_ctx_ops.empty()) return *get_ctx_ops.begin();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&func_op.front());
  Value stream = func_op.getArgument(1);
  return rewriter.create<StreamGetContextOp>(op->getLoc(), stream);
}

LogicalResult ConvertGetGlobalPattern::matchAndRewrite(
    memref::GetGlobalOp get_global_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = get_global_op->getLoc();
  Value context = GetContextFromParentFunc(rewriter, get_global_op);
  auto once_op = rewriter.create<compiler::OnceOp>(
      loc, rewriter.getType<BufferType>(), context, adaptor.name());
  auto cast_op = rewriter.create<CastOp>(loc, get_global_op.getType(),
                                         once_op.getResults());
  rewriter.replaceOp(get_global_op, cast_op.getResults());
  return success();
}

static std::string GetDenseHostTensorTypeName(Type type) {
  if (type.isInteger(1)) return "bool";
  if (type.isSignedInteger() || type.isSignlessInteger())
    return "i" + std::to_string(type.getIntOrFloatBitWidth());
  if (type.isUnsignedInteger())
    return "ui" + std::to_string(type.getIntOrFloatBitWidth());
  if (type.isBF16()) return "bf16";
  if (type.isa<FloatType>())
    return "f" + std::to_string(type.getIntOrFloatBitWidth());
  if (auto complex_type = type.dyn_cast<ComplexType>()) {
    if (complex_type.getElementType().isF32()) return "complex64";
    if (complex_type.getElementType().isF64()) return "complex128";
  }
  llvm_unreachable("Unsupported type");
}

static Value CreateMakeTensor(ConversionPatternRewriter &rewriter, Location loc,
                              StringRef suffix, MemRefType type) {
  std::string name;
  llvm::raw_string_ostream(name) << "tfrt_dht.create_uninitialized_tensor."
                                 << suffix << "." << type.getRank();
  OperationState state(loc, name);
  state.addTypes(rewriter.getType<t::TensorType>());
  state.addAttribute("shape", rewriter.getI64ArrayAttr(type.getShape()));
  return rewriter.createOperation(state)->getResult(0);
}

static Value CreateSetTensor(ConversionPatternRewriter &rewriter, Location loc,
                             StringRef suffix, Value tensor, Value chain,
                             DenseElementsAttr init_values) {
  std::string name;
  llvm::raw_string_ostream(name)
      << "tfrt_dht.set_tensor_with_constant_values." << suffix;
  OperationState state(loc, name);
  state.addTypes(chain.getType());
  state.addOperands({tensor, chain});
  std::vector<Attribute> values;
  values.reserve(init_values.getNumElements());
  llvm::copy(init_values.getValues<Attribute>(), std::back_inserter(values));
  state.addAttribute("values", rewriter.getArrayAttr(values));
  return rewriter.createOperation(state)->getResult(0);
}

// Initializes 'buffer' with constant values from 'global_op'. Returns chain.
static Value CreateSetGlobal(ConversionPatternRewriter &rewriter,
                             memref::GlobalOp global_op, Value buffer,
                             Value chain, Value stream) {
  Location loc = global_op->getLoc();
  auto init_attr = global_op.initial_valueAttr().cast<DenseElementsAttr>();
  auto type_name = GetDenseHostTensorTypeName(init_attr.getElementType());
  Value tensor = CreateMakeTensor(rewriter, loc, type_name, global_op.type());
  chain = CreateSetTensor(rewriter, loc, type_name, tensor, chain, init_attr);
  Type buffer_type = rewriter.getType<ht::HostBufferType>();
  auto buffer_op = rewriter.create<dht::GetBufferOp>(
      loc, buffer_type, chain.getType(), tensor, chain);
  return rewriter.create<MemCopyOp>(loc, buffer, buffer_op.getResult(0), stream,
                                    chain);
}

LogicalResult ConvertGpuModulePattern::matchAndRewrite(
    mlir::gpu::GPUModuleOp module_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto data = module_op->getAttrOfType<StringAttr>(getGpuBinaryAttrName());
  if (!data)
    return rewriter.notifyMatchFailure(module_op, "no device code attribute");
  Location loc = module_op->getLoc();
  auto constants =
      module_op->getAttrOfType<ArrayAttr>(getGpuConstantsAttrName());
  mlir::FunctionType func_type = rewriter.getFunctionType(
      rewriter.getType<ContextType>(), rewriter.getType<ModuleType>());
  FuncOp func_op = rewriter.replaceOpWithNewOp<FuncOp>(
      module_op, module_op.getName(), func_type);
  rewriter.setInsertionPointToEnd(func_op.addEntryBlock());
  Value context = func_op.getArgument(0);
  std::string binary = data.getValue().str();  // Add trailing zero.
  Value module = rewriter.create<ModuleLoadOp>(
      loc, context, StringRef(binary.data(), binary.size() + 1));
  if (constants) {
    // Initialize GPU module symbols.
    Value chain = rewriter.create<compiler::NewChainOp>(loc);
    Value stream = rewriter.create<StreamCreateOp>(loc, context);
    auto cast = [](Attribute attr) { return attr.cast<FlatSymbolRefAttr>(); };
    for (auto constant : llvm::map_range(constants, cast)) {
      auto global_op = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
          func_op, constant);
      Value buffer =
          rewriter.create<ModuleGetGlobalOp>(loc, module, constant.getAttr());
      chain = CreateSetGlobal(rewriter, global_op, buffer, chain, stream);
    }
    chain = rewriter.create<StreamSynchronizeOp>(loc, stream, chain);
    module = rewriter.create<AliasOp>(loc, module.getType(), module, chain);
  }
  rewriter.create<compiler::ReturnOp>(loc, module);
  return success();
}

LogicalResult ConvertMemrefGlobalPattern::matchAndRewrite(
    memref::GlobalOp global_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  mlir::FunctionType func_type = rewriter.getFunctionType(
      rewriter.getType<ContextType>(), rewriter.getType<BufferType>());
  auto func_op = rewriter.replaceOpWithNewOp<FuncOp>(
      global_op, global_op.sym_name(), func_type);
  rewriter.setInsertionPointToEnd(func_op.addEntryBlock());
  Location loc = global_op->getLoc();
  Value context = func_op.getArgument(0);

  // If the global is a GPU module symbol, return that.
  if (auto module_attr =
          global_op->getAttrOfType<FlatSymbolRefAttr>(getGpuModuleAttrName())) {
    auto once_op = rewriter.create<compiler::OnceOp>(
        loc, rewriter.getType<ModuleType>(), ValueRange(context), module_attr);
    Value buffer = rewriter.create<ModuleGetGlobalOp>(
        loc, once_op->getResult(0), adaptor.sym_name());
    rewriter.create<compiler::ReturnOp>(loc, buffer);
    return success();
  }

  // Otherwise, allocate memory and initialize it.
  Value allocator = rewriter.create<AllocatorCreateOp>(loc, context);
  Value stream = rewriter.create<StreamCreateOp>(loc, context);
  uint64_t size_bytes =
      (global_op.type().cast<ShapedType>().getSizeInBits() + 7) / 8;
  Value size = rewriter.create<compiler::ConstantI64Op>(loc, size_bytes);
  Value chain = rewriter.create<compiler::NewChainOp>(loc);
  Value buffer =
      rewriter.create<gpu::MemAllocateOp>(loc, allocator, stream, size, chain);
  chain = CreateSetGlobal(rewriter, global_op, buffer, chain, stream);
  chain = rewriter.create<StreamSynchronizeOp>(loc, stream, chain);
  buffer = rewriter.create<AliasOp>(loc, buffer.getType(), buffer, chain);
  rewriter.create<compiler::ReturnOp>(loc, buffer);
  return success();
}

LogicalResult ConvertLaunchFuncPattern::matchAndRewrite(
    mlir::gpu::LaunchFuncOp launch_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (adaptor.asyncDependencies().empty() || !launch_op.asyncToken())
    return rewriter.notifyMatchFailure(launch_op, "no async deps or no result");
  auto cast_op = adaptor.asyncDependencies().front().getDefiningOp<CastOp>();
  if (!IsCastFromChainAndStream(cast_op))
    return rewriter.notifyMatchFailure(launch_op, "operand not def by cast");

  Location loc = launch_op->getLoc();
  Value chain = cast_op.getOperand(0);
  Value stream = cast_op.getOperand(1);
  Value context = GetContextFromParentFunc(rewriter, launch_op);
  auto func_op = SymbolTable::lookupNearestSymbolFrom<FuncOp>(
      launch_op, adaptor.kernel().getRootReference());
  auto once_op = rewriter.create<compiler::OnceOp>(
      loc, func_op.getType().getResults(), context, func_op.getName());
  auto kernel_name = adaptor.kernel().getLeafReference().getValue();
  auto get_func_op = rewriter.create<ModuleGetFunctionOp>(
      loc, once_op->getResult(0), kernel_name);
  if (once_op.getNumResults() > 1) {
    chain = rewriter.create<compiler::MergeChainsOp>(
        loc, chain.getType(), ValueRange({chain, once_op->getResult(1)}));
  }
  Value shared_mem_size = adaptor.dynamicSharedMemorySize();
  if (!shared_mem_size)
    shared_mem_size = rewriter.create<compiler::ConstantUI32Op>(loc, 0);
  auto cast_to_ui32 = [&](Value value) {
    return typeConverter->materializeTargetConversion(
        rewriter, loc, rewriter.getIntegerType(32, /*isSigned=*/false), value);
  };
  auto new_op = rewriter.create<FunctionLaunchOp>(
      loc, chain.getType(), stream, get_func_op.getResult(),
      cast_to_ui32(adaptor.gridSizeX()), cast_to_ui32(adaptor.gridSizeY()),
      cast_to_ui32(adaptor.gridSizeZ()), cast_to_ui32(adaptor.blockSizeX()),
      cast_to_ui32(adaptor.blockSizeY()), cast_to_ui32(adaptor.blockSizeZ()),
      cast_to_ui32(shared_mem_size), chain, adaptor.operands());
  rewriter.replaceOp(launch_op, CastToToken(rewriter, loc, {new_op, stream}));
  return success();
}

Value materializeConstCast(OpBuilder &builder, ValueRange operands,
                           Type resultType, Location loc) {
  if (!IsTypes<IndexType>(operands.getTypes()) ||
      !resultType.isa<IntegerType>())
    return Value();
  auto const_op = operands.front().getDefiningOp<arith::ConstantOp>();
  if (!const_op) return Value();
  auto type = resultType.cast<IntegerType>();
  auto rewrite = [&](auto dummy) -> Value {
    APInt value = const_op.getValue().cast<IntegerAttr>().getValue();
    if (type.isUnsigned())
      value = value.zextOrTrunc(type.getWidth());
    else
      value = value.sextOrTrunc(type.getWidth());
    auto attr = builder.getIntegerAttr(type, value);
    return builder.create<decltype(dummy)>(loc, type, attr);
  };
  if (type.isUnsignedInteger(32)) return rewrite(compiler::ConstantUI32Op());
  if (type.isUnsignedInteger(64)) return rewrite(compiler::ConstantUI64Op());
  if (type.isInteger(32)) return rewrite(compiler::ConstantI32Op());
  if (type.isInteger(64)) return rewrite(compiler::ConstantI64Op());
  return Value();
}

LogicalResult FoldConstCastPattern::matchAndRewrite(
    CastOp cast_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!IsTypes<IndexType>(cast_op.getOperandTypes()) ||
      !IsTypes<IntegerType>(cast_op.getResultTypes()))
    return rewriter.notifyMatchFailure(cast_op, "not cast from index to int");
  auto const_op = cast_op.getOperand(0).getDefiningOp<arith::ConstantOp>();
  if (!const_op)
    return rewriter.notifyMatchFailure(cast_op, "operand not def by constant");
  auto type = cast_op.getType(0).cast<IntegerType>();
  auto rewrite = [&](auto dummy) {
    APInt value = const_op.getValue().cast<IntegerAttr>().getValue();
    if (type.isUnsigned())
      value = value.zextOrTrunc(type.getWidth());
    else
      value = value.sextOrTrunc(type.getWidth());
    auto attr = rewriter.getIntegerAttr(type, value);
    rewriter.replaceOpWithNewOp<decltype(dummy)>(cast_op, type, attr);
    return success();
  };
  if (type.isUnsignedInteger(32)) return rewrite(compiler::ConstantUI32Op());
  if (type.isUnsignedInteger(64)) return rewrite(compiler::ConstantUI64Op());
  if (type.isInteger(32)) return rewrite(compiler::ConstantI32Op());
  if (type.isInteger(64)) return rewrite(compiler::ConstantI64Op());
  return rewriter.notifyMatchFailure(cast_op, "Unsupported type");
}

LogicalResult InlineConversionAsyncExecPattern::matchAndRewrite(
    conversion::AsyncExecuteOp exec_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (adaptor.asyncDependencies().empty() || !exec_op.getAsyncToken())
    return rewriter.notifyMatchFailure(exec_op, "no async deps or no result");
  auto cast_op = adaptor.asyncDependencies().front().getDefiningOp<CastOp>();
  if (!IsCastFromChainAndStream(cast_op))
    return rewriter.notifyMatchFailure(exec_op, "operand not def by cast");

  // Merge !tfrt_gpu_conversion.async.execute body into parent block.
  Operation *terminator = exec_op.getBody()->getTerminator();
  rewriter.mergeBlockBefore(exec_op.getBody(), exec_op, cast_op.getOperands());
  auto chain_and_stream = {terminator->getOperand(0), cast_op.getOperand(1)};
  auto token = CastToToken(rewriter, exec_op->getLoc(), chain_and_stream);
  rewriter.replaceOp(exec_op, token);
  rewriter.eraseOp(terminator);
  return success();
}

LogicalResult ConvertGpuWaitToChainAndStreamPattern::matchAndRewrite(
    mlir::gpu::WaitOp wait_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto operands = adaptor.getOperands();
  if (operands.empty() || !wait_op.asyncToken())
    return rewriter.notifyMatchFailure(wait_op, "no operands or not async");
  CastOp cast_from_stream_op;
  SmallVector<CastOp, 2> cast_from_event_ops;
  for (auto operand : operands) {
    CastOp cast_op = operand.getDefiningOp<CastOp>();
    if (IsCastFromChainAndEvent(cast_op)) {
      cast_from_event_ops.push_back(cast_op);
      continue;
    }
    if (IsCastFromChainAndStream(cast_op)) {
      if (cast_from_stream_op)
        return rewriter.notifyMatchFailure(wait_op, "more than one stream");
      cast_from_stream_op = cast_op;
      continue;
    }
    return rewriter.notifyMatchFailure(wait_op, "operand not def by cast");
  }

  // Merge operand chains if there is more than one.
  Location loc = wait_op.getLoc();
  Value chain = [&]() -> Value {
    SmallVector<Value, 4> chains;
    if (cast_from_stream_op)
      chains.push_back(cast_from_stream_op.getOperand(0));
    llvm::transform(cast_from_event_ops, std::back_inserter(chains),
                    [](auto cast_op) { return cast_op.getOperand(0); });
    if (chains.size() == 1) return chains.front();
    Type chain_type = rewriter.getType<compiler::ChainType>();
    return rewriter.create<compiler::MergeChainsOp>(loc, chain_type, chains);
  }();

  // Create stream if no operand is cast from stream.
  Value stream = [&]() -> Value {
    if (cast_from_stream_op) return cast_from_stream_op.getOperand(1);
    // Use stream block argument if it exists.
    for (auto argument : wait_op->getBlock()->getArguments())
      if (argument.getType().isa<StreamType>()) return argument;
    Value context = GetContextFromParentFunc(rewriter, wait_op);
    return rewriter.create<StreamCreateOp>(loc, context);
  }();

  // Synchronize stream with all event operands.
  for (auto cast_op : cast_from_event_ops) {
    chain = rewriter.create<StreamWaitOp>(loc, stream, cast_op.getOperand(1),
                                          chain);
  }

  // Cast back to token if stream was synchronized.
  Value token = [&]() -> Value {
    if (cast_from_event_ops.empty()) return cast_from_stream_op.getResult(0);
    return CastToToken(rewriter, wait_op.getLoc(), {chain, stream});
  }();

  // Collect uses in other blocks and terminator uses.
  auto event_uses = llvm::make_filter_range(
      wait_op.asyncToken().getUses(), [&](const OpOperand &operand) {
        Operation *owner = operand.getOwner();
        if (owner->getBlock() != wait_op->getBlock()) return true;
        return owner->mightHaveTrait<OpTrait::IsTerminator>();
      });

  // Replace event uses with cast roundtrip to chain and event.
  if (!event_uses.empty()) {
    auto chain_and_event = CastToChainAndEvent(rewriter, loc, token);
    token = CastToToken(rewriter, loc, {chain_and_event.front(), stream});
    auto cast_from_event = CastToToken(rewriter, loc, chain_and_event);
    for (auto &use : event_uses) use.set(cast_from_event);
  }

  rewriter.replaceOp(wait_op, token);

  return success();
}

LogicalResult ConvertCastToEventRecordPattern::matchAndRewrite(
    CastOp cast_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!IsCastFromChainAndStream(cast_op)) {
    return rewriter.notifyMatchFailure(
        cast_op, "not cast from chain and stream to token");
  }

  if (!llvm::all_of(cast_op->getUsers(), [](Operation *op) {
        return IsCastToChainAndEvent(dyn_cast<CastOp>(op));
      })) {
    return rewriter.notifyMatchFailure(
        cast_op, "not all users are cast to chain and event");
  }

  Location loc = cast_op->getLoc();
  Value chain = adaptor.getOperands().front();
  Value stream = adaptor.getOperands().back();
  Value context = GetContextFromParentFunc(rewriter, cast_op);
  Value event = rewriter.create<EventCreateOp>(loc, context);
  chain = rewriter.create<EventRecordOp>(loc, event, stream, chain);

  for (auto user : cast_op->getUsers())
    rewriter.replaceOp(user, {chain, event});
  rewriter.eraseOp(cast_op);

  return success();
}

LogicalResult ConvertAsyncExecToDoAsyncPattern::matchAndRewrite(
    async::ExecuteOp exec_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Drop !async.token operands, they are not region arguments.
  auto operands = adaptor.operands();
  SmallVector<Value, 4> arguments(operands.begin(), operands.end());
  // Make all captures explicit arguments.
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(exec_op->getRegions(), captures);
  llvm::transform(captures, std::back_inserter(arguments), [&](Value value) {
    return rewriter.getRemappedValue(value);
  });

  SmallVector<Type, 4> arg_types, result_types;
  if (failed(typeConverter->convertTypes(TypeRange(ValueRange(arguments)),
                                         arg_types)) ||
      failed(typeConverter->convertTypes(
          TypeRange(exec_op.getResultTypes()).drop_front(), result_types))) {
    return rewriter.notifyMatchFailure(exec_op, "failed to convert types");
  }

  Location loc = exec_op->getLoc();
  auto do_op = rewriter.create<test::DoAsyncOp>(loc, result_types, arguments);
  Region *region = &do_op.getRegion();
  Block *block =
      rewriter.createBlock(region, region->end(), arg_types,
                           SmallVector<Location, 2>(arg_types.size(), loc));
  BlockAndValueMapping mapping;
  mapping.map(arguments, block->getArguments());
  rewriter.cloneRegionBefore(exec_op.getRegion(), *region, region->end(),
                             mapping);
  rewriter.mergeBlocks(block->getNextNode(), block,
                       block->getArguments().take_front(operands.size()));

  rewriter.setInsertionPoint(exec_op);  // Restore from createBlock() above.
  SmallVector<Value, 4> results = {
      CastTo<mlir::async::TokenType>(rewriter, loc, {})};
  llvm::copy(do_op.getResults(), std::back_inserter(results));
  rewriter.replaceOp(exec_op, results);

  Operation *terminator = region->back().getTerminator();
  rewriter.setInsertionPoint(terminator);
  rewriter.replaceOpWithNewOp<compiler::ReturnOp>(terminator,
                                                  terminator->getOperands());

  return success();
}

LogicalResult FoldAsyncAwaitPattern::matchAndRewrite(
    async::AwaitOp await_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (await_op->getNumResults() == 0)
    return rewriter.eraseOp(await_op), success();
  rewriter.replaceOp(await_op, adaptor.getOperands());
  return success();
}

LogicalResult HoistCreateHandlePattern::matchAndRewrite(
    FuncOp func_op, PatternRewriter &rewriter) const {
  // Check for argument type to prevent infinite recursion. This assumes that
  // no other function that just takes a context needs to hoist those ops.
  // At some point, we likely want to tag the hoisted functions to later
  // collect them and call them as part of program initialization. At that
  // point, we can use that tag to detect recursion.
  if (IsTypes<ContextType>(func_op.getType().getInputs()))
    return rewriter.notifyMatchFailure(func_op, "already hoisted");

  SmallVector<Operation *, 4> create_handle_ops;
  func_op.walk([&](Operation *op) {
    if (isa<BlasCreateOp, DnnCreateOp, SolverCreateOp>(op))
      create_handle_ops.push_back(op);
  });

  if (create_handle_ops.empty())
    return rewriter.notifyMatchFailure(func_op, "no create handle ops");

  mlir::SymbolTable symbol_table(func_op->getParentOp());
  // Map from handle type and context value to tfrt.once callee.
  llvm::SmallDenseMap<std::tuple<Type, Value>, FuncOp> map;
  for (auto *op : create_handle_ops) {
    Type handle_type = op->getResult(0).getType();
    Value context = op->getOperand(0);
    auto pair = map.try_emplace(std::make_tuple(handle_type, context), nullptr);
    if (pair.second) {
      rewriter.setInsertionPoint(func_op);
      Location loc = op->getLoc();
      auto callee_type =
          rewriter.getFunctionType(context.getType(), handle_type);
      auto callee_op = rewriter.create<mlir::FuncOp>(
          loc, op->getName().getIdentifier(), callee_type);
      symbol_table.insert(callee_op);
      rewriter.setInsertionPointToEnd(callee_op.addEntryBlock());
      BlockAndValueMapping mapper;
      mapper.map(context, callee_op.getArgument(0));
      Value handle = rewriter.clone(*op, mapper)->getResult(0);
      rewriter.create<tfrt::compiler::ReturnOp>(loc, handle);
      pair.first->second = callee_op;
    }
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<tfrt::compiler::OnceOp>(
        op, handle_type, context, pair.first->second.getSymName());
  }

  return success();
}

void AddChainAndStreamToFuncPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<AddChainAndStreamToFuncPattern>(&getContext());
  if (failed(applyOpPatternsAndFold(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

void ConvertAsyncToChainAndEventPass::runOnOperation() {
  TypeConverter converter;
  // T -> T
  converter.addConversion([](Type type) { return type; });
  // !async.value<T> -> !async.value<convert(T)>...
  converter.addConversion(
      [&](mlir::async::ValueType type, llvm::SmallVectorImpl<Type> &results) {
        if (failed(converter.convertType(type.getValueType(), results)))
          return failure();
        llvm::transform(results, results.begin(), [](Type type) {
          return mlir::async::ValueType::get(type);
        });
        return success();
      });
  // !gpu.async.token -> !tfrt.chain, !tfrt_gpu.event
  converter.addConversion(
      [](mlir::gpu::AsyncTokenType type, llvm::SmallVectorImpl<Type> &results) {
        results.push_back(compiler::ChainType::get(type.getContext()));
        results.push_back(EventType::get(type.getContext()));
        return success();
      });

  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertAsyncExecToChainAndEventPattern,
               ConvertAsyncYieldToChainAndEventPattern,
               SwapAsyncAwaitOfCastPattern>(converter, &getContext());

  ConversionTarget target(getContext());
  target.addDynamicallyLegalOp<mlir::async::AwaitOp, mlir::async::ExecuteOp,
                               mlir::async::YieldOp>(
      [&](Operation *op) { return converter.isLegal(op); });
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}

static LogicalResult ConvertGpuModuleOps(ModuleOp module_op) {
  SmallVector<Operation *, 4> gpu_module_ops;
  module_op.walk(
      [&](mlir::gpu::GPUModuleOp op) { gpu_module_ops.push_back(op); });
  if (gpu_module_ops.empty()) return success();

  RewritePatternSet patterns(module_op->getContext());
  patterns.add<ConvertGpuModulePattern>(module_op->getContext());
  return success(applyOpPatternsAndFold(gpu_module_ops, std::move(patterns),
                                        /*strict=*/true));
}

void ConvertGpuToTfrtGpuPass::runOnOperation() {
  // Rewrite `gpu.module` before rewriting the referenced `memref.global` ops.
  if (failed(ConvertGpuModuleOps(getOperation()))) return signalPassFailure();

  RewritePatternSet patterns(&getContext());
  auto converter = createMemrefToTfrtGpuConverter();
  patterns.add<ConvertMemsetPattern, ConvertMemcpyPattern,
               ConvertMemrefGlobalPattern, ConvertLaunchFuncPattern>(
      converter, &getContext());
  patterns.add<ConvertGetGlobalPattern, InlineConversionAsyncExecPattern,
               ConvertGpuWaitToChainAndStreamPattern>(&getContext());
  ConversionTarget target(getContext());
  target.addIllegalDialect<mlir::gpu::GPUDialect>();
  target.addIllegalOp<conversion::AsyncExecuteOp, memref::GetGlobalOp,
                      memref::GlobalOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}

void HoistingPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<HoistCreateHandlePattern>(&getContext());
  SmallVector<Operation *, 4> func_ops;
  llvm::copy(getOperation().getOps<FuncOp>(), std::back_inserter(func_ops));
  applyOpPatternsAndFold(func_ops, std::move(patterns), /*strict=*/false);
}

void ReconcileCastsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertCastToEventRecordPattern, FoldConstCastPattern>(
      &getContext());
  populateReconcileUnrealizedCastsPatterns(patterns);
  ConversionTarget target(getContext());
  target.addIllegalOp<CastOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}

void ConvertAsyncToTfrtPass::runOnOperation() {
  TypeConverter converter;
  // T -> T
  converter.addConversion([](Type type) { return type; });
  // !async.token -> null
  converter.addConversion([](mlir::async::TokenType type,
                             SmallVectorImpl<Type> &) { return success(); });
  // !async.value<T> -> T
  converter.addConversion([&](mlir::async::ValueType type) {
    return converter.convertType(type.getValueType());
  });

  RewritePatternSet patterns(&getContext());
  // Folds pairs of A-B-A casts before outlining async.execute regions.
  populateReconcileUnrealizedCastsPatterns(patterns);
  patterns.add<ConvertAsyncExecToDoAsyncPattern, FoldAsyncAwaitPattern>(
      converter, &getContext());

  ConversionTarget target(getContext());
  target.addIllegalOp<async::AwaitOp, async::ExecuteOp, async::YieldOp>();
  target.markUnknownOpDynamicallyLegal([&](Operation *) { return true; });

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}

static Value MaterializeCast(OpBuilder &builder, Type type, ValueRange values,
                             Location loc) {
  if (Value constCast = materializeConstCast(builder, values, type, loc))
    return constCast;
  return builder.create<CastOp>(loc, type, values).getResult(0);
}

StringRef getGpuBinaryAttrName() { return "binary"; }
StringRef getGpuConstantsAttrName() { return "constants"; }
StringRef getGpuModuleAttrName() { return "gpu_module"; }

TypeConverter createMemrefToTfrtGpuConverter() {
  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });
  converter.addConversion([&](BaseMemRefType type) {
    return tfrt::gpu::BufferType::get(type.getContext());
  });
  converter.addArgumentMaterialization(MaterializeCast);
  converter.addSourceMaterialization(MaterializeCast);
  converter.addTargetMaterialization(MaterializeCast);
  return converter;
}

void populateGpuToTfrtGpuPasses(OpPassManager &pm) {
  pm.addPass(std::make_unique<AddChainAndStreamToFuncPass>());
  pm.addPass(std::make_unique<ConvertAsyncToChainAndEventPass>());
  pm.addPass(createCanonicalizerPass());  // Remove unused `cast(get_global)`.
  pm.addPass(std::make_unique<ConvertGpuToTfrtGpuPass>());
  pm.addPass(std::make_unique<ReconcileCastsPass>());
  pm.addPass(std::make_unique<ConvertAsyncToTfrtPass>());
  pm.addPass(std::make_unique<HoistingPass>());
}

void registerPasses() {
  PassRegistration<AddChainAndStreamToFuncPass>();
  PassRegistration<ConvertAsyncToChainAndEventPass>();
  PassRegistration<ConvertGpuToTfrtGpuPass>();
  PassRegistration<ReconcileCastsPass>();
  PassRegistration<ConvertAsyncToTfrtPass>();
  PassRegistration<HoistingPass>();

  PassPipelineRegistration<>(
      "gpu-to-tfrt-gpu",
      "Pass pipeline to convert from MLIR's gpu and async dialects to TFRT.",
      [](OpPassManager &pm) { tfrt::gpu::populateGpuToTfrtGpuPasses(pm); });
}

}  // namespace gpu
}  // namespace tfrt
